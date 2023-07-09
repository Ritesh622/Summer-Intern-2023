import torch
from torchvision import transforms, datasets
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import flwr as fl
from collections import OrderedDict


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

BATCH_SIZE = 32
K = 2  # no. of levels of Quantization
print(K)
R1024 = torch.load("rotmat2pow10.pt", map_location=DEVICE).type(torch.float32)
R512 = torch.load("rotmat2pow9.pt", map_location=DEVICE).type(torch.float32)


def load_dataset():
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.MNIST(
        root='./data', train=True, download=True, transform=trf)
    val_set = datasets.MNIST(root='./data', train=False,
                             download=True, transform=trf)

    # randomly taking 6000 samples from the dataset (per client).
    train_size = len(train_set)//10
    val_size = len(val_set)//10

    train_set = random_split(
        train_set, [train_size, len(train_set)-train_size])[0]
    val_set = random_split(val_set, [val_size, len(val_set)-val_size])[0]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(model, trainloader, epochs):
    model.train(True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for X, y in tqdm(trainloader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            yhat = model(X)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()


def val(model, valloader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    crct, loss = 0, 0.0
    with torch.no_grad():
        for X, y in tqdm(valloader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            yhat = model(X)
            loss += loss_fn(yhat, y)
            crct += (torch.max(yhat.data, 1)[1] == y).sum().item()
    acc = crct / len(valloader.dataset)
    return loss, acc


model = LeNet5().to(DEVICE)
trainloader, valloader = load_dataset()


def encoder(params):

    s = torch.max(params) - torch.min(params)
    b = torch.min(params) + ((s * torch.arange(K, device=DEVICE))/(K-1))
    ids = torch.searchsorted(b, params.contiguous(), side='right')-1
    pts = torch.cat((
        torch.unsqueeze(ids, -1),
        torch.unsqueeze(ids+1, -1)
    ), axis=-1)
    pts[pts == K] = K-1
    brs = b[pts]
    probs = torch.where(brs[..., 1] != brs[..., 0],
                        (params - brs[..., 0]) / (brs[..., 1] - brs[..., 0]), 0)
    encs = torch.bernoulli(probs)
    return encs, brs


def decoder(encs_brs_1024, encs_brs_512):
    dec1024 = torch.where(
        encs_brs_1024[0] == 1,
        encs_brs_1024[1][..., 1],
        encs_brs_1024[1][..., 0]
    )
    dec1024 = torch.matmul(torch.linalg.inv(R1024), dec1024.T).T

    dec512 = torch.where(
        encs_brs_512[0] == 1,
        encs_brs_512[1][..., 1],
        encs_brs_512[1][..., 0]
    )
    dec512 = torch.matmul(torch.linalg.inv(R512), dec512)

    dec = torch.cat((torch.flatten(dec1024), dec512))

    revert = []
    ptr = 0
    for layer in model.parameters():
        size = layer.numel()
        revert.append(dec[ptr: ptr+size].reshape(layer.shape).cpu().numpy())
        ptr += size
    return revert


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print("[SENDING PARAMETERS TO SERVER]")

        flat_params = nn.utils.parameters_to_vector(
            self.model.parameters()).detach()

        params1024 = torch.split(
            flat_params[:flat_params.numel()-flat_params.numel() % 1024], 1024)
        params1024 = torch.stack(params1024)
        params1024 = torch.matmul(R1024, params1024.T).T

        params512 = flat_params[-(flat_params.numel() % 1024):]
        params512 = torch.cat((params512, torch.zeros(int(
            2**torch.ceil(torch.log2(torch.tensor(params512.numel()))) - params512.numel())).to(DEVICE)))
        params512 = torch.matmul(R512, params512)

        encs_brs_1024 = encoder(params1024)
        encs_brs_512 = encoder(params512)
        return decoder(encs_brs_1024=encs_brs_1024,
                       encs_brs_512=encs_brs_512)

    def set_parameters(self, parameters, config):
        param_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in param_dict
        })
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        local_epochs = config["local_epochs"]

        print(f"[FIT, config: {config}]")
        print("[FIT, RECEIVED PARAMETERS FROM SERVER]")

        self.set_parameters(parameters, config)
        train(self.model, self.trainloader, epochs=local_epochs)
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("[EVAL, RECEIVED PARAMETERS FROM SERVER]")
        self.set_parameters(parameters, config)
        loss, acc = val(self.model, self.valloader)
        print("[EVAL, SENDING METRICS TO SERVER]")
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc),
                                                          "losss": float(loss)}


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(model, trainloader, valloader),
    )
