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
K = 64  # no. of levels of Quantization
print(K)
R = torch.load("rotmat1024.pt", map_location=DEVICE).type(torch.float32)


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


def qunatization(params):
    # preprocess:
    params = torch.mm(params, R.T)
    # maxs, mins for each block and thus finding si
    mins = torch.min(params, axis=1, keepdims=True).values
    maxs = torch.max(params, axis=1, keepdims=True).values
    si = maxs - mins

    # quantization levels for each block
    Bi = mins + si*(torch.arange(K)/(K-1))

    # finding B(r) as per paper.
    ids = torch.searchsorted(Bi, params, side='right')-1

    # making them into (B(r+1), B(r))
    points = torch.cat([
        torch.unsqueeze(ids, -1),
        torch.unsqueeze((ids+1), -1)
    ], axis=-1)

    # marking B(r) = B(r+1) for max in each block.
    points[points == K] = K-1

    # converting indices into quantization values.
    Brs = Bi[torch.arange(Bi.shape[0]).unsqueeze(1),
             points.view(Bi.shape[0], -1)].view(points.shape)

    # finding probability for each parameter. probability of max element is 0.
    probs = torch.where(
        Brs[..., 1] != Brs[..., 0],
        (params-Brs[..., 0])/(Brs[..., 1]-Brs[..., 0]),
        0)

    # returns 1s, 0s based on paper.
    encs = torch.bernoulli(probs)

    # replacing 1s with B(r+1) and 0s with B(r) and flattenning the whole array.
    dec = torch.where(
        encs == 1,
        ((3*Brs[..., 1])+Brs[..., 0])/4,
        ((3*Brs[..., 0])+Brs[..., 1])/4
    )

    # reconstructing the parameters into their corresponding shapes.
    dec = torch.mm(params, torch.linalg.inv(R).T).flatten()
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

        # flattening the parameters
        flat_params = nn.utils.parameters_to_vector(
            self.model.parameters()).detach()

        # splitting parameters into 1024 batches each
        params = torch.cat([flat_params,
                            torch.zeros(1024 - (flat_params.numel() % 1024))]).view(-1, 1024)
        return qunatization(params)

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
