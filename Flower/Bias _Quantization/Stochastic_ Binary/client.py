import torch
from torchvision import transforms, datasets
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import flwr as fl
from collections import OrderedDict

BATCH_SIZE = 32
NUM_CLIENTS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


def load_dataset():
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.MNIST(
        root='./data', train=True, download=True, transform=trf)
    val_set = datasets.MNIST(root='./data', train=False,
                             download=True, transform=trf)

    train_size = len(train_set)//NUM_CLIENTS
    val_size = len(val_set)//NUM_CLIENTS

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
    optimizer = optim.Adam(model.parameters())
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
SHAPES = [torch.tensor(i.shape) for i in [val.cpu()
                                          for _, val in model.state_dict().items()]]
trainloader, valloader = load_dataset()


def encoder(params):
    flat_params = []
    for i in params:
        flat_params.extend(torch.Tensor.flatten(i))
    flat_params = torch.tensor(flat_params)
    mini = torch.min(flat_params)
    maxi = torch.max(flat_params)
    probs = (flat_params - mini) / (maxi-mini)
    return torch.bernoulli(probs), mini, maxi


def decoder(bers, mini, maxi):
    a = ((3*maxi)+mini)/4
    b = ((3*mini)+maxi)/4
    bers = torch.where(bers == 1, a, b)
    ptr = 0
    revert = []
    for shp in SHAPES:
        hats = bers[ptr: ptr+torch.prod(shp)]
        hats = hats.reshape(tuple(shp.numpy()))
        revert.append(hats.numpy())
        ptr += torch.prod(shp)
    return revert


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print("[SENDING PARAMETERS TO SERVER]")
        params = [val.cpu() for _, val in self.model.state_dict().items()]
        bers, mini, maxi = encoder(params=params)
        param_hats = decoder(mini=mini, maxi=maxi, bers=bers)
        return param_hats

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
        loss, acc = val(self.model, valloader)
        print("[EVAL, SENDING METRICS TO SERVER]")
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc),
                                                          "losss": float(loss)}


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(model, trainloader, valloader),
    )
