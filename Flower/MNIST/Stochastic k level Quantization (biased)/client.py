import torch
from torchvision import transforms, datasets
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import flwr as fl
from collections import OrderedDict


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

BATCH_SIZE = 32
K = 2  # no. of levels of Quantization
print(K)


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
    """An Encoding function

    Args:
        params (list): list of tensors of model paramters generator.

    Returns:
        ndarray, ndarray: an array of 1s and 0s. 
                        and an array of corresponding B(r+1)s and B(r)s.
    """

    # Flattening the parameters.
    flat_params = nn.utils.parameters_to_vector(model.parameters()).detach()

    # si (as given in the paper)
    si = torch.max(flat_params) - torch.min(flat_params)

    # Quantization levels
    Bi = torch.min(flat_params) + ((si * torch.arange(K))/(K-1))

    # B(r)
    ids = torch.searchsorted(Bi, flat_params, side='right')-1

    # stacking the points as (B(r), B(r+1))
    pts = torch.stack((ids, ids+1)).T

    # for np.max(xi) B(r+1) will be out of index.
    # replacing B(r+1) with B(r) solves the problem.
    pts[pts == K] = K-1

    # converting points to Quantization values
    # from which interval the parameter belongs to.
    brs = Bi[pts]

    # finding probabilities with which encoded value of parameter depends on.
    # giving 0 as probability for np.max(xi)
    probs = torch.where(
        (brs[:, 1] - brs[:, 0]) != 0,
        (flat_params-brs[:, 0]) / (brs[:, 1] - brs[:, 0]),
        0)

    # sending 1s and 0s using the above probailities.
    return torch.bernoulli(probs), brs


def decoder(encs, brs):
    """A decoding function.(biased)

    Args:
        encs (ndarray): an array with 1s and 0s.
        brs (ndarray): corresponding B(r+1) and B(r) values of parameters.

    Returns:
        list: list of ndarrays sent to the server.
    """

    a = ((3*brs[:, 1]) + brs[:, 0])/4
    b = ((3*brs[:, 0]) + brs[:, 1])/4
    # replacing 1s with corresponding (3B(r+1)+B(r))/4
    # and 0s with corresponding (3B(r)+B(r+1))/4.
    torch.where(encs == 1, a, b)

    # reconstructing the parameters into their
    # original shapes from flattened array.
    revert = []
    ptr = 0
    for layer in model.parameters():
        size = layer.numel()
        revert.append(encs[ptr: ptr+size].reshape(layer.shape).numpy())
        ptr += size
    return revert


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print("[SENDING PARAMETERS TO SERVER]")
        params = self.model.parameters()
        encs, brs = encoder(params)
        param_hats = decoder(brs=brs, encs=encs)
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
