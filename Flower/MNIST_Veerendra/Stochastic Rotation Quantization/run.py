import subprocess
import multiprocessing
from torchvision import datasets, transforms

NUM_CLIENTS = 10

trf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=trf)
val_set = datasets.MNIST(root='./data', train=False, download=True, transform=trf)

def run_client(_):
    subprocess.call(["python", "client.py"])


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=NUM_CLIENTS)
    pool.map(run_client, range(NUM_CLIENTS))
