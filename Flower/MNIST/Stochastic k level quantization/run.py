import subprocess
import multiprocessing

NUM_CLIENTS = 10


def run_client(_):
    subprocess.call(["python", "client.py"])


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=NUM_CLIENTS)
    pool.map(run_client, range(NUM_CLIENTS))
