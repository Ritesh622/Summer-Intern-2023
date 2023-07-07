import flwr as fl
from flwr.server.strategy import FedAvg
# from client import LeNet5


def eval_weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    ls = [m['losss'] for _, m in metrics]
    # print(ls)
    print("\t\t", sum(ls)/len(ls))
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round):
    config = {
        "server_round": server_round,
        "local_epochs": 2
    }
    return config

#
# params = [val.cpu().numpy() for _, val in LeNet5().state_dict().items()]


strategy = FedAvg(
    min_available_clients=10,
    min_fit_clients=8,
    min_evaluate_clients=3,
    fraction_fit=0.8,
    fraction_evaluate=0.4,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=eval_weighted_average,
    # initial_parameters=fl.common.ndarrays_to_parameters(params)
)

# strategy = FedAvg(evaluate_metrics_aggregation_fn=eval_weighted_average,
#                   on_fit_config_fn=fit_config)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
)
