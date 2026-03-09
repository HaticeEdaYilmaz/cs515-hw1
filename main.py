import random
import ssl
import numpy as np
import torch
import torch.nn as nn
from parameters import get_params
from models.MLP import MLP
from train import run_training
from test  import run_test
from typing import Dict, Any


# Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(params: Dict[str, Any]) -> MLP:
    """Build an MLP model using the provided parameters."""
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leakyrelu": nn.LeakyReLU
    }

    return MLP(
            input_size   = params["input_size"],
            hidden_sizes = params["hidden_sizes"],
            num_classes  = params["num_classes"],
            dropout      = params["dropout"],
            activation=activation_map[params["activation"]],
            batchnorm= params["batchnorm"]
        )

 


def main() -> None:
    """Set up and run training and/or testing for the model based on params."""
    params = get_params()
    params = get_params()

    set_seed(params["seed"])
    print(f"Seed set to: {params['seed']}")
    print(f"Dataset: {params['dataset']}  |  Model: {params['model']}")
    
    device = torch.device(params["device"])
    
    print(f"Using device: {device}")

    model = build_model(params).to(device)
    print(model)

    if params["mode"] in ("train", "both"):
        run_training(model, params, device)

    if params["mode"] in ("test", "both"):
        checkpoint = torch.load(params["save_path"], map_location=device)
        model = build_model(checkpoint["params"]).to(device)
        run_test(model, checkpoint["params"], device)


if __name__ == "__main__":
    main()