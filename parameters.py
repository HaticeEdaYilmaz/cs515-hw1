import argparse


def get_params():
    parser = argparse.ArgumentParser(description="Deep Learning on MNIST")

    parser.add_argument("--mode",      choices=["train", "test", "both"], default="both")
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--batch_size",type=int,   default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--activation",type=str,default="relu",choices=["relu", "tanh", "leakyrelu"])
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--gamma", type= float, default= 0.5)
    parser.add_argument("--step_size", type=int,default = 5)
    parser.add_argument("--early_stop", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait for improvement before stopping")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    input_size = 784          # 1 × 28 × 28
    mean, std  = (0.1307,), (0.3081,)

    return {
        # Data
        "dataset":      "mnist",
        "data_dir":     "./data",
        "num_workers":  2,
        "mean":         mean,
        "std":          std,

        # Model
        "model":        "mlp",
        "input_size":   input_size,
        "hidden_sizes": args.hidden_sizes,
        "num_classes":  10,
        "dropout":      args.dropout,
        "activation": args.activation,
        "batchnorm": args.batchnorm,
        "scheduler": args.scheduler,
        "gamma": args.gamma,
        "step_size": args.step_size,
        
        # Training
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
        "weight_decay":  args.weight_decay,
        "early_stop": args.early_stop,
        "patience": args.patience,

        # Misc
        "seed":         42,
        "device":       args.device,
        "save_path":    args.save_path,
        "log_interval": 100,

        # CLI
        "mode":         args.mode,
    }