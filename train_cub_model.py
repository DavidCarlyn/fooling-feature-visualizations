import os
from argparse import ArgumentParser

def load_model(model):
    pass

def load_data(batch_size):
    pass

def train_model(model, train_dloader, val_dloader):
    pass

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--outdir", type=str, default="data/models")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    model = load_model(args.model)
    train_dloader, val_dloader = load_data(args.batch_size)
    model = train_model(model, train_dloader, val_dloader)
    save_model(model, os.path.join(args.outdir, f"cub_{args.model}.pt"))