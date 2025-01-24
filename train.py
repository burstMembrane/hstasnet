import os
import sys
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path

from src.losses import l1_loss
from src.solver import Solver
from src.dataset import MUSDB18Dataset
from hstasnet import HSTasNet
from config.parse import parse_config
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def define_loaders(args):
    """Define DataLoaders for the training, validation, and test sets.

    Args:
        args (dict): Dictionary containing the training parameters.

    Returns:
        loaders (dict): Dictionary containing the DataLoaders.
    """

    
        
    root = args['dataset_path']
    sources = args['model_srcs']
    print(f"Loading dataset from {root}")


    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset not found at {root}")

    trn_dataset = MUSDB18Dataset(root, 'train', sources)
    val_dataset = MUSDB18Dataset(root, 'valid', sources)
    tst_dataset = MUSDB18Dataset(root, 'test', sources)
    print(trn_dataset.path)
    # Define DataLoaders.
    trn_loader = DataLoader(trn_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)
    tst_loader = DataLoader(tst_dataset, batch_size=args['batch_size'], shuffle=False)

    # Store DataLoaders in a dictionary.
    loaders = {
        'trn_loader': trn_loader,
        'val_loader': val_loader,
        'tst_loader': tst_loader,
        }
    
    return loaders


def main(args, train=True):
    """Define model, loaders, optimizer, criterion, scheduler, solver and launch training.

    Args:
        args (dict): Dictionary containing the training parameters.

    Returns:
        solver (Solver): Solver instance containing the training information, e.g. model and loss history.
    """

    # Define loaders.
    loaders = define_loaders(args)

    # Define model.
    model = HSTasNet(**args['model_args'])
    os.makedirs(os.path.dirname(args['model_path']), exist_ok=True)

    criterion = l1_loss
    # Define criterion.

    # Define optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    # Define scheduler.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0) 
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.125, patience=3)

    # Define solver.
    solver = Solver(model, criterion, optimizer, scheduler, loaders, args, device=args['device'])
    os.makedirs(os.path.dirname(args['solver_path']), exist_ok=True)

    # Train model.
    solver = solver.train() if train else solver

    # Save log file.
    os.makedirs(os.path.dirname(args['log_path']), exist_ok=True)
   

    return solver


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train the HSTasNet model.')
    parser.add_argument('--config', type=str, default='./config/train.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    # Empty the GPU cache.
    torch.cuda.empty_cache()

    print("*** START TRAINING ***\n")

    # Read parameters for the training routine and the model.
    if not Path(args.config).exists():
        logger.error(f"No configuration file found at {args.config}")
        
    
    config = parse_config(args.config)

    # Train the model.
    solver = main(config, train=True)

    print("\n*** FINISHED TRAINING ***")
