import torch
import pickle
from tqdm import tqdm
import os
import sys
# Add necessary directories to the path.
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_directory, 'data'))
sys.path.append(os.path.join(parent_directory, 'out'))
sys.path.append(os.path.join(parent_directory, 'hstasnet'))
sys.path.append(os.path.join(parent_directory, 'logs'))
sys.path.append(os.path.join(parent_directory, 'src'))
from pathlib import Path

from states import load_model_from_package



class Solver:
    """A class to train and evaluate a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        criterion (nn.Module): The loss function to be optimized.
        optimizer (optim.Optimizer): The optimizer to be used for training.
        scheduler (optim.lr_scheduler._LRScheduler): The learning rate scheduler to be used for training.
        loaders (dict): A dictionary containing the DataLoaders for the training, validation, and test sets.
        args (dict): A dictionary containing additional arguments.
        device (str, optional): The device to use for training. Defaults to 'cpu'.
    """

    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 loaders,
                 args,
                 device='cpu',
                 ):


           # Wrap model for multi-GPU if GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model


        self.device = device
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = args['num_epochs']
        self.loaders = loaders
        self.device = device
        self.model.to(device)
        self.trn_loss_history = torch.zeros(self.num_epochs, device=device)
        self.val_loss_history = torch.zeros(self.num_epochs, device=device)
        self._reset()

    def train(self):

        for epoch in range(self.running_epoch, self.num_epochs):

            print('---------------------------------------')
            
            # Train.
            self.model.train()
            trn_loss = self._run_one_trn_epoch()

            print(f"Train Summary | Epoch {epoch+1:02d} | Loss = {trn_loss:.3f}")

            # Validate.
            self.model.eval()
            with torch.no_grad():             
                val_loss = self._run_one_val_epoch()

            print(f"Validation Summary | Epoch {epoch+1:02d} | Loss = {val_loss:.3f}")

            # Update scheduler.
            self.scheduler.step()
            last_lr = self.scheduler.get_last_lr()[0]
            print(f"\tLearning rate = {last_lr:.6f}")

            # Save model.
      
            self.val_loss_history[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if isinstance(self.model, torch.nn.DataParallel):
                    self.model.module.save_to_path(self.args['model_path'])
                else:
                    self.model.save_to_path(self.args['model_path'])
                print(f"Best model saved at '{self.args['model_path']}'.")
    
            self.running_epoch += 1
        
        self.save_to_path(self.args['solver_path'])
        print(f"Solver saved at '{self.args['solver_path']}'.")
        print('---------------------------------------')

        return self

    def _run_one_trn_epoch(self):

        running_loss = 0.0
        for i, batch_i in enumerate(tqdm(self.loaders['trn_loader'], "Training epoch")):

            # Get the inputs and targets.
            batch_mixture, batch_sources = batch_i
            batch_mixture = batch_mixture.to(self.device)
            batch_sources = batch_sources.to(self.device)

            # Forward pass.
            batch_length = batch_sources.size(-1)
            batch_outputs = self.model(batch_mixture, length=batch_length)

            # Compute loss.
            loss = self.criterion(batch_outputs, batch_sources, reduction='mean')

            # Backward pass and optimization.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss
    
    def _run_one_val_epoch(self):        

        running_loss = 0.0
        for i, batch_i in enumerate(tqdm(self.loaders['val_loader'], "Validating epoch")):

            # Get the inputs and targets.
            batch_mixture, batch_sources = batch_i
            batch_mixture = batch_mixture.to(self.device)
            batch_sources = batch_sources.to(self.device)

            # Forward pass.
            batch_length = batch_sources.size(-1)
            batch_outputs = self.model(batch_mixture, length=batch_length)

            # Compute loss.
            loss = self.criterion(batch_outputs, batch_sources, reduction='mean')
            running_loss += loss.item()

        return running_loss    
    
    def test(self):

        self.model.eval()
        with torch.no_grad():
            tst_loss = self._run_one_tst_epoch()

            print(f"Test Summary  | Loss = {tst_loss:.3f}")
            
    def _run_one_tst_epoch(self):

        running_loss = 0.0
        for i, batch_i in enumerate(tqdm(self.tst_loader, "Testing epoch")):

            # Get the inputs and targets.
            batch_mixture, batch_sources = batch_i
            batch_mixture = batch_mixture.to(self.device)
            batch_sources = batch_sources.to(self.device)

            # Forward pass.
            batch_length = batch_sources.size(-1)
            batch_outputs = self.model(batch_mixture, length=batch_length)

            # Compute loss.
            loss = self.criterion(batch_outputs, batch_sources, reduction='mean')
            running_loss += loss.item()

        return running_loss        

    def _reset(self):

        if self.args['continue_from']:
            print('---------------------------------------')
            print(f"Loading checkpoint solver: '{self.args['continue_from']}'.")     
            checkpoint_path = Path(self.args['continue_from'])
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint '{self.args['continue_from']}' not found.")
            
            model_package = torch.load(str(checkpoint_path))
            self.model.module.load_state_dict(model_package['state_dict'])
            print(f"Model loaded from '{self.args['continue_from']}'.")
      
            # self.optimizer.load_state_dict(package['optimizer_dict'])
            # self.scheduler.load_state_dict(package['scheduler_dict'])
            # self.running_epoch = package['running_epoch']

            
            # self.trn_loss_history[:self.running_epoch] = torch.Tensor(package['trn_loss_history'][:self.running_epoch]).to(self.device)
            # self.val_loss_history[:self.running_epoch] = torch.Tensor(package['val_loss_history'][:self.running_epoch]).to(self.device)
            # TODO: Remove this when the model is trained for real
            self.running_epoch = 89
            self.val_loss_history[89] = 1.042
            self.trn_loss_history[89] = 2.982
        else:
            self.running_epoch = 0

        self.prev_val_loss = float('inf')
        self.best_val_loss = float('inf')

    def serialize(self):
        """Serialize the solver into a dictionary.
        
        Args:    
            solver (Solver): The solver to serialize.

        Returns:
            dict: Dictionary containing the solver's class, arguments, keyword arguments, and state.
        """

        package = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_dict': self.optimizer.state_dict(),
            'scheduler_dict': self.scheduler.state_dict(),
            'running_epoch': self.running_epoch,
            'trn_loss_history': self.trn_loss_history.tolist(),
            'val_loss_history': self.val_loss_history.tolist(),
            }
        
        return package
    
    def save_to_path(self, solver_path):
        """Save the solver to a given file path."""
        # If the model is wrapped in DataParallel, access the underlying model
        model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        solver_package = {
            'model_state_dict': model_to_save.state_dict(),  # Save the underlying model's state dict
            'optimizer_dict': self.optimizer.state_dict(),
            'scheduler_dict': self.scheduler.state_dict(),
            'running_epoch': self.running_epoch,
            'trn_loss_history': self.trn_loss_history.tolist(),
            'val_loss_history': self.val_loss_history.tolist(),
        }
        
        with open(solver_path, 'wb') as solver_file:
            pickle.dump(solver_package, solver_file)

        return solver_path
