import mlflow
import mlflow.pytorch
import torch
import pickle
from tqdm import tqdm
import os
import torchaudio
import numpy as np
from asteroid.metrics import get_metrics
import matplotlib.pyplot as plt
import sys

from mlflow.models.signature import infer_signature
# Add necessary directories to the path.
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_directory, 'data'))
sys.path.append(os.path.join(parent_directory, 'out'))
sys.path.append(os.path.join(parent_directory, 'hstasnet'))
sys.path.append(os.path.join(parent_directory, 'logs'))
sys.path.append(os.path.join(parent_directory, 'src'))
from pathlib import Path
from datetime import datetime
from states import load_model_from_package
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
        
        mlflow.enable_system_metrics_logging()  # Enable system metrics logging
        mlflow.set_experiment(f"/hstasnet_{datetime.now().strftime('%Y%m%d%H%M%S')}")  # Set MLflow experiment
        mlflow.start_run()  # Start MLflow run
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

        # Log hyperparameters to MLflow
        mlflow.log_params({
            **args,
            "num_epochs": self.num_epochs,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "scheduler_step_size": args.get('scheduler_step_size', None),
            "device": self.device,
            "model_name": model.__class__.__name__,
        })

    def train(self):
        for epoch in range(self.running_epoch, self.num_epochs):
            print('---------------------------------------')
            
            # Train.
            self.model.train()
            trn_loss = self._run_one_trn_epoch()

            print(f"Train Summary | Epoch {epoch+1:02d} | Loss = {trn_loss:.3f}")
            mlflow.log_metric("train_loss", trn_loss, step=epoch)

            # Validate.
            self.model.eval()
            with torch.no_grad():
                val_loss = self._run_one_val_epoch()
            # log the GPU memory usage
            print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            mlflow.log_metric("gpu_memory_usage", torch.cuda.memory_allocated() / 1e9, step=epoch)
            print(f"Validation Summary | Epoch {epoch+1:02d} | Loss = {val_loss:.3f}")
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            # Update scheduler.
            self.scheduler.step()
            last_lr = self.scheduler.get_last_lr()[0]
            print(f"\tLearning rate = {last_lr:.6f}")
            mlflow.log_metric("learning_rate", last_lr, step=epoch)

            # Save model if validation loss improves.
            self.val_loss_history[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if isinstance(self.model, torch.nn.DataParallel):
                    self.model.module.save_to_path(self.args['model_path'])
                else:
                    self.model.save_to_path(self.args['model_path'])
                print(f"Best model saved at '{self.args['model_path']}'.")

                # create an input example for the model using a numpy array
                input_example = np.random.randn(1, 2, 16000).astype(np.float32)
                mlflow.pytorch.log_model(
                    self.model,
                    "best_model",
                    input_example=input_example,
                    registered_model_name="hstasnet"
                )
                self.save_to_path(self.args['solver_path'])

            self.running_epoch += 1
        # run tests every five epochs
        # if self.running_epoch % 5 == 0:
     
            self.test()
        self.save_to_path(self.args['solver_path'])
        print(f"Solver saved at '{self.args['solver_path']}'.")
        print('---------------------------------------')

        mlflow.end_run()  # End MLflow run
        return self

    def _run_one_trn_epoch(self):
        running_loss = 0.0
        for i, batch_i in enumerate(tqdm(self.loaders['trn_loader'], "Training epoch")):
            batch_mixture, batch_sources = batch_i
            batch_mixture = batch_mixture.to(self.device)
            batch_sources = batch_sources.to(self.device)

            # Forward pass.
            batch_length = batch_sources.size(-1)
            batch_outputs = self.model(batch_mixture, length=batch_length)

            # Compute loss.
            loss = self.criterion(batch_outputs, batch_sources, reduction='mean')

            # Backward pass and optimisation.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss

    def _run_one_val_epoch(self):
        running_loss = 0.0
        for i, batch_i in enumerate(tqdm(self.loaders['val_loader'], "Validating epoch")):
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
            test_results = self._run_one_tst_epoch()
            print(f"Test Summary  | SDR = {test_results['avg_sdr']:.3f} | SIR = {test_results['avg_sir']:.3f} | SAR = {test_results['avg_sar']:.3f}")
    
    def plot_spectrogram(self, input_audio, title="Spectrogram", output_path=None):
        """Plot a spectrogram and save it to a file."""
        N_FFT = 4096
        N_HOP = 4
        stft = torchaudio.transforms.Spectrogram(
            n_fft=N_FFT,
            hop_length=N_HOP,
            power=None,
        )
        stft = stft(input_audio)
       
        magnitude = stft.abs()
        spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
        _, axis = plt.subplots(1, 1)
        axis.imshow(spectrogram, cmap="viridis", vmin=-60, vmax=0, origin="lower", aspect="auto")
        axis.set_title(title)
        plt.tight_layout()
        axis.savefig(output_path)
    


    def _run_one_tst_epoch(self):
        results = {}
        total_sdr, total_sir, total_sar = 0.0, 0.0, 0.0
        num_batches = 0

        snippet_duration = 30  # 30 seconds
        sample_rate = self.args['sample_rate']
        snippet_length = snippet_duration * sample_rate

        model_sources = self.args['model_srcs']
        test_tracks = self.args['test']['test_tracks']
        print(f"Testing on {len(test_tracks)} tracks.")
        for track_name in test_tracks:
            track_path = Path(self.args['test']['test_dir']) / track_name
            mixture_path = track_path / "mixture.wav"

            # Load mixture
            batch_mixture, _ = torchaudio.load(mixture_path)
            batch_mixture = batch_mixture.to(self.device)
            batch_mixture = batch_mixture.unsqueeze(0)
        
            # Inference
            batch_outputs = self.model(batch_mixture)
            pred_sources = batch_outputs.squeeze(0).detach().cpu()

            os.makedirs(f"output/{track_name}", exist_ok=True)

            out_path = f"output/{track_name}/"

            for stem_idx, stem_name in enumerate(model_sources):
                pred_snippet = pred_sources[stem_idx][:snippet_length]
                print(pred_snippet.size())
                # Save prediction snippet
                # make the directory if it doesn't exist
              
                pred_path = f"{out_path}/{stem_name}_ep_{self.running_epoch}_pred.wav"

                torchaudio.save(pred_path, pred_snippet, sample_rate)

                # create a stft of the prediction
                
                self.plot_spectrogram(pred_snippet, title=f"{track_name} {stem_name} Prediction", output_path=f"{out_path}/{stem_name}_ep_{self.running_epoch}_stft.png")
                # save to a pyplot
                plt.figure()
                plt.imshow(np.abs(stft), aspect='auto', origin='lower')
                plt.colorbar()
                plt.savefig(f"{out_path}/{stem_name}_ep_{self.running_epoch}_stft.png")
                plt.close()

                mlflow.log_artifact(f"{out_path}/{stem_name}_ep_{self.running_epoch}_stft.png", artifact_path="stft_plots")
                mlflow.log_artifact(pred_path, artifact_path="audio_snippets")

                # Load ground truth
                true_path = track_path / f"{stem_name}.wav"
                if true_path.exists():
                    true_snippet, _ = torchaudio.load(true_path)
                    true_snippet = true_snippet.squeeze(0)[:snippet_length]

                    # Save ground truth snippet
                    gt_path = f"{out_path}/{stem_name}_gt.wav"

                    # create a stft of the ground truth
                    stft = torch.stft(true_snippet, n_fft=1024, hop_length=256, win_length=1024, window=torch.hann_window(1024))
                    stft = stft.squeeze(0).cpu().numpy()
                    print(f"STFT shape: {stft.shape}")  # Debugging info
                    # save to a pyplot
                    plt.figure()
                    plt.imshow(np.abs(stft), aspect='auto', origin='lower')
                    plt.colorbar()
                    plt.savefig(f"{out_path}/{stem_name}_gt_stft.png")
                    plt.close()

                    torchaudio.save(gt_path, true_snippet, sample_rate)
                    mlflow.log_artifact(gt_path, artifact_path="audio_snippets")

            # Save mixture snippet
            mixture_snippet = batch_mixture.squeeze(0)[:snippet_length]
            # put the mixture snipped on cpu
            mixture_snippet = mixture_snippet.cpu()
            mixture_snippet_path = f"{out_path}/mixture.wav"
            # create a stft of the mixture
            stft = torch.stft(mixture_snippet, n_fft=1024, hop_length=256, win_length=1024, window=torch.hann_window(1024))
            stft = stft.squeeze(0).cpu().numpy()
            print(f"STFT shape: {stft.shape}")  # Debugging info
            # save to a pyplot
            plt.figure()
            plt.imshow(np.abs(stft), aspect='auto', origin='lower')
            plt.colorbar()
            plt.savefig(f"{out_path}/mixture_stft.png")
            plt.close()

            torchaudio.save(mixture_snippet_path, mixture_snippet, sample_rate)
            mlflow.log_artifact(mixture_snippet_path, artifact_path="audio_snippets")

           # Calculate metrics
            true_sources = []
            for stem_name in model_sources:
                true_path = track_path / f"{stem_name}.wav"
                print(f"Processing source: {stem_name}, Path exists: {true_path.exists()}")  # Debugging info
                if true_path.exists():
                    true_audio, _ = torchaudio.load(true_path)
                    print(f"Loaded audio shape for {stem_name}: {true_audio.shape}")  # Debugging info

                    true_sources.append(true_audio.squeeze(0).cpu().numpy()[:snippet_length])

            if true_sources:
                true_sources = np.array(true_sources)
                logging.debug(f"True sources array shape: {true_sources.shape}")  # Debugging info
                print(f"True sources data: {true_sources}")  # Debugging info

                # collapse mixture to mono
                batch_mixture = batch_mixture.mean(dim=1)

                # collapse sources to mono
                pred_sources = pred_sources.mean(axis=1)
                true_sources = true_sources.mean(axis=1)
                print(f"Collapsed true sources shape: {true_sources.shape}")  # Debugging info

                # pad the mixture to the same length as the sources
                batch_mixture = torch.nn.functional.pad(batch_mixture, (0, true_sources.shape[-1] - batch_mixture.shape[-1]))
                batch_mixture = batch_mixture.cpu().numpy()

                pred_sources = pred_sources.cpu().numpy()

                # ensure pred_sources and true_sources have the same shape
                min_length = min(pred_sources.shape[-1], true_sources.shape[-1])
                pred_sources = np.array([ps[..., :min_length] for ps in pred_sources])
                true_sources = np.array([ts[..., :min_length] for ts in true_sources])
                batch_mixture = batch_mixture[..., :min_length]

                print(f"Shapes after alignment: Pred sources: {pred_sources.shape}, True sources: {true_sources.shape}, Batch mixture: {batch_mixture.shape}")  # Debugging info

                # if any of the sources are all zeros, skip the track as it can lead to ambiguous results
                all_zero_sources = [np.all(ts == 0) for ts in true_sources]
                print(f"All zero sources check: {all_zero_sources}")  # Debugging info
                if np.any(all_zero_sources):
                    print(f"Skipping track {track_name} because all sources are zero.")
                    continue

                metrics = get_metrics(batch_mixture, true_sources, pred_sources, sample_rate=sample_rate, metrics_list=['sdr', 'sir', 'sar'])
                print(f"Metrics for track {track_name}: {metrics}")  # Debugging info

                total_sdr += metrics['sdr']
                total_sir += metrics['sir']
                total_sar += metrics['sar']

                mlflow.log_metrics(metrics, step=self.running_epoch)
                num_batches += 1

            results['avg_sdr'] = total_sdr / num_batches if num_batches > 0 else 0.0
            results['avg_sir'] = total_sir / num_batches if num_batches > 0 else 0.0
            results['avg_sar'] = total_sar / num_batches if num_batches > 0 else 0.0
            mlflow.log_metrics(results, step=self.running_epoch)


            return results



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