import torch
import torch.optim as optim
from .schedules import LinearWarmupExponentialDecay


class Trainer:
    def __init__(self, model, learning_rate=1e-3, warmup_steps=None,
                 decay_steps=100000, decay_rate=0.96,
                 ema_decay=0.999, max_grad_norm=10.0):
        self.model = model
        self.ema_decay = ema_decay
        self.max_grad_norm = max_grad_norm

        if warmup_steps is not None:
            self.learning_rate_schedule = LinearWarmupExponentialDecay(
                learning_rate, warmup_steps, decay_steps, decay_rate)
        else:
            self.learning_rate_schedule = lambda step: learning_rate * (decay_rate ** (step / decay_steps))

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
        
        # EMA parameters
        self.ema_params = [p.clone().detach() for p in model.parameters()]
        self.backup_params = None
        self.step_count = 0

    def update_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        
        # Update learning rate
        self.step_count += 1
        new_lr = self.learning_rate_schedule(self.step_count)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Update EMA
        with torch.no_grad():
            for ema_param, param in zip(self.ema_params, self.model.parameters()):
                ema_param.mul_(self.ema_decay).add_(param, alpha=1 - self.ema_decay)

    def load_averaged_variables(self):
        """Load EMA parameters into the model."""
        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.ema_params):
                param.copy_(ema_param)

    def save_variable_backups(self):
        """Save current model parameters as backup."""
        if self.backup_params is None:
            self.backup_params = [p.clone().detach() for p in self.model.parameters()]
        else:
            for backup_param, param in zip(self.backup_params, self.model.parameters()):
                backup_param.copy_(param)

    def restore_variable_backups(self):
        """Restore model parameters from backup."""
        if self.backup_params is not None:
            with torch.no_grad():
                for param, backup_param in zip(self.model.parameters(), self.backup_params):
                    param.copy_(backup_param)

    def train_on_batch(self, inputs, targets, metrics):
        """Train on a single batch."""
        self.model.train()
        preds = self.model(inputs)
        mae = torch.mean(torch.abs(targets - preds), dim=0)
        mean_mae = torch.mean(mae)
        loss = mean_mae
        self.update_weights(loss)

        nsamples = preds.shape[0]
        metrics.update_state(loss.item(), mean_mae.item(), mae.detach().cpu().numpy(), nsamples)

        return loss.item()

    def test_on_batch(self, inputs, targets, metrics):
        """Test on a single batch."""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(inputs)
            mae = torch.mean(torch.abs(targets - preds), dim=0)
            mean_mae = torch.mean(mae)
            loss = mean_mae

            nsamples = preds.shape[0]
            metrics.update_state(loss.item(), mean_mae.item(), mae.cpu().numpy(), nsamples)

        return loss.item()

    def predict_on_batch(self, inputs, targets, metrics):
        """Predict on a single batch."""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(inputs)

            mae = torch.mean(torch.abs(targets - preds), dim=0)
            mean_mae = torch.mean(mae)
            loss = mean_mae
            nsamples = preds.shape[0]
            metrics.update_state(loss.item(), mean_mae.item(), mae.cpu().numpy(), nsamples)

        return preds
