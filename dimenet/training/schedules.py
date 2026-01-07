import torch


class LinearWarmupExponentialDecay:
    """This schedule combines a linear warmup with an exponential decay."""

    def __init__(self, learning_rate, warmup_steps, decay_steps, decay_rate):
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, step):
        # Linear warmup
        warmup_factor = min(step / self.warmup_steps, 1.0)
        # Exponential decay
        decay_factor = self.decay_rate ** (step / self.decay_steps)
        return self.learning_rate * warmup_factor * decay_factor
