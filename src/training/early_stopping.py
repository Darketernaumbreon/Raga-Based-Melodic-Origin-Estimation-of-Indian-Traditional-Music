import logging

logger = logging.getLogger("RaagAnalysis")

class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    By default, it tracks a metric that we want to maximize (like Validation F1 or Accuracy).
    """
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Parameters
        ----------
        patience : int
            How many epochs to wait after last time validation metric improved.
        min_delta : float
            Minimum change in the monitored quantity to qualify as an improvement.
        mode : str
            'max' if metric should increase (e.g. F1, Acc), 'min' if metric should decrease (e.g. Loss).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if self.mode == 'max':
            self.best_score = -float('inf')
        else:
            self.best_score = float('inf')

    def __call__(self, current_score: float) -> bool:
        if self.mode == 'max':
            is_best = current_score > self.best_score + self.min_delta
        else:
            is_best = current_score < self.best_score - self.min_delta

        if is_best:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"EarlyStopping triggered after {self.counter} epochs without improvement.")
                self.early_stop = True
                
        return self.early_stop
