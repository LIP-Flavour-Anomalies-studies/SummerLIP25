"""
Models validation loss and stops training when it no longer improves.
Teresa 12/07/2025
"""
class EarlyStopping:
    """
    Stops training if validation loss does not improve for a set number of epochs.
    Saves the best model's weights.
    """
    def __init__(self, patience, delta):
        self.patience = patience  # Number of epochs to wait for improvement in validation loss, before stopping the training
        self.delta = delta  # Minimum change in validation loss that qualifies as an improvement
        self.best_score = None  # Best validation score encountered during training
        self.early_stop = False  # Boolean flag that indicates if training should be stopped early
        self.counter = 0  # Counts the number of epochs since the last improvement in validation loss
        self.best_model_state = None  # Stores the state of the model when the best validation loss was observed

    def __call__(self, val_loss, model):
        """Allows the instance to be called like a function during training"""
        # convert validation loss into a score (lower loss = higher score)
        score = -val_loss

        # on first call, set the initial best score and save model weight
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        
        # if current score not significantly better than best, increment the counter
        elif score < self.best_score - self.delta:
            self.counter += 1
            # if counter exceeds patience stop training
            if self.counter >= self.patience:
                self.early_stop = True

        # if there is a significant improvement, save new best model state and reset counter
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        """Call after training ends to restore model to its best version (based on validation loss)."""
        model.load_state_dict(self.best_model_state)