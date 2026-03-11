class EarlyStopper:
  def __init__(self, patience=5, min_delta=0.001):
    """
    Early stopping to monitor validation accuracy.
    
    Args:
      patience: Number of epochs with no improvement after which training will be stopped
      min_delta: Minimum change in validation accuracy to qualify as an improvement
    """
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.best_accuracy = None
    self.stop = False
    self.best_model_state_dict = None
    self.improved = False
  
  def __call__(self, val_accuracy, model_state_dict):
    """
    Check if training should stop based on validation accuracy.
    
    Args:
      val_accuracy: Current validation accuracy
      
    Returns:
      bool: True if training should stop, False otherwise
    """
    if self.best_accuracy is None:
      self.best_accuracy = val_accuracy
    elif val_accuracy > self.best_accuracy + self.min_delta:
      self.best_accuracy = val_accuracy
      self.counter = 0
      self.best_model_state_dict = model_state_dict  # Save the best model state
      self.improved = True
    else:
      self.improved = False
      self.counter += 1
      if self.counter >= self.patience:
        self.stop = True
    
    return self.stop
  

def simple_tokenize(text: str):
    return str(text).lower().split()
