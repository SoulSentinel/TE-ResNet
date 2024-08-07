import numpy as np
from sklearn.metrics import roc_curve

class Evaluator:
    def __init__(self):
        pass

    def compute_eer(self, y_true, y_score):
        """
        Compute the Equal Error Rate (EER)
        
        Args:
        y_true (array-like): True binary labels
        y_score (array-like): Target scores (can be probability estimates of the positive class)
        
        Returns:
        float: The EER
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        fnr = 1 - tpr
        
        # Find the threshold where FPR and FNR are closest
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        
        # Calculate EER
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        
        return eer

    def evaluate(self, y_true, y_score):
        """
        Evaluate the model performance
        
        Args:
        y_true (array-like): True binary labels
        y_score (array-like): Target scores (can be probability estimates of the positive class)
        
        Returns:
        dict: A dictionary containing the EER
        """
        eer = self.compute_eer(y_true, y_score)
        return {"EER": eer}

