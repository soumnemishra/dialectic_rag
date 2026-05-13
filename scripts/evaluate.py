import numpy as np
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss

def expected_calibration_error(probs, labels, n_bins=10):
    """
    Partition predictions into bins and compute ECE = weighted avg |accuracy - confidence|.
    """
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(probs, bins) - 1

    bin_sums = np.bincount(binids, weights=probs, minlength=n_bins)
    bin_true = np.bincount(binids, weights=labels, minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)

    nonzero = bin_total > 0
    bin_acc = bin_true[nonzero] / bin_total[nonzero]
    bin_conf = bin_sums[nonzero] / bin_total[nonzero]

    ece = np.sum(np.abs(bin_acc - bin_conf) * bin_total[nonzero] / len(probs))
    return round(float(ece), 4)

def brier(probs, labels):
    """Brier score for probabilistic outcomes."""
    return round(float(brier_score_loss(labels, probs)), 4)

def abstention_metrics(pred_abstain, true_abstain):
    """
    Precision and Recall for abstention decisions.
    """
    prec = f1_score(true_abstain, pred_abstain, average='binary') # Simplified to F1
    return {
        "abstention_f1": round(float(prec), 4),
        "abstention_accuracy": round(float(accuracy_score(true_abstain, pred_abstain)), 4)
    }

if __name__ == "__main__":
    # Example evaluation usage
    # probs = np.array([0.9, 0.8, 0.4, 0.2])
    # labels = np.array([1, 1, 0, 0])
    # print(f"ECE: {expected_calibration_error(probs, labels)}")
    pass
