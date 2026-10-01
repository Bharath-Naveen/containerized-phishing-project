import numpy as np

from phishguard.evaluation.evaluate import bootstrap_ci, point_metrics


def test_point_metrics_confusion_and_rates():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    m = point_metrics(y, p)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert abs(m["precision"] - 2 / 3) < 1e-9 and abs(m["recall"] - 2 / 3) < 1e-9
    assert abs(m["false_positive_rate"] - 0.5) < 1e-9


def test_majority_baseline_has_zero_f1():
    y = np.array([0, 0, 0, 1])
    m = point_metrics(y, None, pred=np.zeros(4, dtype=int))
    assert m["f1"] == 0.0 and m["roc_auc"] == 0.5


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.6 + rng.random(500) * 0.5, 0, 1)
    point = point_metrics(y, p)
    c = bootstrap_ci(y, p, n_boot=200)
    for k in ("f1", "roc_auc"):
        assert c[k][0] <= point[k] <= c[k][1]
