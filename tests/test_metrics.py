from bioimage.metrics import compute_metrics


def test_metrics_keys() -> None:
    metrics = compute_metrics([0, 1, 1], [0, 1, 0])
    assert set(metrics.keys()) == {"accuracy", "macro_f1", "balanced_accuracy"}
