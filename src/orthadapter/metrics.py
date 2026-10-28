"""Incremental metrics over complete stages, and retained adapter storage."""

import numpy as np


def incremental_metrics(accuracy_matrix, stage_accuracy):
    """Follow the paper's pre-final-stage maximum in average forgetting."""
    stages = len(accuracy_matrix)
    forgetting = [max(accuracy_matrix[t][j] for t in range(j, stages - 1)) - accuracy_matrix[-1][j]
                  for j in range(stages - 1)]
    return {"average_accuracy": float(np.mean(stage_accuracy)),
            "final_accuracy": float(stage_accuracy[-1]),
            "forgetting": float(np.mean(forgetting)) if forgetting else 0.0}


def adapter_storage(model):
    """Return retained factor bytes, excluding optimizer state and activations."""
    return sum(p.numel() * p.element_size() for layer in model.adapted_layers()
               for adapters in [layer.query_adapters, layer.value_adapters]
               for p in adapters.parameters())
