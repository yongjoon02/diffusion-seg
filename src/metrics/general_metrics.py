"""General segmentation metrics using torchmetrics."""

from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
    MulticlassJaccardIndex,
)
from torchmetrics.segmentation.dice import DiceScore as Dice

# Re-export with simpler names
Precision = MulticlassPrecision
Recall = MulticlassRecall
Specificity = MulticlassSpecificity
JaccardIndex = MulticlassJaccardIndex

__all__ = [
    'Dice',
    'Precision', 
    'Recall',
    'Specificity',
    'JaccardIndex',
]
