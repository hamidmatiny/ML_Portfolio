# Fix for torchmetrics Accuracy error
from torchmetrics import Accuracy

# Setup metric with required 'task' parameter
torchmetric_accuracy = Accuracy(task='binary').to(device)  # For binary classification
# For multiclass: Accuracy(task='multiclass', num_classes=num_classes)

# Calculate accuracy
torchmetric_accuracy(y_preds, y_blob_test)