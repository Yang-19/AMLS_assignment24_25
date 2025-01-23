from A.data_prepare import preprocess_data_for_cnn,load_breastmnist_data,preprocess_logistic_regression
from A.CNN import BreastMNISTCNN
from A.training_cnn import train_cnn, evaluate_cnn
from A.Plotting import plot_learning_curves, plot_confusion_matrix
import torch.nn as nn
import torch.optim as optim
from A.logistic_regression import (
    logRegrPredict,
    logistic_regression_newton,
    logistic_regression_regularized,
    predict_labels,
)

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import torch
from matplotlib import pyplot as plt
from B.cnn_new import AdvancedCNN
from B.data_pre_processing import load_and_preprocess_data, preprocess_for_knn
from B.KNN import train_and_test_knn
from B.evaluation import (
    plot_learning_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_comparison,
)
from B.training_new import train_and_evaluate


# Statement that says task A is in operation
print("\n" + "=" * 50)
print(f"{'Task A is being executed'.center(50)}")
print("=" * 50 + "\n")

datapath='Datasets/breastmnist.npz'

# Load and preprocess data
train_data, train_label, val_data, val_label, test_data, test_label = load_breastmnist_data(datapath)
train_loader, val_loader, test_loader = preprocess_data_for_cnn(train_data,val_data,test_data,train_label,val_label,test_label,batch_size=32)
train_data_flat, val_data_flat, test_data_flat = preprocess_logistic_regression(train_data, val_data, test_data)

print("\n" + "=" * 50)
print(f"{'Task A : Logistic Regression'.center(50)}")
print("=" * 50 + "\n")


# Logistic Regression with Scikit-learn
y_pred_sklearn = logRegrPredict(train_data_flat, train_label.ravel(), test_data_flat)
print(confusion_matrix(test_label, y_pred_sklearn))
print(classification_report(test_label, y_pred_sklearn))



# Logistic Regression with Newton's Method
theta_newton = logistic_regression_newton(train_data_flat / 255, train_label.ravel())
y_pred_newton = predict_labels(test_data_flat / 255, theta_newton)
print(confusion_matrix(test_label, y_pred_newton))
print(classification_report(test_label, y_pred_newton))


# Logistic Regression with Regularization
theta_regularized = logistic_regression_regularized(train_data_flat / 255, train_label.ravel())
y_pred_regularized = predict_labels(test_data_flat / 255, theta_regularized)
print(confusion_matrix(test_label, y_pred_regularized))
print(classification_report(test_label, y_pred_regularized))

print("\n" + "=" * 50)
print(f"{'Task A : CNN model'.center(50)}")
print("=" * 50 + "\n")

model = BreastMNISTCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_accuracies = train_cnn(model, train_loader, val_loader, optimizer, criterion, epochs=10)

test_loss, test_accuracy, report, all_preds, all_labels = evaluate_cnn(model, test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
print(report)

plot_learning_curves(train_losses, val_accuracies)
plot_confusion_matrix(all_labels, all_preds,class_names=["Benign", "Malignant"])

# Print statement that says task B is in operation:
print("\n" + "=" * 50)
print(f"{'Task B is being executed'.center(50)}")
print("=" * 50 + "\n")
DATA_PATH = 'Datasets/bloodmnist_224.npz'
BATCH_SIZE = 32
MAX_EPOCHS_ORIGINAL = 20
MAX_EPOCHS_NEW = 50
PATIENCE = 5

# Data preprocessing
train_loader_1, val_loader_1, test_loader_1, num_classes = load_and_preprocess_data(DATA_PATH, batch_size=BATCH_SIZE)
train_data_flat_1, val_data_flat_1, test_data_flat_1, train_label_1, val_label_1, test_label_1, num_classes = preprocess_for_knn(DATA_PATH)

# KNN Method
print("Running KNN method...")
knn_metrics = train_and_test_knn(train_data_flat_1,train_label_1,test_data_flat_1,test_label_1)
print(f"KNN Metrics:\n{knn_metrics}")

# CNN Original Model
print("Training and evaluating the original CNN model...")
model1 = AdvancedCNN(num_classes=num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
y_true, y_pred, y_scores, train_losses, val_accuracies, _ = train_and_evaluate(
    model1, train_loader_1, val_loader_1, test_loader_1, criterion, optimizer,
    max_epochs=MAX_EPOCHS_ORIGINAL, patience=PATIENCE,
    model_path="B/model.pth", metrics_path="B/training_metrics.json", is_new_model=False
)

# CNN New Model with Scheduler
print("Training and evaluating the new CNN model with a scheduler...")
model2 = AdvancedCNN(num_classes=num_classes)
optimizer_new = torch.optim.Adam(model2.parameters(), lr=0.001)
y_true_new, y_pred_new, y_scores_new, train_losses_new, val_accuracies_new, lr_history = train_and_evaluate(
    model2, train_loader_1, val_loader_1, test_loader_1, criterion, optimizer_new,
    max_epochs=MAX_EPOCHS_NEW, patience=PATIENCE,
    model_path="B/model_new.pth", metrics_path="B/training_metrics_new.json", is_new_model=True
)

# Plot Comparisons
print("Plotting comparisons...")
plot_comparison(train_losses, train_losses_new, val_accuracies, val_accuracies_new)

# Confusion Matrix Comparison
class_names = [str(i) for i in range(num_classes)]
plot_confusion_matrix(y_true, y_pred, class_names, title="Original Model Confusion Matrix")
plot_confusion_matrix(y_true_new, y_pred_new, class_names, title="New Model Confusion Matrix")

# ROC Curve Comparison
plt.figure(figsize=(10, 7))
plot_roc_curve(y_true, y_scores, num_classes, title="Original Model")
plot_roc_curve(y_true_new, y_scores_new, num_classes, title="New Model")
plt.show()