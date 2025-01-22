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

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve


train_loader, val_loader, test_loader = preprocess_data_for_cnn(batch_size=32)
# Load and preprocess data
train_data, train_label, val_data, val_label, test_data, test_label = load_breastmnist_data()
train_data_flat, val_data_flat, test_data_flat = preprocess_logistic_regression(train_data, val_data, test_data)

# Logistic Regression with Scikit-learn
y_pred_sklearn = logRegrPredict(train_data_flat, train_label, test_data_flat)
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

model = BreastMNISTCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_accuracies = train_cnn(model, train_loader, val_loader, optimizer, criterion, epochs=30)

test_loss, test_accuracy, report, all_preds, all_labels = evaluate_cnn(model, test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
print(report)

plot_learning_curves(train_losses, val_accuracies)
plot_confusion_matrix(all_labels, all_preds)
