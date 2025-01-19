import medmnist
from medmnist import BreastMNIST
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, roc_curve,roc_auc_score,ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from torchvision.transforms import Compose, ToTensor, Normalize

# sklearn functions implementation
def logRegrPredict(x_train, y_train,xtest ):
    # Build Logistic Regression Model
    logreg = LogisticRegression(solver='newton-cholesky')
    # Train the model using the training sets
    logreg.fit(x_train, y_train)
    y_pred= logreg.predict(xtest)
    #print('Accuracy on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
    return y_pred

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Newton's Method for Logistic Regression
def logistic_regression_newton(x_train, label, max_iter=10000, tol=1e-6):
    # Add intercept term
    intercept = np.ones((x_train.shape[0], 1))
    x_train = np.concatenate((intercept, x_train), axis=1)

    # Initialize parameters
    theta = np.zeros(x_train.shape[1])

    for i in range(max_iter):
        # Predicted probabilities
        z = np.dot(x_train, theta)
        h = sigmoid(z)

        # Gradient
        gradient = np.dot(x_train.T, (label - h))


        # Hessian
        R = np.diag(h * (1 - h))  # Diagonal matrix
        H = np.dot(x_train.T, np.dot(R, x_train))

        # Update parameters using Newton's Method
        try:
            theta_update = np.linalg.solve(H, gradient)  # Solve H * delta = gradient
          
        except np.linalg.LinAlgError:
            print("Hessian is singular, stopping optimization.")
            break

        theta = theta+theta_update

        # Check convergence
        if np.linalg.norm(theta_update, ord=2) < tol:
            print(f"Converged after {i+1} iterations.")
            break

    return theta
# Regularized Logistic Regression using Newton's Method
def logistic_regression_regularized(x_train, label, max_iter=1000, tol=1e-6, lambda_=1.0):
    # Add intercept term
    intercept = np.ones((x_train.shape[0], 1))
    x_train = np.concatenate((intercept, x_train), axis=1)

    # Initialize parameters
    theta = np.zeros(x_train.shape[1])

    for i in range(max_iter):
        # Predicted probabilities
        z = np.dot(x_train, theta)
        h = sigmoid(z)

        # Gradient with L2 regularization
        gradient = np.dot(x_train.T, (label - h)) - (lambda_ * theta) / x_train.shape[0]
        gradient[0] += lambda_ * theta[0] / x_train.shape[0]  # Do not regularize intercept

        # Hessian with L2 regularization
        R = np.diag(h * (1 - h))  # Diagonal matrix
        H = np.dot(x_train.T, np.dot(R, x_train)) + (lambda_ / x_train.shape[0]) * np.eye(x_train.shape[1])
        H[0, 0] -= (lambda_ / x_train.shape[0])  # Do not regularize intercept

        # Update parameters using Newton's Method
        try:
            theta_update = np.linalg.solve(H, gradient)  # Solve H * delta = gradient
        except np.linalg.LinAlgError:
            print("Hessian is singular, stopping optimization.")
            break

        theta += theta_update

        # Check convergence
        if np.linalg.norm(theta_update, ord=2) < tol:
            print(f"Converged after {i+1} iterations.")
            break

    return theta

# Function to predict labels
def predict_labels(x, theta):
    # Add intercept term to test data
    intercept = np.ones((x.shape[0], 1))
    x = np.concatenate((intercept, x), axis=1)

    # Compute probabilities
    z = np.dot(x, theta)
    probabilities = sigmoid(z)

    # Convert probabilities to labels (threshold = 0.5)
    labels = (probabilities >= 0.5).astype(int)

    return labels
# Load the BreastMNIST dataset

#CNN part of the cw
class BreastMNISTCNN(nn.Module):
    def __init__(self):
        super(BreastMNISTCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Conv Layer 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool Layer 1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv Layer 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Pool Layer 2
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the output of the conv layers
            nn.Linear(64 * 7 * 7, 128),  # Fully Connected Layer 1
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(128, 1),  # Fully Connected Layer 2 (output layer)
            nn.Sigmoid()  # Output probability
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Initialize model
model = BreastMNISTCNN()

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, epochs=30):
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)  # Move to GPU if available
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        val_loss, val_accuracy, _, _, _ = evaluate_model(model, val_loader)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {100 * correct/total:.2f}%, "
              f"Validation Loss: {val_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%")

# Validation loop
def evaluate_model(model, loader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            #labels = labels.unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())  # Collect predictions
            all_labels.extend(labels.cpu().numpy())  # Collect true labels
            report = classification_report(all_labels, all_preds, target_names=["Benign", "Malignant"], digits=4)
    return val_loss / len(loader), 100 * correct / total,report, all_preds, all_labels

def plot_confusion_matrix(labels, preds):
    ConfusionMatrixDisplay.from_predictions(labels, preds, display_labels=["Benign", "Malignant"])
    plt.title("Confusion Matrix")
    plt.show()



#train_dataset = BreastMNIST(split='train', download=True)
dataset = BreastMNIST(split='train')
data = np.load('/Users/sammeng/.medmnist/breastmnist.npz')
train_data=data["train_images"]
val_data=data["val_images"]
test_data=data["test_images"]
train_label=data["train_labels"]
val_label=data["val_labels"]
test_label=data["test_labels"]
#To flat it to fit the purpose of logistic regression
train_data_flat = train_data.reshape(train_data.shape[0], -1)
val_data_flat = val_data.reshape(val_data.shape[0], -1)       
test_data_flat = test_data.reshape(test_data.shape[0], -1) 

scaler = MinMaxScaler()
train_data_flat=scaler.fit_transform(train_data_flat)
test_data_flat=scaler.transform(test_data_flat)
#Fet logistic regression using standard library approach
y_pred = logRegrPredict(train_data_flat, train_label,test_data_flat)

print(confusion_matrix(test_label, y_pred))
print('Accuracy on test set: '+str(accuracy_score(test_label,y_pred)))
print(classification_report(test_label,y_pred))
# Compute AUC
auc = roc_auc_score(test_label, y_pred)
print(f"AUC: {auc:.4f}")

#For the implemented logistic regression method with regularization
train_label_R=train_label.ravel()

theta=logistic_regression_regularized(train_data_flat/255,train_label_R)
y_pred_1=predict_labels(test_data_flat/255,theta)

print("Confusion matrix with regularization")
print(confusion_matrix(test_label, y_pred_1))
print('Accuracy of logistic regression with regularization on test set: '+str(accuracy_score(test_label,y_pred_1)))
print(" Classification report of logistic regression with regularization")
print(classification_report(test_label,y_pred_1))
# Compute AUC
auc_1 = roc_auc_score(test_label, y_pred_1)
print(f"AUC_with_regularization: {auc_1:.4f}")

#Logistic regression without regularization
theta1=logistic_regression_newton(train_data_flat/255,train_label_R)
y_pred_2=predict_labels(test_data_flat/255,theta1)
print("Confusion matrix without regularization")
print(confusion_matrix(test_label, y_pred_2))
print('Accuracy of logistic regression without regularization on test set: '+str(accuracy_score(test_label,y_pred_2)))
print(" Classification report of logistic regression without regularization")
print(classification_report(test_label,y_pred_2))
# Compute AUC
auc_2 = roc_auc_score(test_label, y_pred_2)
print(f"AUC_without: {auc_2:.4f}")


fpr_no_reg, tpr_no_reg, _ = roc_curve(test_label, y_pred_2)
fpr_reg, tpr_reg, _ = roc_curve(test_label, y_pred_1)
fpr_sklearn, tpr_sklearn, _ = roc_curve(test_label, y_pred)


# Plot ROC Curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_no_reg, tpr_no_reg, label=f"No Regularization (AUC = {auc_2:.2f})")
plt.plot(fpr_reg, tpr_reg, label=f"With Regularization (AUC = {auc_1:.2f})")
plt.plot(fpr_sklearn, tpr_sklearn, label=f"Scikit-learn (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.show(block=False)

#CNN method
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
# Load the BreastMNIST dataset
train_dataset = BreastMNIST(split='train', download=True)
# Load training, validation, and test sets
train_data = BreastMNIST(split='train', transform=transform, download=True)
val_data = BreastMNIST(split='val', transform=transform, download=True)
test_data = BreastMNIST(split='test', transform=transform, download=True)
dataset = BreastMNIST(split='train')
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Train the model
train_model(model, train_loader, val_loader, epochs=30)
# Evaluate on test set
test_loss, test_accuracy,report,all_preds,all_labels = evaluate_model(model, test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Print classification report
print("Classification Report:\n")
print(report)


# Visualize confusion matrix
plot_confusion_matrix(all_labels, all_preds)
