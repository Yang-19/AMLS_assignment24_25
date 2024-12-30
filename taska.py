import medmnist
from medmnist import BreastMNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, roc_curve,roc_auc_score
from sklearn.preprocessing import MinMaxScaler

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

#For the implemented logistic regression method both with and without regularization
train_label_R=train_label.ravel()

theta=logistic_regression_regularized(train_data_flat/255,train_label_R)
y_pred_1=predict_labels(test_data_flat/255,theta)

print("New confusion matrix")
print(confusion_matrix(test_label, y_pred_1))
print('New accuracy on test set: '+str(accuracy_score(test_label,y_pred_1)))
print("New classification report")
print(classification_report(test_label,y_pred_1))
# Compute AUC
auc = roc_auc_score(test_label, y_pred_1)
print(f"New_AUC: {auc:.4f}")