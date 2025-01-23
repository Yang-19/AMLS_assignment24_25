import numpy as np
from sklearn.linear_model import LogisticRegression

# Sigmoid function
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Clip values to prevent overflow
    return 1 / (1 + np.exp(-z))

# Logistic Regression Using Scikit-learn
def logRegrPredict(x_train, y_train, x_test):
    """
    Build and train a logistic regression model using scikit-learn.

    Parameters:
        x_train (numpy.ndarray): Training feature set.
        y_train (numpy.ndarray): Training labels.
        x_test (numpy.ndarray): Test feature set.

    Returns:
        numpy.ndarray: Predicted labels for the test set.
    """
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    return y_pred

# Logistic Regression Using Newton's Method
def logistic_regression_newton(x_train, label, max_iter=10000, tol=1e-6):
    """
    Train logistic regression using Newton's Method.

    Parameters:
        x_train (numpy.ndarray): Training feature set.
        label (numpy.ndarray): Training labels.
        max_iter (int): Maximum iterations for convergence.
        tol (float): Convergence tolerance.

    Returns:
        numpy.ndarray: Model parameters (theta).
    """
    intercept = np.ones((x_train.shape[0], 1))
    x_train = np.concatenate((intercept, x_train), axis=1)
    theta = np.zeros(x_train.shape[1])

    for i in range(max_iter):
        z = np.dot(x_train, theta)
        h = sigmoid(z)
        gradient = np.dot(x_train.T, (label - h))
        R = np.diag(h * (1 - h))
        H = np.dot(x_train.T, np.dot(R, x_train))
        try:
            theta_update = np.linalg.solve(H, gradient)
        except np.linalg.LinAlgError:
            print("Hessian is singular, stopping optimization.")
            break
        theta += theta_update
        if np.linalg.norm(theta_update, ord=2) < tol:
            print(f"Converged after {i + 1} iterations.")
            break

    return theta

# Regularized Logistic Regression Using Newton's Method
def logistic_regression_regularized(x_train, label, max_iter=1000, tol=1e-6, lambda_=1.0):
    """
    Train logistic regression with L2 regularization using Newton's Method.

    Parameters:
        x_train (numpy.ndarray): Training feature set.
        label (numpy.ndarray): Training labels.
        max_iter (int): Maximum iterations for convergence.
        tol (float): Convergence tolerance.
        lambda_ (float): Regularization parameter.

    Returns:
        numpy.ndarray: Model parameters (theta).
    """
    intercept = np.ones((x_train.shape[0], 1))
    x_train = np.concatenate((intercept, x_train), axis=1)
    theta = np.zeros(x_train.shape[1])

    for i in range(max_iter):
        z = np.dot(x_train, theta)
        h = sigmoid(z)
        gradient = np.dot(x_train.T, (label - h)) - (lambda_ * theta) / x_train.shape[0]
        gradient[0] += lambda_ * theta[0] / x_train.shape[0]  # Do not regularize intercept
        R = np.diag(h * (1 - h))
        H = np.dot(x_train.T, np.dot(R, x_train)) + (lambda_ / x_train.shape[0]) * np.eye(x_train.shape[1])
        H[0, 0] -= (lambda_ / x_train.shape[0])  # Do not regularize intercept
        try:
            theta_update = np.linalg.solve(H, gradient)
        except np.linalg.LinAlgError:
            print("Hessian is singular, stopping optimization.")
            break
        theta += theta_update
        if np.linalg.norm(theta_update, ord=2) < tol:
            print(f"Converged after {i + 1} iterations.")
            break

    return theta

# Predict Labels for Logistic Regression
def predict_labels(x, theta):
    """
    Predict labels using logistic regression parameters.

    Parameters:
        x (numpy.ndarray): Feature set.
        theta (numpy.ndarray): Logistic regression parameters.

    Returns:
        numpy.ndarray: Predicted labels.
    """
    intercept = np.ones((x.shape[0], 1))
    x = np.concatenate((intercept, x), axis=1)
    z = np.dot(x, theta)
    probabilities = sigmoid(z)
    return (probabilities >= 0.5).astype(int)