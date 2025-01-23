from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

#  Functions for using KNN method

def train_and_test_knn(train_data, train_labels, test_data, test_labels):
    """
    Train and evaluate a K-Nearest Neighbors (KNN) classifier using cross-validation
    to find the optimal value of k, and then test the model on the test set.

    Parameters:
        train_data (ndarray): Training data of shape (n_samples, n_features).
        train_labels (ndarray): Training labels of shape (n_samples,).
        test_data (ndarray): Test data of shape (n_samples, n_features).
        test_labels (ndarray): Test labels of shape (n_samples,).

    Returns:
        tuple: A tuple containing:
            - best_k (int): The optimal value of k selected through cross-validation.
            - y_pred (ndarray): Predicted labels for the test set.
            - test_accuracy (float): The accuracy of the KNN classifier on the test set.
    """


    k_values = range(1, 21)
    accuracies = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        scores = cross_val_score(knn, train_data, train_labels.ravel(), cv=5)
        accuracies.append(scores.mean())

    best_k = k_values[accuracies.index(max(accuracies))]
    print(f"Best k: {best_k}, Accuracy: {max(accuracies):.4f}")

    knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    knn.fit(train_data, train_labels.ravel())
    y_pred = knn.predict(test_data)

    print(f"KNN Test Accuracy: {accuracy_score(test_labels, y_pred):.4f}")
    print(classification_report(test_labels, y_pred))