import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix


def plot_learning_curves(train_losses, val_accuracies):
    """
    Plot the learning curves for training loss and validation accuracy over epochs.

    Parameters:
        train_losses (list of float): Training loss for each epoch.
        val_accuracies (list of float): Validation accuracy for each epoch.

    Returns:
        None: Displays the plots of training loss and validation accuracy.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(labels, preds, class_names):
    """
    Plot a confusion matrix.

    Parameters:
        labels (list or array): True class labels.
        preds (list or array): Predicted class labels.
        class_names (list of str): Names of the classes.

    Returns:
        None
    """
    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()