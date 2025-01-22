import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_learning_curves(train_losses, val_accuracies):
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


def plot_confusion_matrix(labels, preds):
    ConfusionMatrixDisplay.from_predictions(labels, preds, display_labels=["Benign", "Malignant"])
    plt.title("Confusion Matrix")
    plt.show()