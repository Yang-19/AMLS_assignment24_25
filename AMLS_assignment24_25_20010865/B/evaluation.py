import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,roc_curve,auc
from sklearn.preprocessing import label_binarize

#  functions for plotting all the graphs

def plot_learning_curves(train_losses, val_losses, title="Learning Curves", save_path=None):
    """
    Plot training and validation loss curves.

    Parameters:
        train_losses (list): List of training loss values over epochs.
        val_losses (list): List of validation loss values over epochs.
        title (str, optional): Title of the plot. Defaults to "Learning Curves".
        save_path (str, optional): Path to save the plot as an image file. If None, the plot is not saved.
    
    Returns:
        matplotlib.figure.Figure: The figure object for the plot. Useful for saving or further manipulation.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        print(f"Learning curves saved to {save_path}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    """
    Plot a confusion matrix with labels.

    Parameters:
        y_true (list or ndarray): True labels.
        y_pred (list or ndarray): Predicted labels.
        class_names (list): List of class names corresponding to labels.
        title (str, optional): Title of the plot. Defaults to "Confusion Matrix".
        save_path (str, optional): Path to save the plot as an image file. If None, the plot is not saved.
    
    Returns:
        matplotlib.figure.Figure: The plot for confusion matrix. 
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_comparison(train_losses, val_accuracies, train_losses_new, val_accuracies_new):
    """
    Compare training loss and validation accuracy of two models.

    Parameters:
        train_losses (list): Training losses of the original model.
        val_accuracies (list): Validation accuracies of the original model.
        train_losses_new (list): Training losses of the new model.
        val_accuracies_new (list): Validation accuracies of the new model.
    
    Returns:
        matplotlib.figure.Figure: The plot of comparsion
    """
    if not (train_losses and val_accuracies and train_losses_new and val_accuracies_new):
        print("One or more metrics are missing. Cannot plot comparison.")
        return

    plt.figure(figsize=(12, 5))

    # Training Loss Comparison
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Original Model", marker='o')
    plt.plot(train_losses_new, label="New Model", marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()

    # Validation Accuracy Comparison
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Original Model", marker='o')
    plt.plot(val_accuracies_new, label="New Model", marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_scores, num_classes,title):
    """
    Plot ROC curve for multiclass classification.
    """
    # Binarize the true labels
    y_true_binarized = label_binarize(y_true, classes=list(range(num_classes)))

    plt.figure(figsize=(10, 7))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=0.5)  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()