import torch
from sklearn.metrics import classification_report
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cnn(model, train_loader, val_loader, optimizer, criterion, epochs=10):

    """
    Train a Convolutional Neural Network (CNN) on a dataset.

    Parameters:
        model (torch.nn.Module): The CNN model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for model weight updates (e.g., Adam, SGD).
        criterion (torch.nn.Module): Loss function used for training (e.g., BCELoss).
        epochs (int, optional): Number of training epochs. Default is 30.

    Returns:
        tuple: A tuple containing:
            - train_losses (list of float): Training loss for each epoch.
            - val_accuracies (list of float): Validation accuracy for each epoch.
    """
    model = model.to(device)
    model.train()
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)

        val_loss, val_accuracy, _, _, _ = evaluate_cnn(model, val_loader)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    return train_losses, val_accuracies


def evaluate_cnn(model, loader):
    """
    Evaluate a Convolutional Neural Network (CNN) on a dataset.

    Parameters:
        model (torch.nn.Module): The trained CNN model to be evaluated.
        loader (DataLoader): DataLoader for the dataset to evaluate (e.g., validation or test set).

    Returns:
        tuple: A tuple containing:
            - val_loss (float): Average loss on the dataset.
            - val_accuracy (float): Accuracy on the dataset as a percentage.
            - report (str): Classification report including precision, recall, F1-score, etc.
            - all_preds (list of float): Predicted labels for the dataset.
            - all_labels (list of float): True labels for the dataset.
    """

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images)
            labels = labels.unsqueeze(1)
            loss = nn.BCELoss()(outputs, labels)
            val_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(loader)
    val_accuracy = 100 * correct / total
    report = classification_report(all_labels, all_preds, target_names=["Benign", "Malignant"], digits=4,zero_division=1)

    return val_loss, val_accuracy, report, all_preds, all_labels