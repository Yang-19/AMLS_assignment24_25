import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import os
import numpy as np
from sklearn.metrics import  classification_report,accuracy_score
# CNN trianing file 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    """
    Train a neural network model while tracking training loss and validation accuracy.

    Parameters:
        model (torch.nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function used for training (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer used for updating model weights (e.g., Adam).
        num_epochs (int, optional): Number of epochs to train the model. Default is 20.

    Returns:
        tuple:
            - train_losses (list of float): Training loss for each epoch.
            - val_accuracies (list of float): Validation accuracy for each epoch.
    """
    model = model.to(device)
    best_val_acc = 0.0
    train_losses = []  # To track training loss
    val_accuracies = []  # To track validation accuracy

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels,*_ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
           # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_acc = 0
        with torch.no_grad():
            total, correct = 0, 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

            val_acc = correct / total
            val_accuracies.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "model.pth")  # Save the best model

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    # Save metrics to a JSON file
    metrics = {"train_losses": train_losses, "val_accuracies": val_accuracies}
    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Training metrics saved to 'training_metrics.json'.")
    return train_losses, val_accuracies  # Return tracked metrics

# New function for dynamically selecting epoch number and learning rate 

def train_model_with_scheduler(model, train_loader, val_loader, criterion, optimizer, max_epochs=50, patience=5, save_path="best_model.pth"):
    """
    Train a neural network model using a learning rate scheduler to dynamically adjust learning rates.

    Parameters:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function used for training (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer used for updating model weights (e.g., Adam).
        max_epochs (int, optional): Maximum number of training epochs. Default is 50.
        patience (int, optional): Number of epochs to wait for validation loss improvement before reducing the learning rate. Default is 5.

    Returns:
        tuple: A tuple containing:
            - train_losses (list of float): Training loss for each epoch.
            - val_losses (list of float): Validation loss for each epoch.
            - val_accuracies (list of float): Validation accuracy for each epoch.
            - lr_history (list of float): Learning rate for each epoch.
    """
    model = model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5)
    best_val_loss = float('inf')
    epochs_no_improve = 0  # Initialize counter for early stopping
    train_losses, val_losses, val_accuracies, lr_history = [], [], [], []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, *_ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate validation accuracy
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Log learning rate
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)
        print(f"Learning Rate: {current_lr:.6f}")

        # Adjust learning rate
        scheduler.step(avg_val_loss)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)  # Save the best model to path
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
            epochs_no_improve = 0  # Reset counter if validation loss improves
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

        # Stop training if no improvement for `patience` epochs
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load the best model state
    model.load_state_dict(torch.load(save_path))
    
    # Save metrics to a JSON file
    metrics = {"train_losses": train_losses, "val_losses": val_losses, "val_accuracies": val_accuracies, "lr_history": lr_history}
    with open("training_metrics_new.json", "w") as f:
        json.dump(metrics, f)
    print("Training metrics saved to 'training_metrics_new.json'.")

    return train_losses, val_losses, val_accuracies, lr_history

# A modified functiono to trian and call new models

def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, max_epochs, patience, model_path, metrics_path, is_new_model=False):
    """
    Train and evaluate a model. If a pre-trained model exists, skip training and load metrics.

    Parameters:
        model (torch.nn.Module): The neural network model to train/evaluate.
        train_loader, val_loader, test_loader: DataLoader for train, validation, and test datasets.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        max_epochs (int): Maximum epochs for training.
        patience (int): Patience for learning rate scheduler.
        model_path (str): Path to save/load the model.
        metrics_path (str): Path to save/load the training metrics.
        is_new_model (bool): Flag for differentiating new vs original models in output/logs.

    Returns:
        tuple: (y_true, y_pred, y_scores, train_losses, val_accuracies, lr_history)
    """
    model_name = "new" if is_new_model else "original"
    if not os.path.exists(model_path):
        print(f"No pre-trained {model_name} model found. Training a new model...")
        train_losses, val_accuracies, lr_history = train_model_with_scheduler(
            model, train_loader, val_loader, criterion, optimizer, max_epochs=max_epochs, patience=patience
        )
        torch.save(model.state_dict(), model_path)
        metrics = {
            "train_losses": train_losses,
            "val_accuracies": val_accuracies,
            "lr_history": lr_history,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        print(f"{model_name.capitalize()} model and metrics saved.")
    else:
        print(f"Pre-trained {model_name} model found. Skipping training...")
        model.load_state_dict(torch.load(model_path))
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            train_losses = metrics.get("train_losses", [])
            val_accuracies = metrics.get("val_accuracies", [])
            lr_history = metrics.get("lr_history", [])
            print(f"Loaded {model_name} metrics.")
        else:
            train_losses, val_accuracies, lr_history = [], [], []
            print(f"No {model_name} metrics found. Learning curves will not be plotted.")

    print(f"Evaluating the {model_name} model...")
    y_true, y_pred, y_scores = test_model(model, test_loader)
    return y_true, y_pred, y_scores, train_losses, val_accuracies, lr_history

# Test the model
def test_model(model, test_loader):
    """
    Evaluate a trained model on a test dataset and compute predictions, probabilities, 
    and classification metrics.

    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        tuple: A tuple containing:
            - y_true (list): Ground truth labels from the test dataset.
            - y_pred (list): Predicted class labels by the model.
            - y_scores (ndarray): Raw output scores (logits or probabilities) from the model.
    """

    model.load_state_dict(torch.load("B/model.pth"))
    model = model.to(device)
    model.eval()

    y_true, y_pred,y_scores = [], [],[]
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_scores.append(outputs.cpu().numpy()) 
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    y_scores = np.vstack(y_scores)  # Convert list to NumPy array
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    return y_true, y_pred,y_scores