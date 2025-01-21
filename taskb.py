import matplotlib.pyplot as plt
import numpy as np
from pandas import *
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, roc_curve,ConfusionMatrixDisplay,auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from medmnist.info import INFO
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import os
import json 
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# functions for CNN
# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define CNN Model
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.residual_block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Add residual connection
        residual = x
        x = self.residual_block(x)
        x += residual
        x = nn.ReLU()(x)
        
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
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

def train_model_with_scheduler(model, train_loader, val_loader, criterion, optimizer, max_epochs=50, patience=5,save_path="best_model.pth"):
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
            - lr_history (list of float): Learning rate for each epoch.

    Example:
        train_losses, val_losses, lr_history = train_model_with_scheduler(
            model, train_loader, val_loader, criterion, optimizer, max_epochs=50, patience=5
        )
    """
    model = model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5)
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    lr_history = []

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
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                # Calculate validation accuracy
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

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

    # Load the best model state
    model.load_state_dict(torch.load(save_path))
     # Save metrics to a JSON file
    metrics = {"train_losses": train_losses, "val_accuracies": val_accuracies,"lr_history": lr_history}
    with open("training_metrics_new.json", "w") as f:
        json.dump(metrics, f)
    print("Training metrics saved to 'training_metrics.json'.")
    return train_losses, val_losses, lr_history

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
    model.load_state_dict(torch.load("model.pth"))
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


def plot_learning_curves(train_losses, val_accuracies):
    """
    Plot training loss and validation accuracy over epochs.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix using sklearn's ConfusionMatrixDisplay.
    """
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.show()


def plot_roc_curve(y_true, y_scores, num_classes):
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
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def plot_learning_rate(lr_history):
    """
    Plot the learning rate vs. epoch number.

    Parameters:
        lr_history (list of float): Learning rate for each epoch.
    """
    epochs = range(1, len(lr_history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lr_history, label="Learning Rate", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs. Epochs")
    plt.legend()
    plt.grid()
    plt.show()


data = np.load('/Users/sammeng/Downloads/Applied_machine_learning/bloodmnist_224.npz')

train_data=data["train_images"]
val_data=data["val_images"]
test_data=data["test_images"]
train_label=data["train_labels"]
val_label=data["val_labels"]
test_label=data["test_labels"]
train_data_flat = train_data.reshape(train_data.shape[0], -1)
val_data_flat = val_data.reshape(val_data.shape[0], -1)       
test_data_flat = test_data.reshape(test_data.shape[0], -1) 

#  Data pre-processing for CNN
# Determine the number of classes
num_classes = len(np.unique(train_label))
print(f"Number of Classes: {num_classes}")
# reshape the train data 
train_tensor = torch.tensor(train_data, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert and permute to (N, C, H, W)
val_tensor = torch.tensor(val_data, dtype=torch.float32).permute(0, 3, 1, 2)
test_tensor = torch.tensor(test_data, dtype=torch.float32).permute(0, 3, 1, 2)
# Reshape train label
train_label_tensor = torch.tensor(train_label, dtype=torch.long).squeeze()  # Remove extra dimensions
val_label_tensor = torch.tensor(val_label, dtype=torch.long).squeeze()
test_label_tensor = torch.tensor(test_label, dtype=torch.long).squeeze()

train_dataset = TensorDataset(train_tensor,train_label_tensor )
val_dataset = TensorDataset(val_tensor,val_label_tensor )
test_dataset = TensorDataset(test_tensor,test_label_tensor )
# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Data pre-processing finished


# KNN method

k_values = range(1, 21)
accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k,weights='distance')
    scores = cross_val_score(knn, train_data_flat, train_label.ravel() ,cv=5)  # 5-fold cross-validation
    accuracies.append(scores.mean())

best_k = k_values[np.argmax(accuracies)]
print(f"Best k: {best_k}, Accuracy: {max(accuracies):.4f}")
# Train the KNN classifier
knn.fit(train_data_flat, train_label.ravel())

# Predict on test data
y_pred_1 = knn.predict(test_data_flat)

# Evaluate the model
accuracy = accuracy_score(test_label, y_pred_1)
print(f"KNN Classifier Accuracy with k={k}: {accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(test_label, y_pred_1))


# Main Script
# Original Model
model1 = AdvancedCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)
y_true, y_pred, y_scores, train_losses, val_accuracies, _ = train_and_evaluate(
    model1, train_loader, val_loader, test_loader, criterion, optimizer, max_epochs=20, patience=5,
    model_path="model.pth", metrics_path="training_metrics.json", is_new_model=False
)

# New Model with Scheduler
model2 = AdvancedCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer_new = optim.Adam(model2.parameters(), lr=0.001)
y_true_new, y_pred_new, y_scores_new, train_losses_new, val_accuracies_new, lr_history = train_and_evaluate(
    model2, train_loader, val_loader, test_loader, criterion, optimizer_new, max_epochs=50, patience=5,
    model_path="model_new.pth", metrics_path="training_metrics_new.json", is_new_model=True
)

# Plot Comparisons
if train_losses and train_losses_new:
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
class_names = [str(i) for i in range(num_classes)] 
# Confusion Matrix Comparison
plot_confusion_matrix(y_true, y_pred, class_names)
plot_confusion_matrix(y_true_new, y_pred_new, class_names)

# ROC Curve Comparison
plt.figure(figsize=(10, 7))
plot_roc_curve(y_true, y_scores, num_classes)
plot_roc_curve(y_true_new, y_scores_new, num_classes)