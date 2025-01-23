import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Files used fro data preprocessing

def load_and_preprocess_data(data_path, batch_size=32):

    """
    Load and preprocess data for training, validation, and testing.

    Parameters:
        data_path (str): Path to the .npz file containing the dataset.
        batch_size (int, optional): Batch size for DataLoader. Default is 32.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - val_loader (DataLoader): DataLoader for the validation dataset.
            - test_loader (DataLoader): DataLoader for the test dataset.
            - num_classes (int): Number of unique classes in the dataset.
    """
    # Load data
    data = np.load(data_path)
    train_data, val_data, test_data = data["train_images"], data["val_images"], data["test_images"]
    train_label, val_label, test_label = data["train_labels"], data["val_labels"], data["test_labels"]

    # Reshape and convert to tensors
    train_tensor = torch.tensor(train_data, dtype=torch.float32).permute(0, 3, 1, 2)
    val_tensor = torch.tensor(val_data, dtype=torch.float32).permute(0, 3, 1, 2)
    test_tensor = torch.tensor(test_data, dtype=torch.float32).permute(0, 3, 1, 2)

    train_label_tensor = torch.tensor(train_label, dtype=torch.long).squeeze()
    val_label_tensor = torch.tensor(val_label, dtype=torch.long).squeeze()
    test_label_tensor = torch.tensor(test_label, dtype=torch.long).squeeze()

    # Create datasets and data loaders
    train_dataset = TensorDataset(train_tensor, train_label_tensor)
    val_dataset = TensorDataset(val_tensor, val_label_tensor)
    test_dataset = TensorDataset(test_tensor, test_label_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(np.unique(train_label))

import numpy as np

def preprocess_for_knn(data_path):
    """
    Preprocess the dataset for K-Nearest Neighbors (KNN) by flattening the data and 
    extracting training, validation, and test sets along with their labels.

    Parameters:
        data_path (str): Path to the `.npz` dataset file.

    Returns:
        tuple: A tuple containing:
            - train_data_flat (ndarray): Flattened training data of shape (n_samples, n_features).
            - val_data_flat (ndarray): Flattened validation data of shape (n_samples, n_features).
            - test_data_flat (ndarray): Flattened test data of shape (n_samples, n_features).
            - train_label (ndarray): Labels for the training data of shape (n_samples,).
            - val_label (ndarray): Labels for the validation data of shape (n_samples,).
            - test_label (ndarray): Labels for the test data of shape (n_samples,).
            - num_classes (int): The number of unique classes in the dataset.
    """
    # Load dataset
    data = np.load(data_path)
    
    train_data = data["train_images"]
    val_data = data["val_images"]
    test_data = data["test_images"]
    train_label = data["train_labels"]
    val_label = data["val_labels"]
    test_label = data["test_labels"]

    # Flatten the image data for KNN
    train_data_flat = train_data.reshape(train_data.shape[0], -1)
    val_data_flat = val_data.reshape(val_data.shape[0], -1)
    test_data_flat = test_data.reshape(test_data.shape[0], -1)

    # Determine the number of unique classes
    num_classes = len(np.unique(train_label))
    print(f"Number of Classes: {num_classes}")

    return train_data_flat, val_data_flat, test_data_flat, train_label, val_label, test_label, num_classes