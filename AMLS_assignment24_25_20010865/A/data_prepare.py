import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_breastmnist_data(datapath):
    """
    Plot a confusion matrix for the given true and predicted labels.

    Parameters:
        y_true (list or ndarray): Ground truth labels.
        y_pred (list or ndarray): Predicted class labels by the model.
        class_names (list of str): Names of the classes to display on the matrix axes.

    Returns:
        None: This function generates and displays the confusion matrix plot.
    """

    # Load data from BreastMNIST
    data = np.load(datapath)
    train_data = data["train_images"]
    val_data = data["val_images"]
    test_data = data["test_images"]
    train_label = data["train_labels"]
    val_label = data["val_labels"]
    test_label = data["test_labels"]
    return train_data, train_label, val_data, val_label, test_data, test_label

def preprocess_logistic_regression(train_data, val_data, test_data):
    """
    Preprocess data for logistic regression by flattening and scaling the input data.

    Parameters:
        train_data (ndarray): Training dataset with shape (n_samples, height, width, channels).
        val_data (ndarray): Validation dataset with shape (n_samples, height, width, channels).
        test_data (ndarray): Test dataset with shape (n_samples, height, width, channels).

    Returns:
        tuple: A tuple containing:
            - train_data_flat (ndarray): Flattened and scaled training data with shape (n_samples, n_features).
            - val_data_flat (ndarray): Flattened and scaled validation data with shape (n_samples, n_features).
            - test_data_flat (ndarray): Flattened and scaled test data with shape (n_samples, n_features).
    """
    scaler = MinMaxScaler()
    train_data_flat = train_data.reshape(train_data.shape[0], -1)
    val_data_flat = val_data.reshape(val_data.shape[0], -1)
    test_data_flat = test_data.reshape(test_data.shape[0], -1)
    train_data_flat = scaler.fit_transform(train_data_flat)
    val_data_flat = scaler.transform(val_data_flat)
    test_data_flat = scaler.transform(test_data_flat)
    return train_data_flat, val_data_flat, test_data_flat



# def preprocess_data_for_cnn(train_data,val_data, test_data,batch_size=32,):
#     """
#     Load and preprocess BreastMNIST data for CNN training.

#     Args:
#         batch_size (int): Batch size for DataLoader.

#     Returns:
#         tuple: train_loader, val_loader, test_loader
#     """
#     transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

#     train_data = BreastMNIST(split='train', transform=transform, download=True)
#     val_data = BreastMNIST(split='val', transform=transform, download=True)
#     test_data = BreastMNIST(split='test', transform=transform, download=True)

#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader, test_loader

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, TensorDataset
import torch

def preprocess_data_for_cnn(train_data, val_data, test_data, train_labels, val_labels, test_labels, batch_size=32):
    """
    Preprocess existing BreastMNIST data for CNN training without downloading.

    Args:
        train_data (ndarray): Training image data of shape (n_samples, height, width, channels).
        val_data (ndarray): Validation image data of shape (n_samples, height, width, channels).
        test_data (ndarray): Test image data of shape (n_samples, height, width, channels).
        train_labels (ndarray): Labels for training data of shape (n_samples,).
        val_labels (ndarray): Labels for validation data of shape (n_samples,).
        test_labels (ndarray): Labels for test data of shape (n_samples,).
        batch_size (int): Batch size for DataLoader.

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
     # Transform the data for CNN (normalize and convert to tensors)
    # Add channel dimension (N, 1, H, W) and apply transformation
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)/ 255.0
    val_data_tensor = torch.tensor(val_data, dtype=torch.float32).unsqueeze(1)/ 255.0
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)/ 255.0
    # Convert labels to tensors and squeeze to ensure shape (N,)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).ravel()
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long).ravel()
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long).ravel()

    # Create datasets
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader