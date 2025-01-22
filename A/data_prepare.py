import torch
from torch.utils.data import DataLoader
from medmnist import BreastMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_breastmnist_data():
    # Load data from BreastMNIST
    data = np.load('/Users/sammeng/.medmnist/breastmnist.npz')
    train_data = data["train_images"]
    val_data = data["val_images"]
    test_data = data["test_images"]
    train_label = data["train_labels"]
    val_label = data["val_labels"]
    test_label = data["test_labels"]
    return train_data, train_label, val_data, val_label, test_data, test_label

def preprocess_logistic_regression(train_data, val_data, test_data):
    scaler = MinMaxScaler()
    train_data_flat = train_data.reshape(train_data.shape[0], -1)
    val_data_flat = val_data.reshape(val_data.shape[0], -1)
    test_data_flat = test_data.reshape(test_data.shape[0], -1)
    train_data_flat = scaler.fit_transform(train_data_flat)
    val_data_flat = scaler.transform(val_data_flat)
    test_data_flat = scaler.transform(test_data_flat)
    return train_data_flat, val_data_flat, test_data_flat



def preprocess_data_for_cnn(batch_size=32):
    """
    Load and preprocess BreastMNIST data for CNN training.

    Args:
        batch_size (int): Batch size for DataLoader.

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    train_data = BreastMNIST(split='train', transform=transform, download=True)
    val_data = BreastMNIST(split='val', transform=transform, download=True)
    test_data = BreastMNIST(split='test', transform=transform, download=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader