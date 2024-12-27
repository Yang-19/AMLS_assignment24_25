from medmnist import BreastMNIST
from torch.utils.data import DataLoader

# Specify dataset information
from medmnist import INFO

info = INFO['breastmnist']
DataClass = getattr(medmnist, info['python_class'])

# Load the dataset
train_dataset = DataClass(split='train', download=True)
val_dataset = DataClass(split='val', download=True)
test_dataset = DataClass(split='test', download=True)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)