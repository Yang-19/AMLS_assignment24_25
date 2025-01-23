# AMLS_assignment24_25
### README

#### Project Description
This project evaluates the performance of various machine learning models for binary and multi-class classification tasks using two datasets, **BreastMNIST** and **BloodMNIST**. Task A focuses on logistic regression (LR) and convolutional neural networks (CNN) for binary classification, while Task B evaluates k-nearest neighbors (KNN) and advanced CNN architectures for multi-class classification. The project aims to provide comparative insights into the efficiency, accuracy, and optimization strategies of different algorithms under consistent conditions.

#### Folder Structure and Role of Files
- **Folder A: Task A (Binary Classification)**
  - **`CNN.py`**: Implements the `BreastMNISTCNN` class, a CNN architecture for binary classification.
  - **`data_prepare.py`**: Prepares the BreastMNIST dataset for logistic regression and CNNs, including normalization and batching.
  - **`logistic_regression.py`**: Implements logistic regression using scikit-learn and custom methods (Newton's method and L2 regularization).
  - **`Plotting.py`**: Provides utility functions to plot learning curves and confusion matrices for Task A models.
  - **`training_cnn.py`**: Defines training and evaluation pipelines for CNN models in Task A.

- **Folder B: Task B (Multi-class Classification)**
  - **`cnn_new.py`**: Implements `AdvancedCNN`, a deeper CNN architecture for multi-class classification with features like residual connections.
  - **`data_pre_processing.py`**: Loads and preprocesses the BloodMNIST dataset for CNN and KNN, including flattening for KNN and channel normalization for CNN.
  - **`evaluation.py`**: Provides utility functions for visualizing learning curves, confusion matrices, and ROC curves for Task B models.
  - **`KNN.py`**: Implements a KNN classifier with hyperparameter tuning (k-selection using cross-validation) and evaluation metrics.
  - **`training_new.py`**: Implements CNN training with dynamic learning rate adjustment, early stopping, and evaluation pipelines.

- **Pretrained Models and Metrics**
  - **`model_new.pth`, `model.pth`**: Saved state dictionaries for trained models in Task B respectively for different CNN training process.
  - **`training_metrics.json`, `training_metrics_new.json`**: Logs training losses and validation accuracies for different models.

#### Required Packages
The following Python libraries are required to run the project:
- **Core Libraries**:
  - `numpy` - Numerical computations.
  - `json` - For saving/loading metrics and configurations.
  - `os` - File management.
  
- **Data Handling and Preprocessing**:
  - `torch` - PyTorch for deep learning model implementation.
  - `torch.nn`, `torch.optim` - Neural network modules and optimization tools.
  - `torch.utils.data` - Data loaders for batching datasets.
  - `scikit-learn` - For logistic regression, KNN, metrics, and preprocessing utilities (e.g., `label_binarize`).

- **Visualization**:
  - `matplotlib` - Plotting learning curves, confusion matrices, and ROC curves.

- **Specialized Packages**:
  - `medmnist` - Provides access to metadata for the BreastMNIST and BloodMNIST datasets.

#### Setup Instructions
1. Install all required libraries using `pip install numpy matplotlib scikit-learn torch medmnist`.
2. Ensure the dataset files (`breastmnist.npz`, `bloodmnist_224.npz`) are placed in the appropriate directories (inside Datasets floder and ensure they have the correct name) specified in the code.
3. Run scripts for Task A or Task B sequentially, starting with data preparation, followed by model training and evaluation.

This project combines traditional and deep learning methods, allowing for a holistic comparison of ML performance in medical imaging classification tasks.
