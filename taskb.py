import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import pandas as pd
from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import medmnist
from medmnist import BloodMNIST
data = np.load('/Users/sammeng/Downloads/Applied_machine_learning/bloodmnist_224.npz')
# Print all keys in the .npz file
print("Keys in the .npz file:", data.keys())