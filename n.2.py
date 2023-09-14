import numpy as np
from sklearn.model_selection import train_test_split
X = abhinav['X']
Y = abhinav['Y']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)