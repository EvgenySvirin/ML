from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy


# Task 1

def mse(y_true:np.ndarray, y_predicted:np.ndarray):
    return np.sum((y_true - y_predicted) ** 2) / y_true.shape[0]

def r2(y_true:np.ndarray, y_predicted:np.ndarray):
    diff2 = (y_true - y_predicted) ** 2
    m = np.mean(y_true)
    tot = np.sum((y_true - m) ** 2)
    resid = np.sum(diff2)
    return (tot - resid) / tot

# Task 2

def add_one_left(X):
    return np.concatenate(([1], X))

class NormalLR:
    def __init__(self):
        self.weights = None # Save weights here
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        x_ext = np.apply_along_axis(add_one_left, 1, X)
        x_ext_tr = x_ext.transpose()
                
        self.x_sword =  np.dot(np.linalg.inv(np.matmul(x_ext_tr, x_ext)), x_ext_tr)
        self.weights = np.dot(self.x_sword, y)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        x_ext = np.apply_along_axis(add_one_left, 1, X)
        return np.dot(x_ext, self.weights) 
    
# Task 3

class MyScaler:
  def fit(self, x):
    maxes = np.copy(x[0])
    mines = np.copy(x[0])
    for line in x:
      for i in range(len(line)):
        maxes[i] = max(maxes[i], line[i])
        mines[i] = min(mines[i], line[i])
    self.maxes = maxes
    self.mines = mines
  
  def transform(self, x):
    mines = self.mines
    maxes = self.maxes
    x_res = np.copy(np.vstack([self.mines, self.maxes])) #just for next vstack usages to work
    for line in x:
      row = np.array([(line[i] - mines[i]) / (maxes[i] - mines[i]) if maxes[i] != mines[i] else 0 for i in range(len(line))])
      x_res = np.vstack([x_res, row])
    return x_res[2:]

  def fit_y(self, y):
    self.maxY = max(y)
    self.minY = min(y)
 
  def transform_y(self, y):
    return np.array([(label - self.minY) / (self.maxY - self.minY) for label in y])
  
  def inverse_transform_y(self, y):
    return np.array([(label * (self.maxY - self.minY) + self.minY) for label in y])


class GradientLR:
    def __init__(self, alpha:float, iterations=10000, l=0.):
        self.alpha = alpha
        self.iterations = iterations
        self.l = l

    def fit(self, X:np.ndarray, y:np.ndarray):
        self.scaler = MyScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        x_ext = np.apply_along_axis(add_one_left, 1, X)
        self.weights = np.zeros(x_ext.shape[1])

        self.scaler.fit_y(y)
        y = self.scaler.transform_y(y)
        for i in range(self.iterations):
            y_predicted = np.dot(x_ext, self.weights)
            gradient = np.dot(y_predicted - y, x_ext) / x_ext.shape[0]
            gradient += np.dot(self.l, np.sign(self.weights))
            self.weights -= np.dot(self.alpha, gradient)

    def predict(self, X:np.ndarray):
        X = self.scaler.transform(X)
        if (X.shape[1] < self.weights.shape[0]):
            X = np.apply_along_axis(add_one_left, 1, X)
        return self.scaler.inverse_transform_y(np.dot(X, self.weights)) 

# Task 4

def get_feature_importance(linear_regression):
    return np.abs(linear_regression.weights[1:])

def get_most_important_features(linear_regression):
    return list(zip(*sorted(enumerate(get_feature_importance(linear_regression)), key=lambda x :x[1], reverse=True)))[0]