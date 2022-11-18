import numpy as np
from sklearn.model_selection import train_test_split
import copy
from typing import NoReturn


# Task 1

def add_one_left(X):
    return np.concatenate(([1], X))

class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """
        self.iterations = iterations
        self.w = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.apply_along_axis(add_one_left, 1, X)
        self.unique_labels = np.unique(y)
        signs = np.array([-1 if p == self.unique_labels[0] else 1 for p in y])
        w = np.zeros(X.shape[1])
        
        biases = X * signs[:, np.newaxis]
        for i in range(self.iterations):
          h = np.sign(np.dot(X, w))
          tfs = np.array(signs != h)
          w +=  np.sum(biases[tfs], axis=0)
        self.w = w
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        if (X.shape[1] < self.w.shape[0]):
            X = np.apply_along_axis(add_one_left, 1, X)
        prod = np.array(np.dot(X, self.w))
        return np.array([self.unique_labels[0] if p <= 0 else self.unique_labels[1] for p in prod])
    
# Task 2
class PerceptronBest:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        """
        self.w = None
        self.iterations = iterations
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.apply_along_axis(add_one_left, 1, X)
        signs =  np.array([-1. if p < 0.5 else 1. for p in y])
        self.w = np.zeros(X.shape[1])
        
        biases = X * signs[:, np.newaxis]
        current_w = np.zeros(X.shape[1])
        best_matches = 0
        for i in range(self.iterations):
          h = np.sign(np.dot(X, current_w))
          tfs = np.array(signs != h)
          matches = len(X) - np.sum(tfs)
          if best_matches < matches:
            best_matches = matches
            self.w = np.copy(current_w)
          if matches == len(X):
            return
          current_w +=  np.sum(biases[tfs], axis=0)
        
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        if (X.shape[1] < self.w.shape[0]):
            X = np.apply_along_axis(add_one_left, 1, X)
        prod = np.array(np.dot(X, self.w))
        return np.array([0 if p <= 0 else 1 for p in prod])

# Task 3

def x_symmetry(image): 
    symmetry = 0
    xs = image.shape[0]
    for i in range(image.shape[1]):
        for j in range(xs):
            cur_val = abs((image[j][i] - image[xs - 1 - j][i]))
            if 0.1 < cur_val:
                symmetry += cur_val**2
    return symmetry

def y_symmetry(image): 
    symmetry = 0
    ys = image.shape[1]
    for i in range(image.shape[0]):
        for j in range(ys):
            cur_val = abs(image[i][j] - image[i][ys - 1 - j])
            if 0.1 < cur_val:
                symmetry += cur_val **2
    return symmetry
    
def transform_images(images):
    transform_result = []
    for image in images:
        inks = image.sum()
        x_sym = x_symmetry(image)
        y_sym = y_symmetry(image)
        incs = np.sum(image)
        transform_result.append([x_sym, y_sym])
    return np.array(transform_result)
