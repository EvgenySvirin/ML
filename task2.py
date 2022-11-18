from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque
from typing import NoReturn
import heapq

class KMeans:
    def __init__(self, n_clusters: int, init: str = "random", 
                 max_iter: int = 300):
        """
        
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются 
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        
        """

        self.init = init
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    
    def getDistsParents(self, points):
      distsParentsTuple = []
      for point in points:
        distsParentsTuple.append(min([(np.linalg.norm(point - self.centers[i]), i) for i in range(len(self.centers)) ]))
      distsParentsTuple = np.array(distsParentsTuple)
      return distsParentsTuple[: , 0], distsParentsTuple[: , 1]
    
    def calculateCenters(self):
      _, parents = self.getDistsParents(self.X)
      sums = np.zeros((len(self.centers), self.X.shape[1]))
      counts = np.zeros(len(self.centers))
      for i in range(len(parents)):
        sums[int(parents[i])] += self.X[i]
        counts[int(parents[i])] += 1

      self.centers = []
      for i in range(len(counts)):
        if (counts[i] != 0):
          self.centers.append(sums[i] / counts[i])
      self.centers = np.array(self.centers)
    

    def plusPlus(self):
      if (len(self.centers) == 0):
        randomInd = int(np.random.uniform(len(self.X)))
        self.centers = np.array([self.X[randomInd]])

      while(len(self.centers) < self.n_clusters):
        dists, _ = self.getDistsParents(self.X)
        M = dists * dists
        M /= max(sum(M), 1)

        newCenter = self.X[np.random.choice(len(self.X), 1, p = M)]
        self.centers = np.vstack([self.centers, newCenter])


    def doIteration(self):
        self.calculateCenters()
        while (len(self.centers) < self.n_clusters):
          self.plusPlus()
          self.calculateCenters()


    def fit(self, X: np.array, y = None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.
        
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать 
            параметры X и y, даже если y не используется).
        
        """
        self.X = X
        #self.y = y
        self.centers = np.array([])

        if (len(X) == self.n_clusters):
          self.centers = self.X
          return
              
        if (self.init == "random"):
          maxVec = [np.max(X[:, i]) for i in range(self.X.shape[1])]
          maxVec = np.array(maxVec)
          minVec = [np.min(X[:, i]) for i in range(self.X.shape[1])]  
          minVec = np.array(minVec)
          self.centers = [minVec + np.random.rand(self.X.shape[1]) * (maxVec - minVec) for i in range(self.n_clusters)] 
          self.centers = np.array(self.centers)

        if (self.init == "sample"):
          self.centers = self.X[np.random.choice(X.shape[0], self.n_clusters)]

        if (self.init == "k-means++"):
          self.plusPlus()

        for it in range(self.max_iter):
          self.doIteration()
        
        
    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера, 
        к которому относится данный элемент.
        
        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.
        
        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров 
            (по одному индексу для каждого элемента из X).
        
        """
        _, parents = self.getDistsParents(X)
        return np.array([int(i) for i in parents])


class DBScan:

    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 leaf_size: int = 40, metric: str = "euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric
        
    def dfs(self, v):
      self.used[v] = 1
      self.res.add(v)
      for to in self.g[v]:
        self.res.add(to)
        if (self.used[to] == 0 and self.core[to] == 1):
          self.dfs(to)

    def startDFS(self, v):
      self.used = np.zeros(len(self.g))
      self.res = set()
      self.dfs(v)


    def fit_predict(self, X: np.array, y = None) -> np.array:
        self.kdtree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        self.g = self.kdtree.query_radius(X, r=self.eps)

        self.core = np.zeros(len(self.g))
        for i in range(len(self.g)):
          if self.min_samples <= len(self.g[i]):
            self.core[i] = 1
        
        labels = np.full(len(self.g), -1)

        k = 0
        for i in range(len(self.g)):
            if self.core[i] == 0:
                continue
            if labels[i] == (-1):
                self.startDFS(i)
                for v in self.res:
                    labels[v] = k
                k += 1
        return labels



class NodeLinked:
   def __init__(self, dataVal=None):
      self.dataVal = dataVal
      self.nextNode = None
 
class Linked:
  def __init__(self, val):
    self.headNode = NodeLinked(val)
    self.lastNode = self.headNode
    self.len = 1
  
  def appendBack(self, anotherLinked):
    self.lastNode.nextNode = anotherLinked.headNode
    self.lastNode = anotherLinked.lastNode
    self.len += anotherLinked.len

  def getList(self):
    res = []
    anotherNode = self.headNode
    while (anotherNode is not None):
      res.append(anotherNode.dataVal)
      anotherNode = anotherNode.nextNode
    return res
  
  

class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """
        
        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры 
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.inf = 100000.0

    
    def recalcDistsAv(self, first, second):
      
      fL = self.clusters[first].len
      sL = self.clusters[second].len

      futureLen = fL + sL

      for i in self.aliveClusters:
        if (i == first or i == second):
          continue
        elif (i < first):
          temp = (self.dists[i][first] * fL + self.dists[i][second] * sL) / (futureLen)
          if (temp != self.dists[i][first]):
            self.dists[i][first] = temp
            heapq.heappush(self.rbt, (temp, i, first))
        elif (i < second):
          temp = (self.dists[first][i] * fL + self.dists[i][second] * sL) / (futureLen)
          if (temp != self.dists[first][i]):           
            self.dists[first][i] = temp
            heapq.heappush(self.rbt, (temp, first, i))
        else:
          temp = (self.dists[first][i] * fL + self.dists[second][i] * sL) / (futureLen)
          if (temp != self.dists[first][i]):
            self.dists[first][i] = temp 
            heapq.heappush(self.rbt, (temp, first, i))

    
    def recalcDistsComp(self, first, second):
      for i in self.aliveClusters:
        if (i == first or i == second):
          continue
        elif (i < first):
          temp = max(self.dists[i][first], self.dists[i][second])
          if (temp != self.dists[i][first]):
            self.dists[i][first] = temp
            heapq.heappush(self.rbt, (temp, i, first))
        elif (i < second):
          temp = max(self.dists[first][i], self.dists[i][second])
          if (temp != self.dists[first][i]):           
            self.dists[first][i] = temp
            heapq.heappush(self.rbt, (temp, first, i))
        else:
          temp = max(self.dists[first][i], self.dists[second][i])
          if (temp != self.dists[first][i]):
            self.dists[first][i] = temp 
            heapq.heappush(self.rbt, (temp, first, i))



    def recalcDistsSin(self, first, second):
      for i in self.aliveClusters:
        if (i == first or i == second):
          continue
        elif (i < first):
          temp = min(self.dists[i][first], self.dists[i][second])
          if (temp != self.dists[i][first]):
            self.dists[i][first] = temp
            heapq.heappush(self.rbt, (temp, i, first))
        elif (i < second):
          temp = min(self.dists[first][i], self.dists[i][second])
          if (temp != self.dists[first][i]):           
            self.dists[first][i] = temp
            heapq.heappush(self.rbt, (temp, first, i))
        else:
          temp = min(self.dists[first][i], self.dists[second][i])
          if (temp != self.dists[first][i]):
            self.dists[first][i] = temp 
            heapq.heappush(self.rbt, (temp, first, i))



    def unite(self):
      curMin, minInd1, minInd2 = heapq.heappop(self.rbt)
      while (self.aliveClustersInds[minInd1] == 0 or 
             self.aliveClustersInds[minInd2] == 0 or 
             self.dists[minInd1][minInd2] != curMin):
        curMin, minInd1, minInd2 = heapq.heappop(self.rbt)

      if (self.linkage == "average"):
        self.recalcDistsAv(minInd1, minInd2)
      elif (self.linkage == "single"):
        self.recalcDistsSin(minInd1, minInd2)
      elif (self.linkage == "complete"):
        self.recalcDistsComp(minInd1, minInd2)
      else:
        self.recalcDistsAv(minInd1, minInd2)

      self.clusters[minInd1].appendBack(self.clusters[minInd2])
      self.aliveClusters.remove(minInd2)
      self.aliveClustersInds[minInd2] = 0
      self.alive -= 1


    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        self.clusters = [Linked(i) for i in range(len(X))]


        self.X = X
        self.labels = np.zeros(len(X))
        self.aliveClusters = list(range(len(X)))
        self.aliveClustersInds = np.full(len(X), 1)
        self.alive = len(X)

        self.dists = np.zeros((len(X), len(X)), dtype = float)
        self.rbt = []

        for i in range(len(X)):
          for k in range(i + 1, len(X)):
            self.dists[i][k] = np.linalg.norm(self.X[i] - self.X[k])
            self.rbt.append((self.dists[i][k], i, k))
        
        heapq.heapify(self.rbt)
        
        while (self.alive != self.n_clusters):  
          self.unite()

        col = 0
        for i in self.aliveClusters:
          res = self.clusters[i].getList()
          self.labels[res] = col
          col += 1
        return np.array([int(l) for l in self.labels])

