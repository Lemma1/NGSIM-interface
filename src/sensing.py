import os
import numpy as np
import pandas as pd
import pickle

from fancyimpute import SoftImpute, KNN, SimpleFill
from sklearn.linear_model import LassoCV

from measures import *
from paras import *

class vk_sensing():
  def __init__(self, method, **kwargs):
    self.clf = None
    self.method = method
    if method == "SoftImpute":
      self.clf = SoftImpute(**kwargs)
    elif method == "KNN":
      self.clf = KNN(**kwargs)
    elif method == "Naive":
      self.clf = SimpleFill()
    else:
      raise("Not Implemented method")

  def fit_transform(self, X_train):
    # print (X_train, np.isnan(X_train).all())
    assert(self.clf is not None)
    X_est = None
    if np.isnan(X_train).any():
      if np.isnan(X_train).all():
        X_est = np.zeros_like(X_train)
      else:
        # print (np.isnan(self.clf.fit_transform(X_train)).any())
        X_est = massage_imputed_matrix(self.clf.fit_transform(X_train))
    else:
        X_est = X_train
    assert (not np.isnan(X_est).any())
    return X_est

  def CVfit(self,X, val_ratio = 0.2):
    mask = np.invert(np.isnan(X))
    sample_mask = np.random.rand(*X.shape) < val_ratio
    X_train = X.copy()
    X_train[mask & (~sample_mask)] = np.nan
    X_val = X.copy()
    X_val[mask & (sample_mask)] = np.nan
    cur_best_err = np.inf
    cur_best_k = None
    for k in GLOB_IMPUTE_K_SWEEP:
      clf = construct_low_rank_imputer(self.method, k)
      if np.isnan(X_train).any():
        if np.isnan(X_train).all():
          X_est = np.zeros_like(X_train)
        else:
          X_est = massage_imputed_matrix(clf.fit_transform(X_train))
      else:
        X_est = X_train
      err = MAE(X_est, X_val)
      # print (k, err, RMSN(X_est, X_val))
      if err < cur_best_err:
        cur_best_err = err
        cur_best_k = k
    if cur_best_k is None:
      cur_best_k = 1
    # print (cur_best_k)
    self.clf = construct_low_rank_imputer(self.method, cur_best_k)

  # def transform(self, X):
  #   assert(self.clf is not None)
  #   clf.transform(X)


class speed_fitting():
  def __init__(self):
    self.clf = None

  def CVfit(self, X_k, X_v, left_Xk = None, right_Xk = None):
    X, Y = self._generate_features(X_k, X_v, left_Xk = left_Xk, right_Xk = right_Xk)
    self.clf = LassoCV(cv=3, random_state=0).fit(X, Y)
    # print ("coef", self.clf.coef_)

  def transform(self, X_k, X_v, left_Xk = None, right_Xk = None):
    X_mat = self._generate_features(X_k, X_v = None, left_Xk = left_Xk, right_Xk = right_Xk)
    pred_Y = self.clf.predict(X_mat).reshape(*X_k.shape)
    pred_Y[~np.isnan(X_v)] = X_v[~np.isnan(X_v)]
    return massage_imputed_matrix(pred_Y)


  def _generate_features(self, X_k, X_v = None, left_Xk = None, right_Xk = None, look_back = 2, space_span = 1,
                          left_look_back = 2, left_space_span = 1, right_look_back = 2, right_space_span = 1):
    available_v_mask = None
    if X_v is not None:
      assert(X_k.shape == X_v.shape)
      available_v_mask = ~np.isnan(X_v)
    else:
      available_v_mask = ~np.isnan(np.zeros_like(X_k))
    X_list = list()
    if X_v is not None:
      Y_list = list()
    for i in range(X_k.shape[0]):
      for j in range(X_k.shape[1]):
        if available_v_mask[i,j]:
          tmp_l = list()
          # tmp_l.append(X_k[i,j])
          for t in range(look_back):
            if j - t >= 0:
              tmp_l.append(X_k[i, j-t])
            else:
              tmp_l.append(X_k[i, 0])
          for s in range(space_span):
            if i + s < X_k.shape[0]:
              tmp_l.append(X_k[i+s, j])
            else:
              tmp_l.append(X_k[X_k.shape[0]-1, j])
            if i - s >= 0:
              tmp_l.append(X_k[i-s, j])
            else:
              tmp_l.append(X_k[0, j])
          

          if left_Xk is not None:
            # tmp_l.append(left_Xk[i,j])
            for t in range(left_look_back):
              if j - t >= 0:
                tmp_l.append(left_Xk[i, j-t])
              else:
                tmp_l.append(left_Xk[i, 0])
            for s in range(left_space_span):
              if i + s < left_Xk.shape[0]:
                tmp_l.append(left_Xk[i+s, j])
              else:
                tmp_l.append(left_Xk[left_Xk.shape[0]-1, j])
              if i - s >= 0:
                tmp_l.append(left_Xk[i-s, j])
              else:
                tmp_l.append(left_Xk[0, j])


          if right_Xk is not None:
            # tmp_l.append(right_Xk[i,j])
            for t in range(right_look_back):
              if j - t >= 0:
                tmp_l.append(right_Xk[i, j-t])
              else:
                tmp_l.append(right_Xk[i, 0])
            for s in range(right_space_span):
              if i + s < right_Xk.shape[0]:
                tmp_l.append(right_Xk[i+s, j])
              else:
                tmp_l.append(right_Xk[right_Xk.shape[0]-1, j])
              if i - s >= 0:
                tmp_l.append(right_Xk[i-s, j])
              else:
                tmp_l.append(right_Xk[0, j])

          X_list.append(tmp_l)
          if X_v is not None:
            Y_list.append(X_v[i,j])
    if X_v is not None:
      return np.array(X_list).astype(np.float), np.array(Y_list).astype(np.float)
    else:
      return np.array(X_list).astype(np.float)


def construct_low_rank_imputer(method, k):
  clf = None
  if method == "SoftImpute":
    clf = SoftImpute(max_rank = k, verbose = False)
  elif method == "KNN":
    clf = KNN(k = k, verbose = False)
  else:
    raise("Not implemented")
  return clf

def massage_imputed_matrix(X, eps = 1e-3):
  new_X = X.copy()
  for i in range(X.shape[0]):
    tmp = X[i]
    if np.sum(tmp > eps) > 0:
      available = np.nanmean(tmp[tmp > eps])
    else:
      available = 0
    for j in range(X.shape[1]):
      if X[i,j] > eps:
        available = X[i,j]
      else:
        new_X[i,j] = available
  return new_X