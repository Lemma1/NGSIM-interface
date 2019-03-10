import os
import numpy as np
import pandas as pd
import pickle

from fancyimpute import SoftImpute

from paras import *

class vk_sensing():
  def __init__(self, method, **kwargs):
    self.clf = None
    self.method = method
    if method = "SoftImpute":
      self.clf = SoftImpute(**kwargs)
    else:
      raise("Not Implemented method")

  def fit(self, X):
    assert(self.clf is not None)
    clf.fit(X)

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
      clf.fit(X_train)
      X_est = clf.transform(X_train)
      err = MAE(X_est, X_val)
      if err < cur_best_err:
        cur_best_err = err
        cuf_best_k = k
    assert(k is not None)
    self.clf = construct_low_rank_imputer(self.method, cur_best_k)
    self.clf.fit(X)

  def transform(self, X):
    assert(self.clf is not None)
    clf.transform(X)


class speed_fitting():
  def __init__(self):
    pass

  def fit(self, X_k, X_v):
    pass

  def _generate_features(self, X_k, X_v):
    pass


def construct_low_rank_imputer(method, k):
  clf = None
  if method == "SoftImpute":
    clf = SoftImpute(max_rank = k)
  else:
    raise("Not implemented")
  return clf

def MAE(X_imputed, X_to_test):
  mask = ~np.isnan(X_to_test)
  return ((X_imputed[missing_mask] - X_to_test[missing_mask]) ** 2).mean()