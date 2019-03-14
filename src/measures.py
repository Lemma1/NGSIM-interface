import os
import numpy as np
from sklearn.metrics import r2_score

def MAE(X_imputed, X_to_test, time_margin = 10, space_margin = 5):
  X_imputed2 = X_imputed[space_margin:-space_margin, time_margin:-time_margin]
  X_to_test2 = X_to_test[space_margin:-space_margin, time_margin:-time_margin]
  mask = ~np.isnan(X_to_test2)
  return ((X_imputed2[mask] - X_to_test2[mask]) ** 2).mean()

def RMSPE(X_imputed, X_to_test, time_margin = 10, space_margin = 5):
  X_imputed2 = X_imputed[space_margin:-space_margin, time_margin:-time_margin]
  X_to_test2 = X_to_test[space_margin:-space_margin, time_margin:-time_margin]
  mask = ~np.isnan(X_to_test2)
  return np.sqrt((((X_imputed2[mask] - X_to_test2[mask])/X_to_test2[mask]) ** 2).mean())

def RMSN(X_imputed, X_to_test, time_margin = 10, space_margin = 5):
  X_imputed2 = X_imputed[space_margin:-space_margin, time_margin:-time_margin]
  X_to_test2 = X_to_test[space_margin:-space_margin, time_margin:-time_margin]
  mask = ~np.isnan(X_to_test2)
  return _rmsn(X_imputed2[mask],  X_to_test2[mask])

def R2(X_imputed, X_to_test, time_margin = 10, space_margin = 5):
  X_imputed2 = X_imputed[space_margin:-space_margin, time_margin:-time_margin]
  X_to_test2 = X_to_test[space_margin:-space_margin, time_margin:-time_margin]
  mask = ~np.isnan(X_to_test2)
  # print (X_to_test2[mask])
  # print (X_imputed2[mask])
  return r2_score(X_to_test2[mask], X_imputed2[mask])

def RMSE(X_imputed, X_to_test, time_margin = 10, space_margin = 5):
  X_imputed2 = X_imputed[space_margin:-space_margin, time_margin:-time_margin]
  X_to_test2 = X_to_test[space_margin:-space_margin, time_margin:-time_margin]
  mask = ~np.isnan(X_to_test2)
  return np.sqrt(((X_to_test2[mask] - X_imputed2[mask]) ** 2).mean())

def NRMSE(X_imputed, X_to_test, time_margin = 10, space_margin = 5):
  X_imputed2 = X_imputed[space_margin:-space_margin, time_margin:-time_margin]
  X_to_test2 = X_to_test[space_margin:-space_margin, time_margin:-time_margin]
  mask = ~np.isnan(X_to_test2)
  return np.sqrt(np.sum((X_to_test2[mask] - X_imputed2[mask]) ** 2) / np.sum(X_to_test2[mask] ** 2))

def SMAPE1(X_imputed, X_to_test, time_margin = 10, space_margin = 5):
  X_imputed2 = X_imputed[space_margin:-space_margin, time_margin:-time_margin]
  X_to_test2 = X_to_test[space_margin:-space_margin, time_margin:-time_margin]
  mask = ~np.isnan(X_to_test2)
  return np.mean(np.abs(X_to_test2[mask] - X_imputed2[mask]) / (X_to_test2[mask] + X_imputed2[mask]))

def SMAPE2(X_imputed, X_to_test, time_margin = 10, space_margin = 5):
  X_imputed2 = X_imputed[space_margin:-space_margin, time_margin:-time_margin]
  X_to_test2 = X_to_test[space_margin:-space_margin, time_margin:-time_margin]
  mask = ~np.isnan(X_to_test2)
  return np.sum(np.abs(X_to_test2[mask] - X_imputed2[mask])) / np.sum(X_to_test2[mask] + X_imputed2[mask])

def _rmsn(predictions, targets):
    return np.sqrt(np.sum((predictions - targets) ** 2) * len(predictions)) / np.sum(targets)
