import os
import numpy as np
from sklearn.metrics import r2_score

def MAE(X_imputed, X_to_test, time_margin = 5, space_margin = 2):
  X_imputed = X_imputed[space_margin:-space_margin, time_margin:-time_margin]
  X_to_test = X_to_test[space_margin:-space_margin, time_margin:-time_margin]
  mask = ~np.isnan(X_to_test)
  return ((X_imputed[mask] - X_to_test[mask]) ** 2).mean()

def RMSPE(X_imputed, X_to_test, time_margin = 5, space_margin = 2):
  X_imputed = X_imputed[space_margin:-space_margin, time_margin:-time_margin]
  X_to_test = X_to_test[space_margin:-space_margin, time_margin:-time_margin]
  mask = ~np.isnan(X_to_test)
  return np.sqrt((((X_imputed[mask] - X_to_test[mask])/X_to_test[mask]) ** 2).mean())

def RMSN(X_imputed, X_to_test, time_margin = 5, space_margin = 2):
  X_imputed = X_imputed[space_margin:-space_margin, time_margin:-time_margin]
  X_to_test = X_to_test[space_margin:-space_margin, time_margin:-time_margin]
  mask = ~np.isnan(X_to_test)
  return _rmsn(X_imputed[mask],  X_to_test[mask])

def R2(X_imputed, X_to_test, time_margin = 5, space_margin = 2):
  X_imputed = X_imputed[space_margin:-space_margin, time_margin:-time_margin]
  X_to_test = X_to_test[space_margin:-space_margin, time_margin:-time_margin]
  mask = ~np.isnan(X_to_test)
  return r2_score(X_to_test[mask], X_imputed[mask])


def _rmsn(predictions, targets):
    return np.sqrt(np.sum((predictions - targets) ** 2) * len(predictions)) / np.sum(targets)
