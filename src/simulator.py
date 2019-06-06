import os 
import numpy as np
import pickle

from measures import *
from ngsim import *
from sensing import *
from paras import *


GLB_NUM_PROCESS = 2 
DENSITY_ITEM = 1
SPEED_ITEM = 2


class simulator():
  def __init__(self, process_list, name, num_spatial_cell, num_temporal_cell):
    # process list: sensing stage, density estimation, speed estimation
    assert (len(process_list) == GLB_NUM_PROCESS)
    self.process_list = process_list
    # assert(self.process_list[0] in ['S1', 'S2', 'S3'])
    assert(self.process_list[0] in ['NI', 'SI', 'KNN', 'II'])
    assert(self.process_list[1] in ['NI', 'SI', 'KNN', 'II', 'LR', 'LR2', 'RF', 'RF2'])
    self.name = name
    self.num_spatial_cell = num_spatial_cell
    self.num_temporal_cell = num_temporal_cell
    self.m_truth = None
    self.m_init_density = None
    self.m_init_speed = None
    self.m_full_density = None
    self.m_full_speed = None
    self.num_lane = len(GLB_LANE_CONSIDERED[self.name])

  def load_ground_truth(self, m_truth):
    self.m_truth = m_truth

  def load_observations(self, m_density, m_speed):
    self.m_init_density = m_density
    good_list = list()
    bad_list = list()
    for i in GLB_LANE_CONSIDERED[self.name]:
      if not np.isnan(self.m_init_density.lane_qkv[i][DENSITY_ITEM]).all():
        good_list.append(i)
      else:
        bad_list.append(i)
    if len(bad_list) > 0:
      for i in bad_list:
        near_i = min(good_list, key=lambda x: abs(x-i))
        self.m_init_density.lane_qkv[i][DENSITY_ITEM] = self.m_init_density.lane_qkv[near_i][DENSITY_ITEM].copy()
    self.m_init_speed = m_speed
    good_list = list()
    bad_list = list()
    for i in GLB_LANE_CONSIDERED[self.name]:
      if not np.isnan(self.m_init_speed.lane_qkv[i][SPEED_ITEM]).all():
        good_list.append(i)
      else:
        bad_list.append(i)
    if len(bad_list) > 0:
      for i in bad_list:
        near_i = min(good_list, key=lambda x: abs(x-i))
        self.m_init_speed.lane_qkv[i][SPEED_ITEM] = self.m_init_speed.lane_qkv[near_i][SPEED_ITEM].copy()

  def estimate_density(self):
    assert (self.m_init_density is not None)
    self.m_full_density = clone_part_mesh(self.m_init_density)
    if self.process_list[0] == 'NI':
      for i in GLB_LANE_CONSIDERED[self.name]:
        clf = vk_sensing('Naive')
        self.m_full_density.lane_qkv[i][DENSITY_ITEM] = clf.fit_transform(self.m_init_density.lane_qkv[i][DENSITY_ITEM])
    elif self.process_list[0] == 'SI':
      for i in GLB_LANE_CONSIDERED[self.name]:
        clf = vk_sensing("SoftImpute")
        clf.CVfit(self.m_init_density.lane_qkv[i][DENSITY_ITEM])
        self.m_full_density.lane_qkv[i][DENSITY_ITEM] = clf.fit_transform(self.m_init_density.lane_qkv[i][DENSITY_ITEM])
    elif self.process_list[0] == 'II':
      for i in GLB_LANE_CONSIDERED[self.name]:
        clf = vk_sensing("II")
        self.m_full_density.lane_qkv[i][DENSITY_ITEM] = clf.fit_transform(self.m_init_density.lane_qkv[i][DENSITY_ITEM])
    elif self.process_list[0] == 'KNN':
      for i in GLB_LANE_CONSIDERED[self.name]:
        clf = vk_sensing("KNN")
        clf.CVfit(self.m_init_density.lane_qkv[i][DENSITY_ITEM])
        self.m_full_density.lane_qkv[i][DENSITY_ITEM] = clf.fit_transform(self.m_init_density.lane_qkv[i][DENSITY_ITEM])   
    else:
      raise ("Not implemented")

  def estimate_speed(self):
    assert (self.m_full_density is not None)
    assert (self.m_init_speed is not None)
    self.spd_clf_list = list()
    self.m_full_speed = clone_part_mesh(self.m_init_speed)
    if self.process_list[1] == 'NI':
      for i in GLB_LANE_CONSIDERED[self.name]:
        clf = vk_sensing('Naive')
        self.m_full_speed.lane_qkv[i][SPEED_ITEM] = clf.fit_transform(self.m_init_speed.lane_qkv[i][SPEED_ITEM])
    elif self.process_list[1] == 'SI':
      for i in GLB_LANE_CONSIDERED[self.name]:
        clf = vk_sensing("SoftImpute")
        clf.CVfit(self.m_init_speed.lane_qkv[i][SPEED_ITEM])
        self.m_full_speed.lane_qkv[i][SPEED_ITEM] = clf.fit_transform(self.m_init_speed.lane_qkv[i][SPEED_ITEM])
    elif self.process_list[1] == 'KNN':
      for i in GLB_LANE_CONSIDERED[self.name]:
        clf = vk_sensing("KNN")
        clf.CVfit(self.m_init_speed.lane_qkv[i][SPEED_ITEM])
        self.m_full_speed.lane_qkv[i][SPEED_ITEM] = clf.fit_transform(self.m_init_speed.lane_qkv[i][SPEED_ITEM])
    elif self.process_list[1] == 'II':
      for i in GLB_LANE_CONSIDERED[self.name]:
        clf = vk_sensing('II')
        self.m_full_speed.lane_qkv[i][SPEED_ITEM] = clf.fit_transform(self.m_init_speed.lane_qkv[i][SPEED_ITEM])
    elif self.process_list[1] == 'LR':
      for i in GLB_LANE_CONSIDERED[self.name]:
        clf = speed_fitting()
        clf.CVfit(self.m_full_density.lane_qkv[i][DENSITY_ITEM], self.m_init_speed.lane_qkv[i][SPEED_ITEM])
        self.m_full_speed.lane_qkv[i][SPEED_ITEM] = clf.transform(self.m_full_density.lane_qkv[i][DENSITY_ITEM], self.m_init_speed.lane_qkv[i][SPEED_ITEM])
        self.spd_clf_list.append(clf)
    elif self.process_list[1] == 'LR2':
      for i in GLB_LANE_CONSIDERED[self.name]:
        # left_i = np.maximum(1, i - 1)
        # right_i = np.minimum(i+1, self.num_lane)
        left_i_idx = np.argmin(np.abs(np.array(GLB_LANE_CONSIDERED[self.name]) - (i-1)))
        left_i = GLB_LANE_CONSIDERED[self.name][left_i_idx]
        right_i_idx = np.argmin(np.abs(np.array(GLB_LANE_CONSIDERED[self.name]) - (i+1)))
        right_i = GLB_LANE_CONSIDERED[self.name][right_i_idx]
        clf = speed_fitting()
        clf.CVfit(self.m_full_density.lane_qkv[i][DENSITY_ITEM], self.m_init_speed.lane_qkv[i][SPEED_ITEM], 
                    left_Xk = self.m_full_density.lane_qkv[left_i][DENSITY_ITEM],
                    right_Xk = self.m_full_density.lane_qkv[right_i][DENSITY_ITEM])
        self.m_full_speed.lane_qkv[i][SPEED_ITEM] = clf.transform(self.m_full_density.lane_qkv[i][DENSITY_ITEM], self.m_init_speed.lane_qkv[i][SPEED_ITEM],
                                                                  left_Xk = self.m_full_density.lane_qkv[left_i][DENSITY_ITEM],
                                                                  right_Xk = self.m_full_density.lane_qkv[right_i][DENSITY_ITEM])
        self.spd_clf_list.append(clf)
    elif self.process_list[1] == 'RF':
      for i in GLB_LANE_CONSIDERED[self.name]:
        clf = speed_fitting()
        self.m_full_speed.lane_qkv[i][SPEED_ITEM] = clf.fit_transform(self.m_full_density.lane_qkv[i][DENSITY_ITEM], self.m_init_speed.lane_qkv[i][SPEED_ITEM])
        self.spd_clf_list.append(clf)
    elif self.process_list[1] == 'RF2':
      for i in GLB_LANE_CONSIDERED[self.name]:
        left_i_idx = np.argmin(np.abs(np.array(GLB_LANE_CONSIDERED[self.name]) - (i-1)))
        left_i = GLB_LANE_CONSIDERED[self.name][left_i_idx]
        right_i_idx = np.argmin(np.abs(np.array(GLB_LANE_CONSIDERED[self.name]) - (i+1)))
        right_i = GLB_LANE_CONSIDERED[self.name][right_i_idx]
        clf = speed_fitting()
        self.m_full_speed.lane_qkv[i][SPEED_ITEM] = clf.fit_transform(self.m_full_density.lane_qkv[i][DENSITY_ITEM], self.m_init_speed.lane_qkv[i][SPEED_ITEM],
                                                                  left_Xk = self.m_full_density.lane_qkv[left_i][DENSITY_ITEM],
                                                                  right_Xk = self.m_full_density.lane_qkv[right_i][DENSITY_ITEM])
        self.spd_clf_list.append(clf)
    else:
      raise ("Not implemented")



  def run_full_estimation(self):
    self.estimate_density()
    self.estimate_speed()

  def get_err(self, name):
    res_dict = dict()
    for i in GLB_LANE_CONSIDERED[name]:
      # print (i)
      res_dict[i] = dict()
      for item, est_m in zip([DENSITY_ITEM, SPEED_ITEM], [self.m_full_density, self.m_full_speed]):
        # print (item)
        res_dict[i][item] = dict()
        # print (est_m.lane_qkv[i][item])
        res_dict[i][item]['MAE'] = MAE(est_m.lane_qkv[i][item], self.m_truth.lane_qkv[i][item])
        res_dict[i][item]['RMSPE'] = RMSPE(est_m.lane_qkv[i][item], self.m_truth.lane_qkv[i][item])
        res_dict[i][item]['RMSN'] = RMSN(est_m.lane_qkv[i][item], self.m_truth.lane_qkv[i][item])
        res_dict[i][item]['R2'] = R2(est_m.lane_qkv[i][item], self.m_truth.lane_qkv[i][item])
        res_dict[i][item]['RMSE'] = RMSE(est_m.lane_qkv[i][item], self.m_truth.lane_qkv[i][item])
        res_dict[i][item]['NRMSE'] = NRMSE(est_m.lane_qkv[i][item], self.m_truth.lane_qkv[i][item])
        res_dict[i][item]['SMAPE1'] = SMAPE1(est_m.lane_qkv[i][item], self.m_truth.lane_qkv[i][item])
        res_dict[i][item]['SMAPE2'] = SMAPE2(est_m.lane_qkv[i][item], self.m_truth.lane_qkv[i][item])
    return res_dict




def process_err(err):
    lane_id = sorted(list(err.keys()))
    k_df = pd.DataFrame(index = lane_id, columns = ['MAE', 'NRMSE', 'R2', 'RMSE', 'RMSN', 'RMSPE', 'SMAPE1', 'SMAPE2'])
    v_df = pd.DataFrame(index = lane_id, columns = ['MAE', 'NRMSE', 'R2', 'RMSE', 'RMSN', 'RMSPE', 'SMAPE1', 'SMAPE2'])
    for ids in lane_id:
        for name in k_df.columns:
            k_df.loc[ids, name] = err[ids][1][name]
            v_df.loc[ids, name] = err[ids][2][name]
    return k_df, v_df