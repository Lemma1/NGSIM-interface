import os
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from ngsim import *
from sensing import *
from measures import *
from simulator import *

def run_three_networks(p_rate = 0.05, meter = 50, miss_rate = 0.05, spd_noise = 3.28084 * 0.0, sensing_power = 2, 
                      sample_rate = 1000, s1 = 'SI', s2 = 'LR2'):



  name = 'i-80'
  data_folder = os.path.join('..', 'data')
  ng = ngsim_data(name)
  ng.load(os.path.join(data_folder, 'processed', name))
  ng.clean()

  TIME_MIN = list(filter(lambda x: x > 1113433300000 + 3000000, ng.snap_ordered_list))[0]
  TIME_MAX = ng.snap_ordered_list[-1]
  Y_MIN = 100
  Y_MAX = 1600
  SPATIAL_NUM = 60
  TEMPORAL_NUM = 90

  ng.down_sample(sample_rate = sample_rate)

  for veh_ID, v in ng.veh_dict.items():
  #     print (veh_ID, v)
      v.build_trajectory(name)


  r_list = list()
  lidar_veh_list = list()
  m2 = mesh(num_spatial_cells = SPATIAL_NUM, num_temporal_cells = TEMPORAL_NUM, name = name)
  m2.init_mesh(Y_MIN, Y_MAX, TIME_MIN, TIME_MAX)
  for veh_ID, v in ng.veh_dict.items():
      if np.random.rand() < p_rate:
          r_list.append(3.28084 * meter)
          lidar_veh_list.append(v)
          m2.update_vehilce(v)
  m2.update_qkv()
  mc = monitor_center(Y_MIN, Y_MAX, TIME_MIN, TIME_MAX, miss_rate = miss_rate, spd_noise = spd_noise, method = 'Tracking')
  mc.install_lidar(lidar_veh_list, r_list)
  mc.detect_all_snap(ng.snap_dict)
  m3 = mesh(num_spatial_cells = SPATIAL_NUM, num_temporal_cells = TEMPORAL_NUM, name = name)
  m3.init_mesh(Y_MIN, Y_MAX, TIME_MIN, TIME_MAX)
  ms = space_mesh(num_spatial_cells = SPATIAL_NUM, name = name)
  ms.build_lane_centerline(ng.snap_dict, TIME_MIN, TIME_MAX)
  ms.init_mesh(Y_MIN, Y_MAX)
  mc.reduce_to_mesh2(m3, ms, name)
  m3.update_qkv2()


  m_true = pickle.load(open("highrestracking60903.pickle", 'rb'))

  if sensing_power == 0:
    mk = m2
    mv = m2
  elif sensing_power == 1:
    mk = m3
    mv = m2
  elif sensing_power == 2:
    mk = m3
    mv = m3
  else:
    raise ("no sensing level")

  s = simulator([s1, s2], name, SPATIAL_NUM, TEMPORAL_NUM)
  s.load_observations(mk, mv)
  s.load_ground_truth(m_true)
  s.run_full_estimation()
  err = s.get_err(name)

  ss1 = s
  m_true1 = m_true

  k_df, v_df = process_err(err)


  name = 'us-101'

  data_folder = os.path.join('..', 'data')
  ng = ngsim_data(name)
  ng.load(os.path.join(data_folder, 'processed', name))
  ng.clean()

  TIME_MIN = ng.snap_ordered_list[0]
  TIME_MAX = ng.snap_ordered_list[-1]
  Y_MIN = 100
  Y_MAX = 2100
  SPATIAL_NUM = 60
  TEMPORAL_NUM = 90

  ng.down_sample(sample_rate = sample_rate)

  for veh_ID, v in ng.veh_dict.items():
  #     print (veh_ID, v)
      v.build_trajectory(name)


  r_list = list()
  lidar_veh_list = list()
  m2 = mesh(num_spatial_cells = SPATIAL_NUM, num_temporal_cells = TEMPORAL_NUM, name = name)
  m2.init_mesh(Y_MIN, Y_MAX, TIME_MIN, TIME_MAX)
  for veh_ID, v in ng.veh_dict.items():
      if np.random.rand() < p_rate:
          r_list.append(3.28084 * meter)
          lidar_veh_list.append(v)
          m2.update_vehilce(v)
  m2.update_qkv()
  mc = monitor_center(Y_MIN, Y_MAX, TIME_MIN, TIME_MAX, miss_rate = miss_rate, spd_noise = spd_noise, method = 'Tracking')
  mc.install_lidar(lidar_veh_list, r_list)
  mc.detect_all_snap(ng.snap_dict)
  m3 = mesh(num_spatial_cells = SPATIAL_NUM, num_temporal_cells = TEMPORAL_NUM, name = name)
  m3.init_mesh(Y_MIN, Y_MAX, TIME_MIN, TIME_MAX)
  ms = space_mesh(num_spatial_cells = SPATIAL_NUM, name = name)
  ms.build_lane_centerline(ng.snap_dict, TIME_MIN, TIME_MAX)
  ms.init_mesh(Y_MIN, Y_MAX)
  mc.reduce_to_mesh2(m3, ms, name)
  m3.update_qkv2()

  m_true = pickle.load(open(os.path.join(data_folder, 'processed', name, "highrestracking60902.pickle"), 'rb'))

  if sensing_power == 0:
    mk = m2
    mv = m2
  elif sensing_power == 1:
    mk = m3
    mv = m2
  elif sensing_power == 2:
    mk = m3
    mv = m3
  else:
    raise ("no sensing level")


  s = simulator([s1, s2], name, SPATIAL_NUM, TEMPORAL_NUM)
  s.load_observations(mk, mv)
  s.load_ground_truth(m_true)
  s.run_full_estimation()
  err = s.get_err(name)

  ss2 = s
  m_true2 = m_true

  k_df, v_df = process_err(err)


  name = 'lankershim'

  data_folder = os.path.join('..', 'data')
  ng = ngsim_data(name)
  ng.load(os.path.join(data_folder, 'processed',name))
  ng.clean()

  TIME_MIN = ng.snap_ordered_list[0]
  TIME_MAX = ng.snap_ordered_list[-1]
  Y_MIN = 100
  Y_MAX = 1500
  SPATIAL_NUM = 60
  TEMPORAL_NUM = 90

  ng.down_sample(sample_rate = sample_rate)

  for veh_ID, v in ng.veh_dict.items():
  #     print (veh_ID, v)
      v.build_trajectory(name)


  r_list = list()
  lidar_veh_list = list()
  m2 = mesh(num_spatial_cells = SPATIAL_NUM, num_temporal_cells = TEMPORAL_NUM, name = name)
  m2.init_mesh(Y_MIN, Y_MAX, TIME_MIN, TIME_MAX)
  for veh_ID, v in ng.veh_dict.items():
      if np.random.rand() < p_rate:
          r_list.append(3.28084 * meter)
          lidar_veh_list.append(v)
          m2.update_vehilce(v)
  m2.update_qkv()
  mc = monitor_center(Y_MIN, Y_MAX, TIME_MIN, TIME_MAX, miss_rate = miss_rate, spd_noise = spd_noise, method = 'Tracking')
  mc.install_lidar(lidar_veh_list, r_list)
  mc.detect_all_snap(ng.snap_dict)
  m3 = mesh(num_spatial_cells = SPATIAL_NUM, num_temporal_cells = TEMPORAL_NUM, name = name)
  m3.init_mesh(Y_MIN, Y_MAX, TIME_MIN, TIME_MAX)
  ms = space_mesh(num_spatial_cells = SPATIAL_NUM, name = name)
  ms.build_lane_centerline(ng.snap_dict, TIME_MIN, TIME_MAX)
  ms.init_mesh(Y_MIN, Y_MAX)
  mc.reduce_to_mesh2(m3, ms, name)
  m3.update_qkv2()


  m_true = pickle.load(open(os.path.join(data_folder, 'processed', name, "highrestracking60902.pickle"), 'rb'))

  if sensing_power == 0:
    mk = m2
    mv = m2
  elif sensing_power == 1:
    mk = m3
    mv = m2
  elif sensing_power == 2:
    mk = m3
    mv = m3
  else:
    raise ("no sensing level")

  s = simulator([s1, s2], name, SPATIAL_NUM, TEMPORAL_NUM)
  s.load_observations(mk, mv)
  s.load_ground_truth(m_true)
  s.run_full_estimation()
  err = s.get_err(name)

  ss3 = s
  m_true3 = m_true

  k_df, v_df = process_err(err)

  return [ss1, ss2, ss3, m_true1, m_true2, m_true3]