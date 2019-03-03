import os
import numpy as np
import pandas as pd
import datetime
import pytz
from shapely.geometry import Polygon, LineString
# from sortedcontainers import SortedDict

from paras import *


GLB_DEBUG = False
GLB_ROUNDING_100MS = -2
GLB_UNIXTIME_GAP = 100
GLB_TIME_THRES = 10000
GLB_LANE_CONSIDERED = [1,2,3,4,5,6]

class ngsim_data():
  def __init__(self, name):
    self.name = name
    self.vr_dict = dict()
    self.snap_dict = dict()
    self.veh_dict = dict()
    self.snap_ordered_list = list()
    self.veh_ordered_list = list()

  def read_from_csv(self, filename):
    f = open(filename, 'rb')
    line = f.readline()
    counter = 0
    self.vr_dict = dict()
    self.snap_dict = dict()
    self.veh_dict = dict()
    while(line):
      if counter % 10000 == 0:
        print counter
      if counter > 10000 and GLB_DEBUG:
        break
      line = f.readline().strip('\n').strip('\r').strip('\t')
      # print line
      if line == "":
        continue
      words = line.split(',')
      
      assert (len(words) == NUM_COLS)
      if words[GLB_loc_colidx] == self.name:
        tmp_vr = vehicle_record()
        tmp_vr.build_from_raw(counter, line)
        self.vr_dict[tmp_vr.ID] = tmp_vr
        counter += 1

        if tmp_vr.unixtime not in self.snap_dict.keys():
          self.snap_dict[tmp_vr.unixtime] = snapshot(tmp_vr.unixtime)
        self.snap_dict[tmp_vr.unixtime].add_vr(tmp_vr)

        if tmp_vr.veh_ID not in self.veh_dict.keys():
          self.veh_dict[tmp_vr.veh_ID] = vehicle(tmp_vr.veh_ID)
        self.veh_dict[tmp_vr.veh_ID].add_vr(tmp_vr)

    self.snap_ordered_list = list(self.snap_dict.keys())
    self.veh_ordered_list = list(self.veh_dict.keys())
    self.snap_ordered_list.sort()
    self.veh_ordered_list.sort()

    for tmp_unixtime, tmp_snap in self.snap_dict.iteritems():
      tmp_snap.sort_vehs()
    for tmp_vehID, tmp_veh in self.veh_dict.iteritems():
      tmp_veh.sort_time()
    f.close()

  def dump(self, folder, vr_filename = 'vr_file.csv', v_filename = 'v_file.csv',
                          snapshot_filename = 'ss_file.csv'):
    f_vr = open(os.path.join(folder, vr_filename), 'wb')
    for vr_ID, vr in self.vr_dict.iteritems():
      f_vr.write(vr.to_string() + '\n')
    f_vr.close()
    f_v = open(os.path.join(folder, v_filename), 'wb')
    for _, v in self.veh_dict.items():
      f_v.write(v.to_string() + '\n')
    f_v.close()
    f_ss = open(os.path.join(folder, snapshot_filename), 'wb')
    for _, ss in self.snap_dict.items():
      f_ss.write(ss.to_string() + '\n')
    f_ss.close() 


  def load(self, folder, vr_filename = 'vr_file.csv', v_filename = 'v_file.csv',
                          snapshot_filename = 'ss_file.csv'):
    self.vr_dict = dict()
    self.snap_dict = dict()
    self.veh_dict = dict()
    f_vr = open(os.path.join(folder, vr_filename), 'rb')
    for line in f_vr:
      if line == '':
        continue
      words = line.rstrip('\n').rstrip('\r').split(',')
      assert(len(words) == 14)
      tmp_vr = vehicle_record()
      tmp_vr.build_from_processed(self.name, words)
      self.vr_dict[tmp_vr.ID] = tmp_vr
    f_vr.close()

    f_v = open(os.path.join(folder, v_filename), 'rb')
    for line in f_v:
      if line == '':
        continue
      words = line.rstrip('\n').rstrip('\r').split(',')
      assert(len(words) > 1)
      tmp_v = vehicle()
      tmp_v.build_from_processed(words, self.vr_dict)
      self.veh_dict[tmp_v.veh_ID] = tmp_v
    f_v.close()

    f_ss = open(os.path.join(folder, snapshot_filename), 'rb')
    for line in f_ss:
      if line == '':
        continue
      words = line.rstrip('\n').rstrip('\r').split(',')
      assert(len(words) > 1)
      tmp_ss = snapshot()
      tmp_ss.build_from_processed(words, self.vr_dict)
      self.snap_dict[tmp_ss.unixtime] = tmp_ss
    f_ss.close()

    self.snap_ordered_list = list(self.snap_dict.keys())
    self.veh_ordered_list = list(self.veh_dict.keys())
    self.snap_ordered_list.sort()
    self.veh_ordered_list.sort()

    for tmp_unixtime, tmp_snap in self.snap_dict.iteritems():
      tmp_snap.sort_vehs()
    for tmp_vehID, tmp_veh in self.veh_dict.iteritems():
      tmp_veh.sort_time()


class vehicle_record():
  def __init__(self):
    self.ID = None
    self.veh_ID = None
    # self.frame_ID = None
    self.unixtime = None

  def build_from_raw(self, ID, s1):
    self.ID = ID
    words = s1.split(',')
    assert(len(words) == NUM_COLS)
    tz = pytz.timezone(timezone_dict[words[GLB_loc_colidx]])
    self.veh_ID = np.int(words[GLB_vehID_colidx])
    # self.frame_ID = np.int(words[GLB_frmID_colidx])
    self.unixtime = np.int(words[GLB_glbtime_colidx]) 
    self.time = datetime.datetime.fromtimestamp(np.float(self.unixtime) / 1000,
                     tz)
    self.x = np.float(words[GLB_locx_colidx])
    self.y = np.float(words[GLB_locy_colidx])
    self.lat = np.float(words[GLB_glbx_colidx])
    self.lon = np.float(words[GLB_glby_colidx])
    self.spd = np.float(words[GLB_vehspd_colidx])
    self.acc = np.float(words[GLB_vehacc_colidx])
    self.lane_ID = np.int(words[GLB_laneID_colidx])
    # self.intersection_ID = np.int(words[GLB_interID_colidx])
    self.pred_veh_ID = np.int(words[GLB_pred_colidx])
    self.follow_veh_ID = np.int(words[GLB_follow_colidx])
    self.shead = np.float(words[GLB_shead_colidx])
    self.thead = np.float(words[GLB_thead_colidx])

  def build_from_processed(self, name, words):
    assert(len(words) == 14)
    self.ID = np.int(words[0])
    self.veh_ID = np.int(words[1])
    self.unixtime = np.int(words[2])
    tz = pytz.timezone(timezone_dict[name])
    self.time = datetime.datetime.fromtimestamp(np.float(self.unixtime) / 1000,
                     tz)
    self.x = np.float(words[3])
    self.y = np.float(words[4])
    self.lat = np.float(words[5])
    self.lon = np.float(words[6])
    self.spd = np.float(words[7])
    self.acc = np.float(words[8])
    self.lane_ID = np.int(words[9])
    self.pred_veh_ID = np.int(words[10])
    self.follow_veh_ID = np.int(words[11])
    self.shead = np.float(words[12])
    self.thead = np.float(words[13])

  def __str__(self):
    return ("Vehicle record: {}, vehicle ID: {}, unixtime: {}, time: {}, x: {}, y: {}".format(
              self.ID, self.veh_ID, self.unixtime, 
              self.time.strftime("%Y-%m-%d %H:%M:%S"), self.lane_ID, self.y, self.x))

  def __repr__(self):
    return self.__str__()

  def to_string(self):
    return ','.join([str(e) for e in [self.ID, self.veh_ID, self.unixtime, 
                                      self.x, self.y,
                                      self.lat, self.lon, self.spd, self.acc, self.lane_ID,
                                      self.pred_veh_ID, self.follow_veh_ID, 
                                      self.shead, self.thead]])

class snapshot():
  def __init__(self, unixtime = None):
    self.unixtime = unixtime
    self.vr_list = list()

  def build_from_processed(self, words, vr_dict):
    assert(len(words) > 1)
    self.unixtime = np.int(words[0])
    self.vr_list = list(map(lambda x: vr_dict[np.int(x)], words[1:]))

  def add_vr(self, vr):
    assert (vr.unixtime == self.unixtime)
    self.vr_list.append(vr)

  def sort_vehs(self, ascending = True):
    self.vr_list = sorted(self.vr_list, key = lambda x: (x.y, x.lane_ID), reverse = (not ascending))

  def __str__(self):
    return ("Snapshot: unixtime: {}, number of vehs: {}".format(self.unixtime, len(self.vr_list)))
  
  def __repr__(self):
    return self.__str__()

  def to_string(self):
    return ','.join([str(e) for e in [self.unixtime] + list(map(lambda x: x.ID, self.vr_list))])

class vehicle():
  def __init__(self, veh_ID = None):
    self.veh_ID = veh_ID
    self.vr_list = list()
    self.trajectory = dict()

  def build_from_processed(self, words, vr_dict):
    assert(len(words) > 1)
    self.veh_ID = np.int(words[0])
    self.vr_list = list(map(lambda x: vr_dict[np.int(x)], words[1:]))

  def add_vr(self, vr):
    assert (vr.veh_ID == self.veh_ID)
    self.vr_list.append(vr)

  def sort_time(self, ascending = True):
    self.vr_list = sorted(self.vr_list, key = lambda x: (x.unixtime), reverse = (not ascending))

  def __str__(self):
    return ("Vehicle: veh_ID: {}, number of unixtimes: {}".format(self.veh_ID, len(self.vr_list)))
  
  def __repr__(self):
    return self.__str__()

  def to_string(self):
    return ','.join([str(e) for e in [self.veh_ID] + list(map(lambda x: x.ID, self.vr_list))])

  # downsampl, interval unit: ms
  def down_sample(self, interval = 2000): 
    self.sampled_vr_list = list()
    cur_time = (np.round(np.random.rand() * interval + GLB_UNIXTIME_GAP/2, GLB_ROUNDING_100MS) 
                          + self.vr_list[0].unixtime)
    for tmp_vr in self.vr_list():
      if tmp_vr.unixtime - cur_time >= 2000:
        self.sampled_vr_list.append(tmp_vr)
        cur_time = tmp_vr.unixtime

  # def _get_stayed_lanes(self):
  #   return list(set(list(map(lambda x: x.lane_ID, self.vr_list))))

  def _get_lane_separated_vrs(self):
    lane2vr_dict = dict()
    # stayed_lanes = self._get_stayed_lanes()
    for vr in self.vr_list:
      if vr.lane_ID in GLB_LANE_CONSIDERED:
        if vr.lane_ID not in lane2vr_dict.keys():
          lane2vr_dict[vr.lane_ID] = list()
        lane2vr_dict[vr.lane_ID].append(vr)
    return lane2vr_dict

  def build_trajectory(self):
    self.trajectory = dict()
    lane2vr_dict = self._get_lane_separated_vrs()
    for lane_ID, tmp_vr_list in lane2vr_dict.iteritems():
      # print lane_ID
      tmp_traj = trajectory(GLB_TIME_THRES)
      tmp_traj.construct_trajectory(tmp_vr_list)
      # print self.vr_list
      # print tmp_traj.trajectory_list
      tmp_traj.build_poly_list()
      self.trajectory[lane_ID] = tmp_traj 


class trajectory():
  def __init__(self, thres):
    self.threshold = thres
    self.trajectory_list = list()
    self.polygon_list = list()
    self.polyline_list = list()

  def construct_trajectory(self, vr_list):
    # print vr_list
    assert (len(vr_list) > 0)
    self.trajectory_list = list()
    cur_time = vr_list[0].unixtime
    tmp_trj = [vr_list[0]]
    for tmp_vr in vr_list[1:]:
      if tmp_vr.unixtime - cur_time > self.threshold:
        if len(tmp_trj) > 1:
          self.trajectory_list.append(tmp_trj)
        tmp_trj = [tmp_vr]
      else:
        tmp_trj.append(tmp_vr)
      cur_time = tmp_vr.unixtime
    if len(tmp_trj) > 1:
      self.trajectory_list.append(tmp_trj)

  def build_poly_list(self):
    self.polygon_list = list()
    if len(self.trajectory_list) > 0:
      for traj in self.trajectory_list:
        tmp_polyline, tmp_polygon = self._build_poly(traj)
        self.polyline_list.append(tmp_polyline)
        self.polygon_list.append(tmp_polygon)

  def _build_poly(self, traj):
    assert(len(traj) > 1)
    point_list = list()
    for i in range(len(traj)):
      point_list.append((traj[i].unixtime, traj[i].y))
    tmp_polyline = LineString(point_list)
    for i in reversed(range(len(traj))):
      if traj[i].shead == 0: 
        point_list.append((traj[i].unixtime, traj[i].y + 1000))
      else:
        point_list.append((traj[i].unixtime, traj[i].y + traj[i].shead))
    p = Polygon(point_list)
    # print p
    assert(p.is_valid)
    return tmp_polyline, p



class lidar():
  def __init__(self):
    self.lidar_ID = None


class mesh():
  def __init__(self, num_spatial_cells = None, num_temporal_cells = None, num_lane = None):
    self.num_spatial_cells = num_spatial_cells
    self.num_temporal_cells = num_temporal_cells
    self.num_lane = num_lane
    self.mesh_storage = dict()

  def init_mesh(self, min_space, max_space, min_time, max_time):
    assert(self.num_spatial_cells is not None)
    assert(self.num_temporal_cells is not None)
    assert(self.num_lane is not None)
    self.mesh_storage = dict()
    space_breaks = np.linspace(min_space, max_space, self.num_spatial_cells + 1)
    time_breaks = np.linspace(min_time, max_time, self.num_temporal_cells + 1)
    for i in range(1, self.num_lane+1):
      self.mesh_storage[i] = dict()
      for j in range(self.num_spatial_cells):
        self.mesh_storage[i][j] = dict()
        for k in range(self.num_temporal_cells):
          tmp_p = Polygon([(time_breaks[k], space_breaks[j]), (time_breaks[k+1], space_breaks[j]), 
                            (time_breaks[k+1], space_breaks[j+1]), (time_breaks[k], space_breaks[j+1])])
          #[polygon, area, time, distance, q, k, v]
          self.mesh_storage[i][j][k] = [tmp_p, [], [], [], None, None, None]

  def update_vehilce(self, v):
    for lane_ID in v.trajectory.keys():
      tmp_traj = v.trajectory[lane_ID]
      for j in self.mesh_storage[lane_ID].keys():
        for k in self.mesh_storage[lane_ID][j].keys():
          tmp_poly = self.mesh_storage[lane_ID][j][k][0]
          assert(len(tmp_traj.polygon_list) == len(tmp_traj.polyline_list))
          for i in range(len(tmp_traj.polygon_list)):
            v_poly = tmp_traj.polygon_list[i]
            v_line = tmp_traj.polyline_list[i]

            tmp_v_line = tmp_poly.intersection(v_line)
            # print tmp_poly.exterior.coords.xy
            # print list(tmp_v_line.coords)
            # print tmp_v_line.is_empty
            if not tmp_v_line.is_empty:
              assert(len(tmp_v_line.coords) > 1)
              self.mesh_storage[lane_ID][j][k][2].append(tmp_v_line.coords[-1][0] - tmp_v_line.coords[0][0])
              self.mesh_storage[lane_ID][j][k][3].append(tmp_v_line.coords[-1][1] - tmp_v_line.coords[0][1])

              tmp_area = tmp_poly.intersection(v_poly).area
              assert(tmp_area>0)
              self.mesh_storage[lane_ID][j][k][1].append(tmp_area)