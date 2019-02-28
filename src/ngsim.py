import os
import numpy as np
import pandas as pd
import datetime
import pytz

# from sortedcontainers import SortedDict

from paras import *


GLB_DEBUG = False

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
    sorted(self.vr_list, key = lambda x: (x.y, x.x), reverse = ascending)

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

  def build_from_processed(self, words, vr_dict):
    assert(len(words) > 1)
    self.veh_ID = np.int(words[0])
    self.vr_list = list(map(lambda x: vr_dict[np.int(x)], words[1:]))

  def add_vr(self, vr):
    assert (vr.veh_ID == self.veh_ID)
    self.vr_list.append(vr)

  def sort_time(self, ascending = True):
    sorted(self.vr_list, key = lambda x: (x.unixtime, x.veh_ID), reverse = ascending)

  def __str__(self):
    return ("Vehicle: veh_ID: {}, number of unixtimes: {}".format(self.veh_ID, len(self.vr_list)))
  
  def __repr__(self):
    return self.__str__()

  def to_string(self):
    return ','.join([str(e) for e in [self.veh_ID] + list(map(lambda x: x.ID, self.vr_list))])

class lidar():
  def __init__(self):
    self.lidar_ID = None




class mesh():
  def __init__(self):
    self.spatial_res = None
    self.temporal_res = None
    self.num_lane = None

  def init_storage(self):
    pass