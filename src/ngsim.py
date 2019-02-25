import os
import numpy as np
import pandas as pd
import datetime
import pytz

from sortedcontainers import SortedDict

from paras import *



class ngsim_data():
  def __init__(self, name):
    self.name = name
    self.rv_list = list()

  def read_from_csv(self, filename):
    f = open(filename, 'rb')
    line = f.readline()
    while(line):
      line = f.readline().strip('\n').strip('\r').strip('\t')
      # print line
      if line == "":
        continue
      words = line.split(',')
      assert (len(words) == NUM_COLS)
      if words[GLB_loc_colidx] == self.name:
        # print line
        tmp_vr = vehicle_record(line)
        self.rv_list.append(tmp_vr)
    f.close()

class vehicle_record():
  def __init__(self):
    self.veh_ID = None
    # self.frame_ID = None
    self.unixtime = None

  def __init__(self, s1):
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

  def __str__(self):
    return ("Vehicle record: vehicle ID: {}, frame ID: {}, time: {}".format(
              self.veh_ID, self.frame_ID, self.time.strftime("%Y-%m-%d %H:%M:%S")))

  def __repr__(self):
    return self.__str__()


class snapshot():
  def __init__(self):
    self.unixtime = None
    self.vr_list = list()

  def add_vr(self, vr):
    assert (vr.unixtime == self.unixtime)
    self.vr_list.append(vr)


class vehicle():
  def __init__(self):
    self.veh_ID = None
    self.vr_list = list()

  def add_vr(self, vr):
    assert (vr.veh_ID == self.veh_ID)
    self.vr_list.append(vr)


class lidar():
  def __init__(self):
    self.lidar_ID = None