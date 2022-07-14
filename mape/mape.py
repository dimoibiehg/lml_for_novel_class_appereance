from common.goal import *
from common.stream import *
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import time
from sklearn import cluster, mixture, metrics
import numpy as np
import random

class MAPE():
    def __init__(self, stream:Stream):
        self.stream:Stream = stream
    
    def monitor_and_analyse(self):
        features, targets, verf_times = self.stream.read_current_cycle()
        # here, always verfication time is less than 10 minutes
        # otherwise the verification time of the selected features should be considered 
        targets_pl = targets[TargetType.PACKETLOSS]
        targets_la = targets[TargetType.LATENCY]
        targets_ec = targets[TargetType.ENERGY_CONSUMPTION]