from common.goal import *
from common.stream import *
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import time
from sklearn import cluster, mixture, metrics
import numpy as np
import random
from mape.mape import MAPE
stream_addr = "./data"
p1 = list(range(1,151))
p2 = list(range(151, 250))
p3 = list(range(251, 351))
data_order = p1 + p2 + p3
data_ranges = [[p1[0], p1[-1]], [p2[0], p2[-1]], [p3[0], p3[-1]]]

stream = fetch_stream(len(data_order), data_order =  data_order, stream_addr = stream_addr)
mape = MAPE(stream)
mape.monitor_and_analyse()