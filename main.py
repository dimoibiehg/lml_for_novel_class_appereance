from common.goal import *
from common.stream import *
from common.learning import *
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import time
from sklearn import cluster, mixture, metrics
import numpy as np
import random
from mape.mape import MAPE
from mape.mape_ml2asr import MAPE_ML2ASR
from visualizer.visualizer import plot_distributions
from sklearn import cluster, mixture, metrics, preprocessing

stream_addr = "./data"
p1 = list(range(1,151))
p2 = list(range(151, 250))
p3 = list(range(251, 351))
data_order = p1 + p2 + p3
data_ranges = [[p1[0], p1[-1]], [p2[0], p2[-1]], [p3[0], p3[-1]]]



stream = fetch_stream(len(data_order), data_order =  data_order, stream_addr = stream_addr)
target_scaler, classifiers = offline_processing_simple_mape()
mape1 = MAPE(stream, classifiers, target_scaler)
for i in tqdm(range(348)):
    mape1.monitor_and_analyse()

names = ["ًReference", "ML2ASR"]
colors = ["#E15F99", "lightseagreen", "rgb(127, 60, 141)"]#, "rgb(175, 100, 88)"]
plot_distributions([["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(mape1.best_selected_targets) - 249)],
                   [[x[0] for x in mape1.best_selected_targets[0:150]] + 
                    [x[0] for x in mape1.best_selected_targets[150:249]] +
                    [x[0] for x in mape1.best_selected_targets[250:]]], names, colors, 
                    "Different scenarios", "Packet loss (%)", 0.06, 0.99, 
                    is_group = True,file_name = "pl_compare_with_drift_distribution",
                    )

plot_distributions([["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(mape1.best_selected_targets) - 249)],
                   [[x[1] for x in mape1.best_selected_targets[0:150]] + 
                    [x[1] for x in mape1.best_selected_targets[150:249]] +
                    [x[1] for x in mape1.best_selected_targets[250:]]], names, colors, 
                    "Different scenarios", "Packet loss (%)", 0.06, 0.99, 
                    is_group = True,file_name = "ec_compare_with_drift_distribution",
                    )


# stream = fetch_stream(len(data_order), data_order =  data_order, stream_addr = stream_addr)
# # mape = MAPE(stream)
# mape2 = MAPE_ML2ASR(stream)
# for i in tqdm(range(348)):
#     mape2.monitor_and_analyse()


# print(len(mape.best_selected_targets))
    
# names = ["ًReference", "ML2ASR"]
# colors = ["#E15F99", "lightseagreen", "rgb(127, 60, 141)"]#, "rgb(175, 100, 88)"]
# plot_distributions([["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(mape1.best_selected_targets) - 249),
#                     ["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(mape2.best_selected_targets) - 249)], 
#                    [[x[0] for x in mape1.best_selected_targets[0:150]] + 
#                     [x[0] for x in mape1.best_selected_targets[150:249]] +
#                     [x[0] for x in mape1.best_selected_targets[250:]],
#                     [x[0] for x in mape2.best_selected_targets[0:150]] + 
#                     [x[0] for x in mape2.best_selected_targets[150:249]] +
#                     [x[0] for x in mape2.best_selected_targets[250:]]], names, colors, 
#                     "Different scenarios", "Packet loss (%)", 0.06, 0.99, 
#                     is_group = True,file_name = "pl_compare_with_drift_distribution",
#                     )

# plot_distributions([["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(mape1.best_selected_targets) - 249),
#                     ["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(mape2.best_selected_targets) - 249)], 
#                    [[x[2] for x in mape1.best_selected_targets[0:150]] + 
#                     [x[2] for x in mape1.best_selected_targets[150:249]] +
#                     [x[2] for x in mape1.best_selected_targets[250:]],
#                     [x[2] for x in mape2.best_selected_targets[0:150]] + 
#                     [x[2] for x in mape2.best_selected_targets[150:249]] +
#                     [x[2] for x in mape2.best_selected_targets[250:]]], names, colors, 
#                     "Different scenarios", "Packet loss (%)", 0.06, 0.99, 
#                     is_group = True,file_name = "ec_compare_with_drift_distribution",
#                     )

