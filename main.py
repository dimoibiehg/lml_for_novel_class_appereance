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
from lifelong_self_adaptation.lifelong_self_adaptation import LifelongSelfAdaptation
from ideal_self_adaptation.ideal_self_adaptation import IdealSelfAdaptation
import sys

def ideal_classifcation(targets_pl, targets_ec):
    max_classifier_idx = -1
    max_prob = -1
    classifiers = retrieve_from_pickle('./files/ideal_classifiers.pkl')
    
    for j in range(len(classifiers)):
        x = [targets_pl, targets_ec]
        m_dist_x = np.dot((x-classifiers[j][0]).transpose(), np.linalg.inv(classifiers[j][1]))
        m_dist_x = np.dot(m_dist_x, (x-classifiers[j][0]))
        proba = 1-stats.chi2.cdf(m_dist_x, 3)
        # if(proba < self.proba_not_a_member_threshold):
        #     pass
        # else:
        if(proba > max_prob):
            max_classifier_idx = j
            max_prob = proba
    return max_classifier_idx, max_prob

stream_addr = "./data"
p1 = list(range(1,151))
p2 = list(range(151, 250))
p3 = list(range(251, 351))
data_order = p1 + p2 + p3
data_ranges = [[p1[0], p1[-1]], [p2[0], p2[-1]], [p3[0], p3[-1]]]

stream = fetch_stream(len(data_order), data_order =  data_order, stream_addr = stream_addr)
target_scaler, classifiers = offline_processing_simple_mape(data_order, stream_addr)
mape = MAPE(stream, classifiers, target_scaler)
ideal_mape = IdealSelfAdaptation(mape)
ideal_mape.start()

sys.exit()

stream = fetch_stream(len(data_order), data_order =  data_order, stream_addr = stream_addr)
target_scaler, classifiers = offline_processing_simple_mape(data_order, stream_addr)
mape = MAPE(stream, classifiers, target_scaler)
lml_mape = LifelongSelfAdaptation(mape)
lml_mape.start()

# sys.exit()


stream = fetch_stream(len(data_order), data_order =  data_order, stream_addr = stream_addr)
target_scaler, classifiers = offline_processing_simple_mape(data_order, stream_addr)
mape1 = MAPE(stream, classifiers, target_scaler)
for i in tqdm(range(348)):
    mape1.monitor_and_analyse()
    
ideal_classifiers = retrieve_from_pickle('./files/ideal_classifiers.pkl')

# names = ["ًReference", "ML2ASR"]
# colors = ["#E15F99", "lightseagreen", "rgb(127, 60, 141)"]#, "rgb(175, 100, 88)"]
# plot_distributions([["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(mape1.best_selected_targets) - 249)],
#                    [[x[0] for x in mape1.best_selected_targets[0:150]] + 
#                     [x[0] for x in mape1.best_selected_targets[150:249]] +
#                     [x[0] for x in mape1.best_selected_targets[250:]]], names, colors, 
#                     "Different scenarios", "Packet loss (%)", 0.06, 0.99, 
#                     is_group = True,file_name = "pl_compare_with_drift_distribution",
#                     )

# plot_distributions([["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(mape1.best_selected_targets) - 249)],
#                    [[x[1] for x in mape1.best_selected_targets[0:150]] + 
#                     [x[1] for x in mape1.best_selected_targets[150:249]] +
#                     [x[1] for x in mape1.best_selected_targets[250:]]], names, colors, 
#                     "Different scenarios", "Packet loss (%)", 0.06, 0.99, 
#                     is_group = True,file_name = "ec_compare_with_drift_distribution",
#                     )

stream = fetch_stream(len(data_order), data_order =  data_order, stream_addr = stream_addr)
target_scaler, feature_scaler, classifiers, selected_features_indices = \
                                            offline_processing_mape_with_ml2asr(data_order, stream_addr,)
mape2 = MAPE_ML2ASR(stream, target_scaler, feature_scaler, classifiers, selected_features_indices)
for i in tqdm(range(348)):
    mape2.monitor_and_analyse()
    
# stream = fetch_stream(len(data_order), data_order =  data_order, stream_addr = stream_addr)
# # mape = MAPE(stream)
# mape2 = MAPE_ML2ASR(stream)
# for i in tqdm(range(348)):
#     mape2.monitor_and_analyse()




# print(len(mape.best_selected_targets))
    
names = ["ًReference", "State-of-the-art (ML2ASR)", "LML"]
colors = ["#E15F99", "lightseagreen", "rgb(127, 60, 141)", "rgb(175, 100, 88)"]
plot_distributions([["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(mape1.best_selected_targets) - 249),
                    ["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(mape2.best_selected_targets) - 249),
                    ["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(lml_mape.mape.best_selected_targets) - 249)], 
                   [[x[0] for x in mape1.best_selected_targets[0:150]] + 
                    [x[0] for x in mape1.best_selected_targets[150:249]] +
                    [x[0] for x in mape1.best_selected_targets[250:]],
                    [x[0] for x in mape2.best_selected_targets[0:150]] + 
                    [x[0] for x in mape2.best_selected_targets[150:249]] +
                    [x[0] for x in mape2.best_selected_targets[250:]],
                    [x[0] for x in lml_mape.mape.best_selected_targets[0:150]] + 
                    [x[0] for x in lml_mape.mape.best_selected_targets[150:249]] +
                    [x[0] for x in lml_mape.mape.best_selected_targets[250:]]], names, colors, 
                    "Adaptation Cycles", "Packet loss (%)", 0.06, 0.99, 
                    is_group = True,file_name = "pl_compare_with_drift_distribution",
                    )

plot_distributions([["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(mape1.best_selected_targets) - 249),
                    ["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(mape2.best_selected_targets) - 249),
                    ["1-150"] * 150 + ["151-249"] * 99 + ["250-350"] * (len(lml_mape.mape.best_selected_targets) - 249)], 
                   [[x[1] for x in mape1.best_selected_targets[0:150]] + 
                    [x[1] for x in mape1.best_selected_targets[150:249]] +
                    [x[1] for x in mape1.best_selected_targets[250:]],
                    [x[1] for x in mape2.best_selected_targets[0:150]] + 
                    [x[1] for x in mape2.best_selected_targets[150:249]] +
                    [x[1] for x in mape2.best_selected_targets[250:]],
                    [x[1] for x in lml_mape.mape.best_selected_targets[0:150]] + 
                    [x[1] for x in lml_mape.mape.best_selected_targets[150:249]] +
                    [x[1] for x in lml_mape.mape.best_selected_targets[250:]]], names, colors, 
                    "Adaptation Cycles", "Energy Consumption (mC)", 0.06, 0.25, 
                    is_group = True,file_name = "ec_compare_with_drift_distribution",
                    )

