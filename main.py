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
from common.helper import retrieve_from_pickle
from scipy import stats

def best_config(mape):
    random.seed(1)
    ideal_pairs = [] 
    ideal_classes = []
    ideal_classifiers = retrieve_from_pickle('./files/ideal_classifiers.pkl')
    # print(ideal_classifiers)
    for i in tqdm(range(len(mape.best_selected_targets))):
        pairs = []
        classes= []
        for class_idx, classifier in enumerate(ideal_classifiers):
            for x,y in zip(mape.targets_pl[(i * mape.feature_size):((i+1) * mape.feature_size)], 
                        mape.targets_ec[(i * mape.feature_size):((i+1) * mape.feature_size)]):
                abs_mu_diff = np.abs(np.array(classifier[0]) - np.array([x,y]))
                if(np.sum([((x / (3*classifier[1][idx])) ** 2) for idx,x in enumerate(abs_mu_diff)]) <= 1):
                    pairs.append([x,y])
                    classes.append(class_idx)
            if(len(pairs) > 0):
                sample_idx = random.sample(list(range(len(pairs))), 1)[0]
                ideal_pairs.append(pairs[sample_idx])
                ideal_classes.append(classes[sample_idx])
                break
    return ideal_pairs, ideal_classes

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

def compute_ideal_distance_measure(mape, idx_list):
    ideal_distances = []
    _, ideal_classes = best_config(mape)
    for j in range(0, len(idx_list), 2):
        ideal_distances.append(0)
        for i in range(idx_list[j], idx_list[j+1]):  
            x = mape.best_selected_targets[i]
            max_classifier_idx_ideal = ideal_classes[i]
            max_classifier_idx_real, _ = ideal_classifcation(x[0], x[1])
            ideal_distances[-1] += (max_classifier_idx_real - max_classifier_idx_ideal)
        ideal_distances[-1] /= ((idx_list[j+1] - idx_list[j]) * 4)

    return ideal_distances

stream_addr = "./data"
p1 = list(range(1,151))
p2 = list(range(151, 250))
p3 = list(range(251, 351))
data_order = p1 + p2 + p3
data_ranges = [[p1[0], p1[-1]], [p2[0], p2[-1]], [p3[0], p3[-1]]]

# stream = fetch_stream(len(data_order), data_order =  data_order, stream_addr = stream_addr)
# target_scaler, classifiers = offline_processing_simple_mape(data_order, stream_addr)
# mape = MAPE(stream, classifiers, target_scaler)
# ideal_mape = IdealSelfAdaptation(mape)
# ideal_mape.start()

# sys.exit()

# stream = fetch_stream(len(data_order), data_order =  data_order, stream_addr = stream_addr)
# target_scaler, classifiers = offline_processing_simple_mape(data_order, stream_addr)
# mape = MAPE(stream, classifiers, target_scaler)
# lml_mape = LifelongSelfAdaptation(mape)
# lml_mape.start()

# print(compute_ideal_distance_measure(lml_mape.mape, \
#                                [0, 250, 250, len(lml_mape.mape.best_selected_targets)]))
# # result: [-0.085, -0.08585858585858586]
# sys.exit()


stream = fetch_stream(len(data_order), data_order =  data_order, stream_addr = stream_addr)
target_scaler, classifiers = offline_processing_simple_mape(data_order, stream_addr)
mape1 = MAPE(stream, classifiers, target_scaler)
for i in tqdm(range(348)):
    mape1.monitor_and_analyse()

ideal_pairs, ideal_classes = best_config(mape1)

names = ["Ideal Selection", "Managing System"]
colors = ["#E15F99", "lightseagreen", "rgb(127, 60, 141)", "rgb(175, 100, 88)"]
plot_distributions([["1-249"] * (249) + ["250-350"] * (len(mape1.best_selected_targets) - 249),
                    ["1-249"] * (249) + ["250-350"] * (len(mape1.best_selected_targets) - 249)], 
                   [[x[0] for x in mape1.best_selected_targets[0:249]] + 
                    [x[0] for x in mape1.best_selected_targets[250:]],
                    [x[0] for x in ideal_pairs[0:249]] + 
                    [x[0] for x in ideal_pairs[250:]]], names, colors, 
                    "Adaptation Cycles", "Packet loss (%)", 0.06, 0.99, 
                    is_group = True,file_name = "pl_problem_quality_attributes_dist",
                    )

plot_distributions([["1-249"] * (249) + ["250-350"] * (len(mape1.best_selected_targets) - 249),
                    ["1-249"] * (249) + ["250-350"] * (len(mape1.best_selected_targets) - 249)], 
                   [[x[1] for x in mape1.best_selected_targets[0:249]] + 
                    [x[1] for x in mape1.best_selected_targets[250:]],
                    [x[1] for x in ideal_pairs[0:249]] + 
                    [x[1] for x in ideal_pairs[250:]]], names, colors, 
                    "Adaptation Cycles", "Packet loss (%)", 0.06, 0.99, 
                    is_group = True,file_name = "ec_problem_quality_attributes_dist",
                    )
       
# print(compute_ideal_distance_measure(mape1, \
#                                [0, 250, 250, len(mape1.best_selected_targets)]))
# result: [-0.023, 0.3647959183673469]
sys.exit()
# random.seed(1)
# ideal_pairs = [] 
# ideal_classes = []
# ideal_classifiers = retrieve_from_pickle('./files/ideal_classifiers.pkl')
# print(ideal_classifiers)
# for i in tqdm(range(len(mape1.best_selected_targets))):
#     pairs = []
#     classes= []
#     for class_idx, classifier in enumerate(ideal_classifiers):
#         for x,y in zip(mape1.targets_pl[(i * mape1.feature_size):((i+1) * mape1.feature_size)], 
#                        mape1.targets_ec[(i * mape1.feature_size):((i+1) * mape1.feature_size)]):
#             abs_mu_diff = np.abs(np.array(classifier[0]) - np.array([x,y]))
#             if(np.sum([((x / (3*classifier[1][idx])) ** 2) for idx,x in enumerate(abs_mu_diff)]) <= 1):
#                 pairs.append([x,y])
#                 classes.append(class_idx)
#         if(len(pairs) > 0):
#             sample_idx = random.sample(list(range(len(pairs))), 1)[0]
#             ideal_pairs.append(pairs[sample_idx])
#             ideal_classes.append(classes[sample_idx])
#             break


# print(ideal_pairs)        
        

sys.exit()

# names = ["Reference", "ML2ASR"]
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
    
names = ["Reference", "State-of-the-art (ML2ASR)", "LML"]
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

