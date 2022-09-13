from common.goal import *
from common.stream import *
from common.learning import *
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import time
from sklearn import cluster, mixture, metrics, preprocessing
import numpy as np
import random
from sklearn.mixture import GaussianMixture
import sys
from scipy import stats
import statistics
import warnings 

class MAPE():
    def __init__(self, stream:Stream, classifiers, target_scaler):
        self.stream:Stream = stream
        # self.initial_training_cycle_num = initial_training_cycle_num
        self.maximum_class_num = 5
        self.classifiers = classifiers # order is matter for goal satisfaction
        # self.features = []
        self.targets_pl = []
        # self.targets_la = []
        self.targets_ec = []
        self.component_num = 1
        self.scaler = target_scaler
        self.proba_not_a_member_threshold = 0.001 # 3-sigma 
        self.out_of_class_limit_num = 3
        self.best_selected_targets = [] #(pl, la, ec)
        self.verified_count = []
        self.feature_size = 0
        self.probas = []
    def human_ordering(self, class_type):
        if(class_type == 1):
            return [1]
        return None
        
    def monitor_and_analyse(self):
        features, targets, verf_times = self.stream.read_current_cycle()
        # self.features.append(features)
        # here, always total verfication times is much less than 10 minutes (around 3 minutes)
        # otherwise the verification time of the selected features should be considered 
        self.feature_size = len(features)
        targets_pl = targets[TargetType.PACKETLOSS]
        # targets_la = targets[TargetType.LATENCY]
        targets_ec = targets[TargetType.ENERGY_CONSUMPTION]
        
        self.targets_pl.extend(targets_pl)
        # self.targets_la.extend(targets_la)
        self.targets_ec.extend(targets_ec)
        #fail-safe option
        selected_options_idx = 0
        
        probas = []
        detected_classes = []
        # is_training = True
        # select an appropriate adaptation option based on classification goal
        # if(self.stream.current_cyle < self.initial_training_cycle_num):
        #     pass
        # elif(self.stream.current_cyle == self.initial_training_cycle_num):
        #     self.scaler = preprocessing.MinMaxScaler()
        #     scaled_data = self.scaler.fit_transform(list(zip(self.targets_pl, self.targets_ec)))
        #     self.compnent_num, classifier = find_component_num(scaled_data)
        #     # print(self.compnent_num)
        #     for i in range(self.compnent_num):
        #         self.classifiers.append([classifier.means_[i], classifier.covariances_[i]])
        #     self.classifiers_order = self.human_ordering(1) 
        #     self.feature_size = len(features)
        # else:
            # # develope 3-sigma test
            # out_of_class = []
            # detected_classes = [[] for i in range(self.compnent_num)]
        # is_training = False
        for i in range(len(targets_pl)):                
            max_classifier_idx, max_prob = \
                    self.classification(targets_pl[i], targets_ec[i])
            
            # not all data will be used for analysis (determined in planning statge here)
            # this collection is happened for the matter validation part
            probas.append(max_prob)
            self.probas.append(max_prob)
            detected_classes.append(max_classifier_idx)
                
        self.plan(probas, detected_classes, targets_pl, targets_ec)
                # if(max_classifier_idx > -1):
                    # detected_classes[max_classifier_idx].append(i)
                    # if(self.classifiers_order[max_classifier_idx] == 1):
                    #     # choose the best option 
                    #     self.execute(i)
                    #     return
                        
                # else:
                #     out_of_class.append([targets_pl[i], targets_ec[i]])    
                        # print(self.stream.current_cyle)
                        # print(proba)
    
    def plan(self, probas, detected_classes, targets_pl, targets_ec):
        best_option_idx = -1
        # if(is_training):
        #     best_option_idx = np.argmin(targets_ec)
        #     self.best_selected_targets.append([targets_pl[best_option_idx], targets_la[best_option_idx], targets_ec[best_option_idx]])
        # else:
            # for i in range(self.out_of_class_limit_num):    
            #     selected_adaptation_option = random.sample(range(len(probas)), 1)[0]
            #     if(probas[selected_adaptation_option] > self.proba_not_a_member_threshold):
            #         best_option_idx = selected_adaptation_option
            #         break
            # if(best_option_idx < 0):
            #     best_option_idx = random.sample(range(len(probas)), 1)[0]
            #     self.verified_count.append(self.out_of_class_limit_num + 1)
            # else:
        options_idx_random_order = list(range(len(probas)))
        random.shuffle(options_idx_random_order)   
        
        num_verification_counter = 0
        for j in range(len(self.classifiers)):
            out_of_class_limit_counter = 0 
            for i in options_idx_random_order:
                if(j == 0):
                    num_verification_counter += 1    
                
                if(out_of_class_limit_counter > self.out_of_class_limit_num):
                    if(detected_classes[i] == j):
                        best_option_idx = i
                        break
                else:
                    if(detected_classes[i] == j):
                        if((probas[i] > self.proba_not_a_member_threshold)):
                            best_option_idx = i
                            break
                        else:
                            # print(probas[i])
                            out_of_class_limit_counter += 1
                            
            if(best_option_idx > -1):
                break
        
        
        if(best_option_idx < 0):
            # fail_safe_option
            best_option_idx = 0
            warnings.warn("no option classified as member of exisiting classes!")
            
        self.verified_count.append(num_verification_counter)
        self.best_selected_targets.append([targets_pl[best_option_idx], targets_ec[best_option_idx]])
            
        self.execute(best_option_idx)
                
    def execute(self, best_option_index):
        return 1
    
    def classification(self, targets_pl, targets_ec):
        max_classifier_idx = -1
        max_prob = -1
        for j in range(len(self.classifiers)):
            scaled_data = self.scaler.transform([[targets_pl, targets_ec]])
            x = scaled_data[0]
            m_dist_x = np.dot((x-self.classifiers[j][0]).transpose(), np.linalg.inv(self.classifiers[j][1]))
            m_dist_x = np.dot(m_dist_x, (x-self.classifiers[j][0]))
            proba = 1-stats.chi2.cdf(m_dist_x, 3)
            # if(proba < self.proba_not_a_member_threshold):
            #     pass
            # else:
            if(proba > max_prob):
                max_classifier_idx = j
                max_prob = proba
        return max_classifier_idx, max_prob