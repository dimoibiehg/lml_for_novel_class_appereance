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
from sklearn.mixture import GaussianMixture
import sys
from scipy import stats
import statistics
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm

class MAPE_ML2ASR():
    def __init__(self, stream:Stream, 
                 target_scaler, 
                 feature_scaler, 
                 classifiers, 
                 selected_features_indices,
                 initial_training_cycle_num = 40):
        self.stream:Stream = stream
        self.initial_training_cycle_num = initial_training_cycle_num
        self.maximum_class_num = 5
        self.classifiers = classifiers # order is matter for goal satisfaction
        self.features = []
        self.targets_pl = []
        self.targets_la = []
        self.targets_ec = []
        self.compnent_num = 1
        self.target_scaler = target_scaler
        self.feature_scaler = feature_scaler
        self.proba_not_a_member_threshold = 0.001 # 3-sigma 
        self.out_of_class_limit_num = 3
        self.best_selected_targets = [] #(pl, la, ec)
        self.pl_learning_model = SGDRegressor()
        self.ec_learning_model = SGDRegressor()
        self.selected_features_indices = selected_features_indices
        self.verified_count = []
    def human_ordering(self, class_type):
        if(class_type == 1):
            return [1]
        return None
        
    def monitor_and_analyse(self):
        features, targets, verf_times = self.stream.read_current_cycle()
        self.features.extend(features)
        # here, always total verfication times is much less than 10 minutes (around 3 minutes)
        # otherwise the verification time of the selected features should be considered 
        targets_pl = targets[TargetType.PACKETLOSS]
        targets_ec = targets[TargetType.ENERGY_CONSUMPTION]
        
        self.targets_pl.extend(targets_pl)
        self.targets_ec.extend(targets_ec)
        
        #fail-safe option
        selected_options_idx = 0
        
        probas = []
        detected_classes = []
        
        # select an appropriate adaptation option based on classification goal
        if(self.stream.current_cyle <= self.initial_training_cycle_num):
            is_training = True
            scaled_feature_data = self.feature_scaler.fit_transform(features)
            self.pl_learning_model.partial_fit(scaled_feature_data[:,self.selected_features_indices], targets_pl)
            self.ec_learning_model.partial_fit(scaled_feature_data[:,self.selected_features_indices], targets_ec)
            
            # print("fitted")
        else:
            # # develope 3-sigma test
            # out_of_class = []
            # detected_classes = [[] for i in range(self.compnent_num)]
            selected_scaled_feature_data = self.feature_scaler.transform(features)[:, self.selected_features_indices]
            targets_pl_predicts = self.pl_learning_model.predict(selected_scaled_feature_data)
            targets_ec_predicts = self.pl_learning_model.predict(selected_scaled_feature_data)
            
            is_training = False
            for i in range(len(targets_pl)):                
                max_classifier_idx, max_prob = \
                        self.classification(targets_pl[i], targets_ec[i])            
                
                # not all data will be used for analysis (determined in planning statge here)
                # this collection is happened for the matter validation part
                probas.append(max_prob)
                detected_classes.append(max_classifier_idx)
                
        self.plan(probas, detected_classes, features, targets_pl, targets_ec, is_training=is_training)
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
    
    def plan(self, probas, detected_classes, features, targets_pl, targets_ec, is_training = False):
        best_option_idx = -1
        num_verification_counter = 0
        if(is_training):
            best_option_idx = np.argmin(targets_ec)
            self.best_selected_targets.append([targets_pl[best_option_idx], targets_ec[best_option_idx]])
            num_verification_counter = len(targets_ec)
        else:
            probas_classes = []
            ind_classes = []
            for i in range(len(self.classifiers)):
                probas_classes.append([])
                ind_classes.append([])
                for ind, x in enumerate(detected_classes):
                    if x == i:
                        probas_classes[-1].append(probas[ind])
                        ind_classes[-1].append(ind)
            sorted_probas_arg_per_class = [np.argsort(x) for x in probas_classes]
            
            for j in range(len(self.classifiers)):
                sorted_probas_arg = sorted_probas_arg_per_class[j]
                out_of_class_limit_counter = 0 
                for i in range(1, len(sorted_probas_arg)+1):
                    ind = ind_classes[j][sorted_probas_arg[-i]]
                    max_classifier_idx, max_prob = \
                        self.classification(targets_pl[ind], targets_ec[ind])
                    
                    if(out_of_class_limit_counter > self.out_of_class_limit_num):
                        best_option_idx = ind
                        break
                    else:
                        if(max_prob > self.proba_not_a_member_threshold):
                            best_option_idx = ind
                            break
                        else:
                            out_of_class_limit_counter += 1
                    num_verification_counter += 1
                    self.pl_learning_model.partial_fit([features[ind]], [targets_pl[ind]])
                    self.ec_learning_model.partial_fit([features[ind]], [targets_ec[ind]])
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
            scaled_data = self.target_scaler.transform([[targets_pl, targets_ec]])
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