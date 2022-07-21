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

class MAPE():
    def __init__(self, stream:Stream, initial_training_cycle_num = 40):
        self.stream:Stream = stream
        self.initial_training_cycle_num = initial_training_cycle_num
        self.maximum_class_num = 5
        self.classifiers = [] # order is matter for goal satisfaction
        self.features = []
        self.targets_pl = []
        self.targets_la = []
        self.targets_ec = []
        self.compnent_num = 1
        self.scaler = []
    def monitor_and_analyse(self):
        features, targets, verf_times = self.stream.read_current_cycle()
        self.features.append(features)
        # here, always total verfication times is much less than 10 minutes (around 3 minutes)
        # otherwise the verification time of the selected features should be considered 
        targets_pl = targets[TargetType.PACKETLOSS]
        targets_la = targets[TargetType.LATENCY]
        targets_ec = targets[TargetType.ENERGY_CONSUMPTION]
        
        self.targets_pl.extend(targets_pl)
        self.targets_la.extend(targets_la)
        self.targets_ec.extend(targets_ec)
        
        #fail-safe option
        selected_options_idx = 0
        # select an appropriate adaptation option based on classification goal
        if(self.stream.current_cyle < self.initial_training_cycle_num):
            pass
        elif(self.stream.current_cyle == self.initial_training_cycle_num):
            self.scaler = preprocessing.MinMaxScaler()
            scaled_data = self.scaler.fit_transform(list(zip(targets_pl, targets_ec)))
            self.compnent_num, classifier = find_component_num(scaled_data)
            for i in range(self.compnent_num):
                self.classifiers.append([classifier.means_[i], classifier.covariances_[i]])
        else:
            # develope 3-sigma test
            for j in range(len(self.classifiers)):
                for i in range(len(targets_pl)):
                    scaled_data = self.scaler.transform([[targets_pl[i], targets_ec[i]]])
                    x = scaled_data[0]
                    m_dist_x = np.dot((x-self.classifiers[j][0]).transpose(),np.linalg.inv(self.classifiers[j][1]))
                    m_dist_x = np.dot(m_dist_x, (x-self.classifiers[j][0]))
                    print(1-stats.chi2.cdf(m_dist_x, 3))
                    # probs = classifier.predict_proba(scaled_data)
                    # print(probs)
                    # sys.exit()
            pass
            
            
            