from mape.mape import MAPE
from common.learning import *
from visualizer.visualizer import *
import sys
import copy
import plotly.graph_objects as go
from sklearn import cluster, mixture, metrics
from common.helper import *

class IdealSelfAdaptation:
    def __init__(self, mape:MAPE, cycle_len=10, mape_cycle_len=350,
                 percentage_out_of_class_threshold=20):
        self.mape:MAPE = mape # for the matter of storage performance, using stored knowledge in the MAPE loop  
        self.cycle_len = cycle_len
        self.mape_cycle_len = mape_cycle_len
        self.percentage_out_of_class_threshold = percentage_out_of_class_threshold
    def start(self):
        counter_detection = 0
        action_list_idxs = [148, 248, 348]
        for i in tqdm(range(1, self.mape_cycle_len)):
            if(i in action_list_idxs):
                idx_action = action_list_idxs.index(i)
                # task identification
                out_of_class_counter = 0
                out_of_class_indices = []
                start_idx = ((action_list_idxs[idx_action-1]-1) * self.mape.feature_size) if idx_action > 0 else 0
                end_idx = (action_list_idxs[idx_action] - 1) * self.mape.feature_size
                num_of_total_verfied_counter = self.cycle_len * self.mape.feature_size
                # print(start_idx)
                # print(end_idx)
                for ind, x in enumerate(self.mape.probas[start_idx:end_idx]):
                    # print(x)
                    if(x < self.mape.proba_not_a_member_threshold):
                        out_of_class_counter += 1
                        out_of_class_indices.append(ind)
                out_of_class_percentage = (out_of_class_counter * 100.0) / num_of_total_verfied_counter
                if(out_of_class_percentage > self.percentage_out_of_class_threshold):
                    counter_detection += 1
                    #new task[s] (class[es]) detected
                    out_of_class_data = list(zip([self.mape.targets_pl[x+start_idx] for x in out_of_class_indices], 
                                                 [self.mape.targets_ec[x+start_idx] for x in out_of_class_indices]))
                    scaled_out_of_class_data = self.mape.scaler.transform(out_of_class_data)
                    component_num, classifier = find_component_num(scaled_out_of_class_data)
                    
                    # classifiers_to_plot = copy.deepcopy(self.mape.classifiers)
                    # for i in range(component_num):
                    #     classifiers_to_plot.append([classifier.means_[i], classifier.covariances_[i]])
                    
                    # inpput from human in task-based knowledge miner
                    if(counter_detection == 1):
                        polygon = (lambda x: ((13.8<=x[1]) and (x[1]<=14.5) and (7.0<=x[0]) and (x[0]<=22.0)))
                        
                        human_classes = [[scaled_out_of_class_data[ind] for ind, x in enumerate(out_of_class_data) if polygon(x)], 
                                        [scaled_out_of_class_data[ind] for ind, x in enumerate(out_of_class_data) if not polygon(x)]]
                        
                        # classifiers_to_plot = copy.deepcopy(self.mape.classifiers)
                        for human_class in human_classes:
                            bgmm = mixture.GaussianMixture(n_components = 1) 
                            bgmm.fit(human_class)
                            # classifiers_to_plot.append([bgmm.means_[0], bgmm.covariances_[0]])
                            self.mape.classifiers.append([bgmm.means_[0], bgmm.covariances_[0]])
                    
                    # print(len(self.mape.classifiers))
                    if(counter_detection == 2):
                        for i in range(component_num):
                            self.mape.classifiers.append([classifier.means_[i], classifier.covariances_[i]])
                        store_in_pickle("./files/ideal_classifiers.pkl", self.mape.classifiers)
                        # fig = plot_classifiers(self.mape.classifiers)
                        # fig.add_trace(go.Scatter(x = [t[0] for t in scaled_out_of_class_data], 
                        #             y = [t[1] for t in scaled_out_of_class_data], mode="markers", marker=dict(size=2, opacity=0.05)))
                        # # fig.add_trace(go.Scatter(x = [t[0] for t in out_of_class_data], 
                        # #                y = [t[1] for t in out_of_class_data], mode="markers", marker=dict(size=2, opacity=0.05)))
                        # fig.update_layout(showlegend=True)
                        # fig.show()
                        
                        
                        # print(f'component num:{component_num}')
                        # sys.exit()
                    # pass
                else:
                    # no new task detected
                    #do nothing
                    print("no new task identified")
            
                        
                
                
            self.mape.monitor_and_analyse()
    # def human_analysis_stages(data, stage):
    #     if(stage == 1):
            