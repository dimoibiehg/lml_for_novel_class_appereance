from enum import Enum
from sklearn import cluster, mixture, metrics
from sklearn import preprocessing
import plotly.graph_objects as go
import numpy as np
from common.goal import *
from common.stream import *
from tqdm import tqdm 
from sklearn.ensemble import ExtraTreesRegressor
from common.learning import *
class LearnerType(Enum):
    SKLEARN = 1
    KERAS = 2
    

class IDScaler:
    def fit(self, x):
        pass

    def partial_fit(self, x):
        pass

    def fit_transform(self, x):
        return x

    def transform(self, x):
        return x

def find_component_num(data, max_comp_num=4, second_derivative_threshold = 0.9):    
    ks = range(1,max_comp_num+1)
    BIC = []
    BGMM : list[mixture.GaussianMixture] = []
    for i in ks:
        bgmm = mixture.GaussianMixture(n_components = i)
        bgmm.fit(data)
        BGMM.append(bgmm)
        BIC.append(bgmm.bic(np.array(data)))

    #find valley
    BIC_derivative = [(BIC[i] - BIC[i-1]) for i in range(1, len(BIC))]
    normalized_BIC_derivative = preprocessing.MinMaxScaler().fit_transform([[x]for x in BIC_derivative])
    BIC_second_derivative = [(normalized_BIC_derivative[i] - normalized_BIC_derivative[i-1]) 
                                            for i in range(1, len(normalized_BIC_derivative))]
    
    # print(normalized_BIC_derivative)
    # print(BIC_second_derivative)
    pass_threshold_ind = 0
    for ind, x in enumerate(BIC_second_derivative):
        if(x > second_derivative_threshold):
            pass_threshold_ind = ind + 1
            break
    component_ind = pass_threshold_ind + 1 
    
    
    # fig = go.Figure(data=[go.Scatter(y = BIC)])
    # fig.show()
    return component_ind, BGMM[component_ind - 1]

def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    det = np.linalg.det(sigma)
    norm_const = 1.0/ ( math.pow((2*math.pi),float(size)/2) * math.pow(det,1.0/2) )
    x_mu = np.matrix(x - mu)
    inv = np.linalg.inv(sigma)        
    result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
    return norm_const * result

# offline processings
def data_gathering_for_offline_processing(data_order, stream_addr, initial_training_cycle_num = 40):
    stream = fetch_stream(len(data_order), data_order =  data_order, stream_addr = stream_addr)
    targets_pl = []
    targets_ec = []
    all_features = []
    for i in tqdm(range(initial_training_cycle_num)):
        features, targets, verf_times = stream.read_current_cycle()
        # here, always total verfication times is much less than 10 minutes (around 3 minutes)
        # otherwise the verification time of the selected features should be considered 
        targets_pl.extend(targets[TargetType.PACKETLOSS])
        targets_ec.extend(targets[TargetType.ENERGY_CONSUMPTION])
        all_features.extend(features)
    
    return all_features, targets_pl, targets_ec

def offline_processing_simple_mape(data_order, stream_addr, initial_training_cycle_num = 40):
    _, targets_pl, targets_ec = \
        data_gathering_for_offline_processing(data_order, stream_addr, initial_training_cycle_num=initial_training_cycle_num)

    classifiers = []
    # target_scaler = preprocessing.MinMaxScaler()
    target_scaler = IDScaler()
    total_collected_target_data = list(zip(targets_pl, targets_ec))
    #global range, world knowledge (these ranges can be tighter and changed domain by domain)
    target_scaler.fit([[0, 13.5], [50, 15.5]]) 
    scaled_target_data = target_scaler.transform(total_collected_target_data)
    compnent_num, classifier = find_component_num(scaled_target_data)
    for i in range(compnent_num):
        classifiers.append([classifier.means_[i], classifier.covariances_[i]])
    return target_scaler, classifiers

def offline_processing_mape_with_ml2asr(data_order, stream_addr, initial_training_cycle_num = 40):
    features, targets_pl, targets_ec = \
        data_gathering_for_offline_processing(data_order, stream_addr, initial_training_cycle_num=initial_training_cycle_num)

    classifiers = []    
    # target_scaler = preprocessing.MinMaxScaler()
    target_scaler = IDScaler()
    total_collected_target_data = list(zip(targets_pl, targets_ec))
    #global range, world knowledge (these ranges can be tighter and changed domain by domain)
    target_scaler.fit([[0, 13.5], [50, 15.5]])
    scaled_target_data = target_scaler.transform(total_collected_target_data)
    compnent_num, classifier = find_component_num(scaled_target_data)
    for i in range(compnent_num):
        classifiers.append([classifier.means_[i], classifier.covariances_[i]])
        
    
    feature_scaler = preprocessing.StandardScaler()
    scaled_feature_data = feature_scaler.fit_transform(features)
    reg = ExtraTreesRegressor(random_state=50)
    reg.fit(scaled_feature_data, total_collected_target_data)
    selected_features_indices = [ind for ind, x in enumerate(reg.feature_importances_) if x > 0.0]
    
    return target_scaler, feature_scaler, classifiers, selected_features_indices

