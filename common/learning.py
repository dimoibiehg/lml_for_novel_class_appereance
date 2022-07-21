from enum import Enum
from sklearn import cluster, mixture, metrics
from sklearn import preprocessing
import plotly.graph_objects as go
import numpy as np
class LearnerType(Enum):
    SKLEARN = 1
    KERAS = 2
    

def find_component_num(data, max_comp_num=15):    
    ks = range(1,max_comp_num+1)
    BIC = []
    BGMM : list[mixture.GaussianMixture] = []
    for i in ks:
        bgmm = mixture.GaussianMixture(n_components = i)
        bgmm.fit(data)
        BGMM.append(bgmm)
        BIC.append(bgmm.bic(np.array(data)))

    BIC_derivative = [(BIC[i] - BIC[i-1]) for i in range(1, len(BIC))]
    component_ind=  np.argmax(np.abs(BIC_derivative)) + 1 
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