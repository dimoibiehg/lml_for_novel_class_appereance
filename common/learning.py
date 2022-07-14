from enum import Enum
from sklearn import cluster, mixture, metrics
import plotly.graph_objects as go

class LearnerType(Enum):
    SKLEARN = 1
    KERAS = 2
    

def find_component_num(data, max_comp_num=15):    
    ks = range(1,max_comp_num+1)

    BIC = []
    BGMM = []
    for i in tqdm(ks):
        bgmm = mixture.GaussianMixture(n_components = i)
        bgmm.fit(data)
        BGMM.append(bgmm)
        BIC.append(bgmm.bic(np.array(data)))

    BIC_derivative = [(BIC[i] - BIC[i-1]) for i in range(1, len(BIC))]
    component_ind=  np.argmax(np.abs(BIC_derivative)) + 1 
    # fig = go.Figure(data=[go.Scatter(y = BIC)])
    # fig.show()
    return component_ind