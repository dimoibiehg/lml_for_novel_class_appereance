import plotly.graph_objects as go
import plotly.express as px
from common.learning import norm_pdf_multivariate
import numpy as np
from scipy import linalg
import math 

def plot_distributions(x, y, names, colors, x_title, y_title, legend_x_pos, legend_y_pos, file_name = None, is_group = False, show_plot = True, save_plot = True):
    
    data = []
    for i in range(len(y)):
        data.append(go.Box(x = x[i], y =  y[i], name = names[i], marker_color = colors[i], boxmean='sd'))
    
    fig = go.Figure(data = data)
    fig.update_layout(
        yaxis = dict(title = y_title),
        xaxis = dict(title = x_title),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=legend_y_pos,
            xanchor="left",
            x=legend_x_pos
        )
    ) 
    if(is_group):
        fig.update_layout(boxmode='group')       
    
    if(show_plot):
        fig.show()
    
        if(save_plot):
            if(file_name is not None):
                fig.write_image("./figures/" + file_name + ".pdf")
                fig.write_html("./figures/" + file_name + ".htm")
                

def compute_classifier_contour(mean, cov):
    v, w = linalg.eigh(cov)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])
    # Plot an ellipse to show the Gaussian component
    a =  v[1]
    b =  v[0]
    x_origin = mean[0]
    y_origin = mean[1]
    x_ = [ ]
    y_ = [ ]

    for t in range(0,361,5):
        x = a*(math.cos(math.radians(t))) + x_origin
        x_.append(x)
        y = b*(math.sin(math.radians(t))) + y_origin
        y_.append(y)
    
    return x_, y_

def plot_classifiers(classifiers, x_sample_range=[-5.0, 200], x_sample_rate=60.0,
                                  y_sample_range=[-5.0, 200], y_sample_rate=60.0):
    # x = np.linspace(x_sample_range[0],x_sample_rate,x_sample_range[1])
    # y = np.linspace(y_sample_range[0],y_sample_rate,y_sample_range[1])
    # X, Y = np.meshgrid(x, y)
    # X_ = X.tolist()
    # X_ = [y for x in X_ for y in x]
    # Y_ = Y.tolist()
    # Y_ = [y for x in Y_ for y in x]
    # Samples = [[X_[i],Y_[i]] for i in range(len(X_))]
    
    # Zs = []
    # component_ind = len(classifiers)
    # for n in range(component_ind):
    #     Zs.append([])
    #     # print(n)
    #     # print(len(Zs[n]))
    #     for i in range(len(Samples)):
    #         Zs[n].append(norm_pdf_multivariate(x = np.array(Samples[i]),
    #                                            mu = classifiers[n][0],
    #                                            sigma = classifiers[n][1]))
    # fig:go.Figure = go.Figure(data=[])
    
    # for i in range(component_ind):
    #     fig.add_trace(go.Contour(z=Zs[i],
    #                 x=X_,
    #                 y=Y_,
    #                 contours_coloring='lines',
    #                     colorscale = px.colors.qualitative.Safe,
    #                 line_width=1,
    #                 showscale=False))

    # # fig.data = fig.data[::-1]
    # return fig
    component_ind = len(classifiers)
    fig:go.Figure = go.Figure() 
    for n in range(component_ind):
        print(classifiers[n])
        x_1_,y_1_ = compute_classifier_contour(classifiers[n][0], classifiers[n][1])
        x_2_,y_2_ = compute_classifier_contour(classifiers[n][0], [2 * x for x in classifiers[n][1]])
        x_3_,y_3_ = compute_classifier_contour(classifiers[n][0], [3 * x for x in classifiers[n][1]])
    
        fig.add_trace(go.Scatter(x=x_1_ , y=y_1_, mode='lines',
                          showlegend=False,
                          line=dict(width=1)))
        fig.add_trace(go.Scatter(x=x_2_ , y=y_2_, mode='lines',
                          showlegend=False,
                          line=dict(width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=x_3_ , y=y_3_, mode='lines',
                          showlegend=False,
                          line=dict(width=1, dash='dot')))
    return fig
    
    
    