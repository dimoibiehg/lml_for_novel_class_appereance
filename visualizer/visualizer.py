import plotly.graph_objects as go
import plotly.express as px

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