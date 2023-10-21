import plotly.graph_objects as go
import numpy as np




def plot_classifier_results(currency_info, current_value):
    
    plot_bgcolor = "white"
    quadrant_colors = [plot_bgcolor,  "#2bad4e", "#85e043", "#eff229", "#f2a529", "#f25829"] 
    quadrant_text = ["", "<b>Strong Buy</b>", "<b>Buy</b>", "<b>Neutral</b>", "<b>Sell</b>", "<b>Strong Sell</b>"]
    n_quadrants = len(quadrant_colors) - 1
    
    
    min_value = 0
    max_value = 100
    hand_length = np.sqrt(2) / 4
    hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))
    
    fig = go.Figure(
        data=[
            go.Pie(
                values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
                rotation=90,
                hole=0.5,
                marker_colors=quadrant_colors,
                text=quadrant_text,
                textinfo="text",
                hoverinfo="skip",
            ),
        ],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=0,t=10,l=10,r=10),
            width=450,
            height=450,
            paper_bgcolor=plot_bgcolor,
            annotations=[
                go.layout.Annotation(
                    text=f"<b>{currency_info}",
                    x=0.5, xanchor="center", xref="paper",
                    y=0.25, yanchor="bottom", yref="paper",
                    showarrow=False,
                )
            ],
            shapes=[
                go.layout.Shape(
                    type="circle",
                    x0=0.48, x1=0.52,
                    y0=0.48, y1=0.52,
                    fillcolor="#333",
                    line_color="#333",
                ),
                go.layout.Shape(
                    type="line",
                    x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                    y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                    line=dict(color="#333", width=4)
                )
            ]
        )
    )
    
    return fig
    
    
    
    

    
    
    
    



    
    
 

