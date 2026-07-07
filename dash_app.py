import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# Use JupyterDash for running Dash apps directly in Colab
from jupyter_dash import JupyterDash

# Load the data (assuming it's saved from previous steps)
df_dashboard = pd.read_csv('diabetes_patient_data.csv')

# Define the features available for clustering
available_features = ['Glukosa', 'Tekanan_Darah', 'IMT', 'Insulin', 'Usia']

# Initialize the Dash app
app = JupyterDash(__name__)

app.layout = html.Div([
    html.H1("K-Means Clustering Diabetes Pasien"),

    html.Div([
        html.Label("Pilih Fitur X-axis:"),
        dcc.Dropdown(
            id='x-axis-feature',
            options=[{'label': i, 'value': i} for i in available_features],
            value='Glukosa'
        ),
        html.Label("Pilih Fitur Y-axis:"),
        dcc.Dropdown(
            id='y-axis-feature',
            options=[{'label': i, 'value': i} for i in available_features],
            value='IMT'
        ),
        html.Label("Jumlah Kluster (K):"),
        dcc.Slider(
            id='num-clusters-slider',
            min=2,
            max=5,
            step=1,
            value=3,
            marks={i: str(i) for i in range(2, 6)}
        ),
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '20px'}),

    dcc.Graph(id='kmeans-graph')
])

@app.callback(
    Output('kmeans-graph', 'figure'),
    [Input('x-axis-feature', 'value'),
     Input('y-axis-feature', 'value'),
     Input('num-clusters-slider', 'value')]
)
def update_graph(x_feature, y_feature, num_clusters):
    if x_feature is None or y_feature is None:
        return {}

    # Select features for clustering
    features_to_cluster = df_dashboard[[x_feature, y_feature]].copy()

    # Standardize the data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_to_cluster)
    df_scaled = pd.DataFrame(features_scaled, columns=[x_feature, y_feature])

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_dashboard['Cluster'] = kmeans.fit_predict(df_scaled)

    # Create scatter plot with Plotly Express
    fig = px.scatter(
        df_dashboard,
        x=x_feature,
        y=y_feature,
        color='Cluster',
        title=f'K-Means Clustering (K={num_clusters}) dengan {x_feature} dan {y_feature}',
        hover_data={'Cluster': True, x_feature: ':.2f', y_feature: ':.2f'},
        labels={'Cluster': 'Kluster ID'}
    )

    fig.update_layout(transition_duration=500)

    return fig

# Jalankan aplikasi Dash
# Ini akan menampilkan dashboard langsung di output sel Colab.
app.run_server(mode='inline')