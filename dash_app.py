import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dash_table # Import dash_table

# Use JupyterDash for running Dash apps directly in Colab
from jupyter_dash import JupyterDash

# Load the data (assuming it's saved from previous steps)
df_dashboard = pd.read_csv('diabetes_patient_data.csv')

# Define the features available for clustering
available_features = ['Glukosa', 'Tekanan_Darah', 'IMT', 'Insulin', 'Usia']

# Initialize the Dash app
app = JupyterDash(__name__)

app.layout = html.Div([
    html.H1("Dashboard Interaktif K-Means Clustering Diabetes"),

    html.Div([
        html.Div([
            html.Label("Pilih Fitur X-axis:"),
            dcc.Dropdown(
                id='x-axis-feature',
                options=[{'label': i, 'value': i} for i in available_features],
                value='Glukosa'
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("Pilih Fitur Y-axis:"),
            dcc.Dropdown(
                id='y-axis-feature',
                options=[{'label': i, 'value': i} for i in available_features],
                value='IMT'
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("Jumlah Kluster (K):"),
            dcc.Slider(
                id='num-clusters-slider',
                min=2,
                max=5,
                step=1,
                value=3,
                marks={i: str(i) for i in range(2, 6)}
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

    ], style={'border': '1px solid #ccc', 'padding': '10px', 'margin': '10px'}),

    # Container for dynamic cluster count cardboxes
    html.Div(id='cluster-count-cardboxes', style={'display': 'flex', 'flex-wrap': 'wrap', 'gap': '10px', 'margin-top': '20px', 'justify-content': 'center'}),

    dcc.Graph(id='kmeans-graph'),

    html.H3("Ringkasan Kluster K-Means (Mean Fitur per Kluster)"), # Added title for the table
    dash_table.DataTable( # Added DataTable
        id='kmeans-cluster-summary-table',
        columns=[],
        data=[]
    )
])

@app.callback(
    Output('kmeans-graph', 'figure'),
    Output('cluster-count-cardboxes', 'children'), # New output for cardboxes
    Output('kmeans-cluster-summary-table', 'data'), # New output for table data
    Output('kmeans-cluster-summary-table', 'columns'), # New output for table columns
    [Input('x-axis-feature', 'value'),
     Input('y-axis-feature', 'value'),
     Input('num-clusters-slider', 'value')]
)
def update_graph(x_feature, y_feature, num_clusters):
    if x_feature is None or y_feature is None:
        return {}, [], [], []

    # Make a copy to avoid SettingWithCopyWarning and ensure df_dashboard remains pristine for subsequent runs
    df_plot = df_dashboard.copy()

    # Select features for clustering
    features_to_cluster = df_plot[[x_feature, y_feature]].copy()

    # Standardize the data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_to_cluster)
    df_scaled = pd.DataFrame(features_scaled, columns=[x_feature, y_feature])

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_plot['Cluster'] = kmeans.fit_predict(df_scaled) # Assign to 'Cluster' for interactive results

    # Create scatter plot with Plotly Express
    fig = px.scatter(
        df_plot,
        x=x_feature,
        y=y_feature,
        color='Cluster',
        title=f'K-Means Clustering (K={num_clusters}) dengan {x_feature} dan {y_feature}',
        hover_data={'Cluster': True, x_feature: ':.2f', y_feature: ':.2f'},
        labels={'Cluster': 'Kluster ID'},
        color_continuous_scale=px.colors.qualitative.Plotly
    )

    # Add cluster count and center annotations
    cluster_counts = df_plot.groupby('Cluster').size().reset_index(name='Count')
    cluster_centers_scaled = kmeans.cluster_centers_

    # Inverse transform centers back to original scale for annotations
    inverse_transformed_centers = scaler.inverse_transform(cluster_centers_scaled)

    for i, count_row in cluster_counts.iterrows():
        cluster_id = count_row['Cluster']
        count = count_row['Count']
        center_x = inverse_transformed_centers[cluster_id, 0]
        center_y = inverse_transformed_centers[cluster_id, 1]
        fig.add_annotation(
            x=center_x,
            y=center_y,
            text=f"Kluster {cluster_id} (n={count})",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=0,
            ay=-30,
            font=dict(size=10, color="#636363"),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4
        )

    fig.update_layout(transition_duration=500)

    # Generate dynamic cardboxes for cluster counts
    cardboxes = []
    for _, row in cluster_counts.iterrows():
        cardboxes.append(
            html.Div([
                html.H4(f"Kluster {row['Cluster']}", style={'margin-bottom': '5px'}),
                html.P(f"Jumlah Pasien: {row['Count']}", style={'fontSize': '1.2em', 'fontWeight': 'bold'})
            ], style={
                'border': '1px solid #ddd',
                'borderRadius': '5px',
                'padding': '15px',
                'textAlign': 'center',
                'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)',
                'minWidth': '150px',
                'backgroundColor': '#f9f9f9'
            })
        )

    # Generate K-Means cluster summary table
    # Use available_features for the summary, not just x and y features
    cluster_summary = df_plot.groupby('Cluster')[available_features].mean().reset_index()
    summary_columns = [{'name': col, 'id': col} for col in cluster_summary.columns]
    summary_data = cluster_summary.to_dict('records')


    return fig, cardboxes, summary_data, summary_columns

# Workaround for JupyterDash TypeError when re-running cells
# This ensures the internal _server_threads state is clean before starting a new server.
if hasattr(app, '_server_threads'):
    app._server_threads = {}

app.run_server(mode='inline')
