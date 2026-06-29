# !pip install dash==2.11.1 dash-bootstrap-components==1.4.0
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc


def create_patient_data(num_patients: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    data = {
        'patient_id': range(1, num_patients + 1),
        'age': np.random.randint(18, 70, num_patients),
        'gender': np.random.choice(['Male', 'Female'], num_patients, p=[0.5, 0.5]),
        'symptom_fever': np.random.choice([0, 1], num_patients, p=[0.6, 0.4]),
        'symptom_cough': np.random.choice([0, 1], num_patients, p=[0.55, 0.45]),
        'symptom_sore_throat': np.random.choice([0, 1], num_patients, p=[0.7, 0.3]),
        'symptom_fatigue': np.random.choice([0, 1], num_patients, p=[0.5, 0.5]),
        'symptom_headache': np.random.choice([0, 1], num_patients, p=[0.65, 0.35]),
        'diagnosis': np.random.choice(['Flu', 'Common Cold', 'Bronchitis', 'Allergy'], num_patients, p=[0.3, 0.4, 0.2, 0.1])
    }
    df = pd.DataFrame(data)

    def assign_diagnosis(row):
        if row['symptom_fever'] == 1 and row['symptom_cough'] == 1:
            if row['symptom_sore_throat'] == 1:
                return 'Flu'
            return 'Bronchitis'
        if row['symptom_cough'] == 1 and row['symptom_sore_throat'] == 1:
            return 'Common Cold'
        if row['symptom_fatigue'] == 1 and row['symptom_headache'] == 1:
            return 'Flu'
        return row['diagnosis']

    df['diagnosis'] = df.apply(assign_diagnosis, axis=1)
    bins = [18, 30, 45, 60, 70]
    labels = ['18-29 (Dewasa)', '30-44 (Orang Tua)', '45-59 (Lansia)', '60-70 (Lansia Lanjut)']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    # Convert categorical columns to string type to avoid FutureWarning with plotly.express
    df['diagnosis'] = df['diagnosis'].astype(str)
    df['gender'] = df['gender'].astype(str)
    df['age_group'] = df['age_group'].astype(str)

    return df


df = create_patient_data()



def build_diagnosis_figure(filtered_df: pd.DataFrame):
    fig = px.histogram(
        filtered_df,
        x='diagnosis',
        color='diagnosis',
        title='Distribusi Diagnosa Penyakit',
        template='plotly_white',
        text_auto=True
    )

    fig.update_layout(
        xaxis_title_text='Diagnosis Penyakit',
        yaxis_title_text='Jumlah Pasien',
        height=430,
        bargap=0.2,
        barmode='group'
    )
    return fig


def build_symptom_figure(filtered_df: pd.DataFrame):
    symptom_cols = [
        'symptom_fever',
        'symptom_cough',
        'symptom_sore_throat',
        'symptom_fatigue',
        'symptom_headache'
    ]
    symptoms = filtered_df.melt(
        id_vars=['patient_id'],
        value_vars=symptom_cols,
        var_name='Symptom',
        value_name='Present'
    )
    symptoms = symptoms[symptoms['Present'] == 1].copy()
    symptoms['Symptom'] = symptoms['Symptom'].str.replace('symptom_', '').str.replace('_', ' ').str.title()
    fig = px.histogram(
        symptoms,
        x='Symptom',
        color='Symptom',
        title='Prevalensi Gejala pada Dataset',
        template='plotly_white',
        text_auto=True
    )

    fig.update_layout(
        yaxis_title='Jumlah Pasien',
        xaxis_title='Gejala',
        showlegend=True,
        height=430,
        bargap=0.25,
        barmode='group'
    )
    return fig


def build_age_figure(filtered_df: pd.DataFrame):
    fig = px.box(
        filtered_df,
        x='diagnosis',
        y='age',
        color='diagnosis',
        title='Distribusi Usia per Diagnosa',
        template='plotly_white'
    )
    fig.update_layout(xaxis_title='Diagnosis', yaxis_title='Usia', showlegend=False)
    return fig


def build_sunburst_figure(filtered_df: pd.DataFrame):
    fig = px.sunburst(
        filtered_df,
        path=['diagnosis', 'gender', 'age_group'],
        values='patient_id',
        color='diagnosis',
        title='Distribusi Pasien Berdasarkan Diagnosis, Gender, dan Kelompok Usia',
        template='plotly_white'
    )
    fig.update_traces(hovertemplate='<b>%{label}</b><br>Jumlah Pasien: %{value}<br>Persentase: %{percentParent:.2%}')
    return fig


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Dashboard Plotly Diagnosa Penyakit'

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1('Dashboard Plotly Diagnosa Penyakit', className='text-center text-primary mb-4'), width=12)
    ]),
    dbc.Row([
        dbc.Col(
            [
                dbc.Label('Pilih Diagnosis'),
                dcc.Dropdown(
                    id='diagnosis-dropdown',
                    options=[{'label': diag, 'value': diag} for diag in sorted(df['diagnosis'].unique())] + [{'label': 'All', 'value': 'All'}],
                    value='All',
                    clearable=False
                )
            ],
        md=4),
        dbc.Col(
            [
                dbc.Label('Pilih Gender'),
                dcc.Dropdown(
                    id='gender-dropdown',
                    options=[{'label': gender, 'value': gender} for gender in sorted(df['gender'].unique())] + [{'label': 'All', 'value': 'All'}],
                    value='All',
                    clearable=False
                )
            ],
        md=4),
        dbc.Col(
            [
                dbc.Label('Pilih Kelompok Usia'),
                dcc.Dropdown(
                    id='age-group-dropdown',
                    options=[{'label': age, 'value': age} for age in df['age_group'].unique()] + [{'label': 'All', 'value': 'All'}],
                    value='All',
                    clearable=False
                )
            ],
        md=4),
    ], className='mb-4'),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Total Pasien', className='card-title'),
                html.H2(id='total-patients', className='card-text')
            ])
        ], color='primary', inverse=True), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Diagnosis Unik', className='card-title'),
                html.H2(id='unique-diagnosis', className='card-text')
            ])
        ], color='info', inverse=True), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Kelompok Usia', className='card-title'),
                html.H2(id='unique-age-groups', className='card-text')
            ])
        ], color='success', inverse=True), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5('Gender Terpilih', className='card-title'),
                html.H2(id='selected-gender', className='card-text')
            ])
        ], color='warning', inverse=True), md=3),
    ], className='mb-4'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='diagnosis-chart', figure=build_diagnosis_figure(df), style={'height': '450px'}), md=6),
        dbc.Col(dcc.Graph(id='symptom-chart', figure=build_symptom_figure(df), style={'height': '450px'}), md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='age-chart', figure=build_age_figure(df), style={'height': '450px'}), md=6),
        dbc.Col(dcc.Graph(id='sunburst-chart', figure=build_sunburst_figure(df), style={'height': '450px'}), md=6),
    ])
], fluid=True)


@app.callback(
    [Output('total-patients', 'children'),
     Output('unique-diagnosis', 'children'),
     Output('unique-age-groups', 'children'),
     Output('selected-gender', 'children'),
     Output('diagnosis-chart', 'figure'),
     Output('symptom-chart', 'figure'),
     Output('age-chart', 'figure'),
     Output('sunburst-chart', 'figure')],
    [Input('diagnosis-dropdown', 'value'),
     Input('gender-dropdown', 'value'),
     Input('age-group-dropdown', 'value')]
)
def update_dashboard(diagnosis_value, gender_value, age_group_value):
    filtered_df = df.copy()
    if diagnosis_value != 'All':
        filtered_df = filtered_df[filtered_df['diagnosis'] == diagnosis_value]
    if gender_value != 'All':
        filtered_df = filtered_df[filtered_df['gender'] == gender_value]
    if age_group_value != 'All':
        filtered_df = filtered_df[filtered_df['age_group'] == age_group_value]

    total_patients = len(filtered_df)
    if filtered_df.empty:
        unique_diagnosis = 'Tidak ada data'
    elif diagnosis_value == 'All':
        unique_diagnosis = 'All'
    else:
        unique_diagnosis = diagnosis_value
    if filtered_df.empty:
        unique_age_groups = 'Tidak ada data'
    elif age_group_value == 'All':
        unique_age_groups = 'All'
    else:
        age_groups = filtered_df['age_group'].dropna().astype(str).unique().tolist()
        unique_age_groups = ', '.join(age_groups) if age_groups else 'Tidak ada data'
    selected_gender = gender_value if gender_value != 'All' else 'All'

    if filtered_df.empty:
        diagnosis_fig = px.bar(title='Tidak ada data untuk filter terpilih', template='plotly_white')
        symptom_fig = px.bar(title='Tidak ada data untuk filter terpilih', template='plotly_white')
        age_fig = px.box(title='Tidak ada data untuk filter terpilih', template='plotly_white')
        sunburst_fig = px.sunburst(title='Tidak ada data untuk filter terpilih', template='plotly_white')
    else:
        diagnosis_fig = build_diagnosis_figure(filtered_df)
        symptom_fig = build_symptom_figure(filtered_df)
        age_fig = build_age_figure(filtered_df)
        sunburst_fig = build_sunburst_figure(filtered_df)

    return (
        f'{total_patients}',
        f'{unique_diagnosis}',
        f'{unique_age_groups}',
        selected_gender,
        diagnosis_fig,
        symptom_fig,
        age_fig,
        sunburst_fig
    )


if __name__ == '__main__':
    app.run(debug=True, port=8050)