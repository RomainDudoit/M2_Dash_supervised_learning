import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Row import Row
import plotly.express as px
import pandas as pd
from dash.dependencies import Input,Output,State
import os
import pandas as pd
import json
from dash.exceptions import PreventUpdate
from dash import dash_table
import numpy as np
import base64
import io
import cchardet as chardet
from detect_delimiter import detect
from sklearn.cluster import KMeans
import dash_daq as daq
from sklearn.preprocessing import StandardScaler

# Fonction qui permet de filtrer les fichiers qui peuvent être sélectionné suivant une liste d'extensions prédéfinie.
def allowed_files(path,extensions):
    allowed_files=[]
    for file in os.listdir(path):
        if file.endswith(extensions):
            allowed_files.append(file)
        else:
            continue
    return allowed_files

def number_of_observations(file_path):
    with open(file_path) as fp:
        count = 0
        for _ in fp:
            count += 1
    return count

# Fonction qui permet de lire un fichier csv ou xls et qui retoune un pandas dataframe
def get_pandas_dataframe(file_path):
    if file_path.endswith('.csv'):
        with open(r'%s' %file_path, "rb") as f:
            msg = f.read()
            firstline = f.readline()
            detection = chardet.detect(msg)
            encoding= detection["encoding"]
        f.close()

        with open(r'%s' %file_path) as f:
            delimiter = detect(f.readline())
        f.close()

        df = pd.read_csv(file_path,encoding=encoding,sep=delimiter)

    elif file_path.endswith(('.xls','.xlsx')):
        df = pd.read_excel(file_path)

    return df

# (à supprimer ??) Fonction qui permet de lire un fichier csv ou xls et qui retoune un datatable
def parse_contents(contents, filename):
    with open(r'%s' %filename) as f:
        delimiter = detect(f.readline())
    f.close()
    content_type, content_string = contents.split(delimiter)
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'Il y a eu une erreur dans le format du fichier.'
        ])

    return dbc.Col(
        html.Div([
        html.H5(filename),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            fixed_rows={'headers': True},
            page_size=20,
            style_cell={'textAlign': 'left','minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
            style_table={'height': '400px', 'overflowY': 'scroll','overflowX': 'scroll'},
            style_header={'backgroundColor': 'dark','fontWeight': 'bold'}
        ),
        html.Hr(),  # horizontal line
    ],className='container-fluid'),
    width=10
    )
