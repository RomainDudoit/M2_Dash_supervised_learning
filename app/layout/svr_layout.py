import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Row import Row
from numpy.core.fromnumeric import size
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
import dash_daq as daq
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split


regression_svm = dbc.Card(          
    children=[
        html.H2(html.B(html.P("Support Vector Regressor", className="card-text"))),html.Br(),

        html.Div(
            [
                html.Div(
                    children = 
                    [
                        html.H3(html.B("Settings")),html.Hr(),
                        html.H4(html.B("Optimisation des hyperparamètres :")),html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Label("Taille de l'échantillon d'entrainement", html_for="train_size",style={'font-weight': 'bold'}),
                                    width=5
                                ),
                                dbc.Col(
                                    dcc.Slider(
                                        id='train_size',min=0.0,max=1.0,step=0.1,value=0.7,tooltip={"placement": "bottom", "always_visible": True}
                                        #className="col-sm-6 col-md-5 col-lg-4",# Taille de la slider sur 3 colonnes 
                                    ),width=5
                                )
                            ]
                        ),
                        html.B("GridSearchCV_number_of_folds "),html.I("par défaut=10"),html.Br(),

                        html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée pour l'optimisation des hyperparamètres.", className="card-text"),
                        dcc.Input(id="svr_gridCV_k_folds", type="number", placeholder="input with range",min=1,max=100, step=1,value=5),html.Br(),html.Br(),
                    
                        html.B("GridSearchCV_scoring "),html.I("par défaut = 'MSE'"),html.Br(),
                        html.P("Selectionner la méthode de scoring pour l'optimisation des hyperparamètres."),
                        dcc.Dropdown(
                            id='svr_gridCV_scoring',
                            options=[
                                {'label': "MSE", 'value': "neg_mean_squared_error"},
                                {'label': "R2", 'value': "r2"},
                            ],
                            value = 'neg_mean_squared_error'
                        ),html.Br(),html.Br(),

                        html.B("GridSearchCV_njobs "),html.I("par défaut=-1"),html.Br(),
                        html.P("Selectionner le nombre de coeurs (-1 = tous les coeurs)", className="card-text"),
                        dcc.Dropdown(
                            id="svr_GridSearchCV_njobs",
                            options= [{'label': 'None', 'value': 'None'}] + [{'label': -1, 'value': -1}] + [{'label':i, 'value':i} for i in range(1,33)],
                            value = -1
                        ),html.Br(),html.Br(),

                        dbc.Button("valider GridSearchCV",color ="info",id='svr_button_GridSearchCV',n_clicks=0),
                        
                        html.Br(),html.Hr(),
                        
                        html.H4(html.B("Paramètrage du modèle et Fit & Predict :")),html.Br(),

                
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Label("Taille de l'échantillon de test", html_for="test_size",style={'font-weight': 'bold'}),
                                    width=5
                                ),
                                dbc.Col(
                                    dcc.Slider(
                                        id='test_size',min=0.0,max=1.0,step=0.1,value=0.3,tooltip={"placement": "bottom", "always_visible": True}
                                        #className="col-sm-6 col-md-5 col-lg-4",# Taille de la slider sur 3 colonnes 
                                    ),width=5
                                )
                            ]

                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Random seed", html_for="random_state",style={'font-weight': 'bold'}),
                                        dbc.Input(id='random_state',type='number'),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("K-folds ", html_for="k_fold",style={'font-weight': 'bold'}),
                                        dbc.Input(id='k_fold',value=5,type='number'),
                                    ]
                                )
                            ],
                        ),
                        html.Br(),html.Br(),

                        # Paramètres de l'algo
                        dbc.Row(
                            [
                                # Type du noyau
                                dbc.Col(
                                    [
                                        dbc.Label("Type de noyau (kernel)", html_for="svm_kernel_selection",style={'font-weight': 'bold'}),
                                        dcc.Dropdown(
                                            id='svm_kernel_selection',
                                            options=[
                                                {'label': 'linéaire', 'value': 'linear'},
                                                {'label': 'polynomial', 'value': 'poly'},
                                                {'label': 'RBF', 'value': 'rbf'},
                                                {'label': 'Sigmoïde', 'value': 'sigmoid'},
                                            ],
                                            value = 'rbf'
                                        ),
                                    ],
                                ),

                                # Degré pour noyau polynomial
                                dbc.Col(
                                    [
                                        dbc.Label("Degré (pour noyau polynomial)", html_for="svm_kernel_selection",style={'font-weight': 'bold'}),
                                        dbc.Input(id='svm_degre',type='number',min=0,max=4,step=1,value=0,),
                                    ],
                                )
                            ]
                        ),

                        html.Br(),
                        dbc.Row(
                            [
                                # Paramètre de régularisation
                                dbc.Col(
                                    [
                                        dbc.Label("Régularisation (C)", html_for="svm_regularisation_selection",style={'font-weight': 'bold'}),
                                        dbc.Input(id='svm_regularisation_selection',type='number',min=0,max=100,step=0.1,value=0.1,),
                                    ],
                                ),
                            ],style={'margin-bottom': '1em'}
                        ),

                        dbc.Row(
                            [
                                # Epsilon 
                                dbc.Col(
                                    [
                                        dbc.Label("Epsilon (ε)",html_for='svm_epsilon',style={'font-weight': 'bold'}),
                                        dbc.Input(id='svm_epsilon',type='number',value=0.1,min=0,max=100,step=0.1),
                                    ],
                                )
                            ]
                        )
                    

                    ],className='col-6'
                ),
                html.Div(
                    [
                        html.H3(html.B("Résultats :")),html.Hr(),
                        dcc.Loading(
                            id="svr-ls-loading-1", 
                            children=[html.Div(id="res_svr_GridSearchCV")], 
                            type="default"
                        ),
                        #html.Div(id="res_svr_GridSearchCV"),html.Br(),html.Hr(),
                        html.Div(id="res_KNeighborsRegressor_FitPredict"),html.Br(),html.Hr(),
                        html.Div(id="res_KNeighborsRegressor_CrossValidation")
                    ],
                    className='col-6'
                )
            ],className="row"
        ),

            html.Br(),html.Br(),
            
            html.Br(),html.Br(),
            
            dbc.Button("Valider fit & predict", color="danger",id='smv_button',n_clicks=0),
            html.Div(id='res_svm'),
            html.Div(id='test')
        ],
    body=True
)