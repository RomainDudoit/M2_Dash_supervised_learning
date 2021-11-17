from tkinter.constants import NONE
from dash import dcc
from dash import html
from dash.development.base_component import Component
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Collapse import Collapse
from dash_bootstrap_components._components.Row import Row
from numpy.core.numeric import cross
from numpy.random.mtrand import random_integers
import plotly.express as px
import pandas as pd
from dash.dependencies import Input,Output,State
import os
import pandas as pd
import json
from dash.exceptions import PreventUpdate
from dash import dash_table
import numpy as np
import plotly.graph_objects as go
import time
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import validation_curve, GridSearchCV

from fonctions.various_functions import get_pandas_dataframe
from fonctions.algo_functions import build_smv
from sklearn.svm import SVR

from layout.layout import location_folder, dataset_selection, target_selection,features_selection
from layout.layout import regression_tabs, classification_tabs

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, mean_squared_error, roc_curve, r2_score
from math import sqrt
from matplotlib import pyplot

# (Régression) SVM

def Gridsearch(app):
    @app.callback(
        #Output(component_id='svr-ls-loading-output-1',component_property='children'),
        Output(component_id='res_svr_GridSearchCV',component_property='children'),   # Affichage des meilleurs paramètres 
        Input(component_id='svr_button_GridSearchCV',component_property='n_clicks'), # Validation du Gridsearch
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='train_size',component_property='value'),
        State(component_id='svr_gridCV_k_folds',component_property='value'),
        State(component_id='svr_GridSearchCV_njobs',component_property='value'),
        State(component_id='svr_gridCV_scoring',component_property='value'))

    def GridsearchSVM (n_clicks,file,target,features,train_size,k_fold,n_jobs,metric):
        
        if (n_clicks==0):
            PreventUpdate
        else:
            t1 = time.time()
            df = get_pandas_dataframe(file)
            X= df[features]
            y= df[target]
            
            X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=train_size)

            numerical_features = make_column_selector(dtype_include=np.number)
            categorical_features = make_column_selector(dtype_exclude=np.number)

            categorical_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(drop='first',sparse=False))
            numerical_pipeline = make_pipeline(SimpleImputer(),StandardScaler())

            preprocessor = make_column_transformer((numerical_pipeline,numerical_features),
                                                (categorical_pipeline,categorical_features))


            clf = SVR()
            model = Pipeline([('preprocessor',preprocessor),('clf',clf)])
            params = {
                'clf__kernel':['linear','poly','rbf','sigmoid'],
                'clf__degree': [i for i in range(1,6)],
                'clf__gamma': ['scale','auto'],
                'clf__coef0': [i for i in np.arange(0.1,1,0.2)],
                'clf__C' : [i for i in np.arange(0.1,1,0.2)],
                'clf__epsilon' : [i for i in np.arange(0.1,1,0.2)]
            }
            grid = GridSearchCV(model,params,scoring=metric,cv=k_fold,n_jobs=n_jobs)
            grid.fit(X_train,y_train)

            t2 = time.time()
            diff = t2 - t1

            y_pred = grid.predict(X_test)

            return (
                [
                    html.P("Paramètres optimaux : {}".format(grid.best_params_)),
                    html.P("Meilleur score : {}".format(grid.best_score_))
                ]
            )

def fit_predict(app):
    # Fit et predict 
    @app.callback(
        Output('res_svm','children'),
        Input('smv_button','n_clicks'),
        State(component_id='file_selection',component_property='value'),
        State(component_id='target_selection',component_property='value'),
        State(component_id='features_selection',component_property='value'),
        State(component_id='test_size',component_property='value'),
        State(component_id='random_state',component_property='value'),
        State(component_id='k_fold',component_property='value'),
        State(component_id='svm_kernel_selection',component_property='value'),          # Noyau
        State(component_id='svm_regularisation_selection',component_property='value'),  # C
        State(component_id='svm_epsilon',component_property='value'),
        State('svm_degre','value'))

    def svm (n_clicks,file,target,features,test_size,random_state,k_fold,kernel,regularisation,epsilon,degre):

        if (n_clicks == 0):
            PreventUpdate
        else:
            df = get_pandas_dataframe(file)

            X= df[features]
            y= df[target]

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)

            model = build_smv(kernel,regularisation,epsilon)
            score = cross_validate(model,X_train,y_train,cv=k_fold,scoring=('r2','neg_mean_squared_error'),return_train_score=True)

            rsquared = score['test_r2'].mean()
            mse = score['test_neg_mean_squared_error'].mean()

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            fig = px.imshow(df.corr())
            
            #fig = px.scatter_matrix(df,dimensions=features)

            # train_score, val_score = validation_curve(model,X_train,y_train,param_name='svr__C',param_range=np.arange(0,100),cv=k_fold)
            
            # fig = go.Figure()

            # fig.add_trace(go.Scatter(x=np.arange(0,100), y=val_score.mean(axis=1),mode='lines',name='validation score'))
            # fig.add_trace(go.Scatter(x=np.arange(0,100), y=train_score.mean(axis=1),mode='lines',name='training score'))
            # fig.update_layout(title="Score en fonction de C")

            return [    
                        html.Div(
                            [
                                dbc.Label("Validation score"),
                                html.P('R² : '+str(rsquared)),
                                html.P('MSE : '+str(mse))
                            ]
                        ),
                        dcc.Graph(figure=fig)
                    ]

