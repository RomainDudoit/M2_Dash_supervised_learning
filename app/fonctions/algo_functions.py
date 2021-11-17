import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Row import Row
from pkg_resources import NullProvider
import plotly.express as px
import pandas as pd
from dash.dependencies import Input,Output,State
import os
import pandas as pd
import json
from dash.exceptions import PreventUpdate
from sklearn.tree import DecisionTreeClassifier
from dash import dash_table
import numpy as np
import base64
import io
import cchardet as chardet
from detect_delimiter import detect
from sklearn.cluster import KMeans
import dash_daq as daq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, mean_squared_error, r2_score
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline

import time 


def get_best_params(X,Y,clf,params,cv,scoring,njobs):
    if clf == "KNeighborsClassifier":
        if scoring not in ["f1_binary","recall_binary","precision_binary","recall_micro","recall_macro","recall_weighted","precision_micro","precision_macro","precision_weighted"]:
            grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=scoring, cv=cv, n_jobs=njobs)
        else:
            if scoring == "f1_binary":
                if len(set(list(Y))) > 2:
                    grid_search = "le nombre de classe n'est pas binaire, veuillez selectionner une métrique dont l'average est différent de 'binary'"
                    return grid_search
                else:
                    grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(f1_score,average="binary", pos_label = sorted(list(set(list(Y))))[0]))
            if scoring == "recall_binary":
                if len(set(list(Y))) > 2:
                    grid_search = "le nombre de classe n'est pas binaire, veuillez selectionner une métrique dont l'average est différent de 'binary'"
                    return grid_search
                else:
                    grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(recall_score,average="binary", pos_label = sorted(list(set(list(Y))))[0]))
            if scoring == "recall_micro":
                grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(recall_score,average="micro"), cv=cv, n_jobs=njobs)
            if scoring == "recall_macro":
                grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(recall_score,average="macro"), cv=cv, n_jobs=njobs)
            if scoring == "recall_weighted":
                grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(recall_score,average="weighted"), cv=cv, n_jobs=njobs)
            if scoring == "precision_binary":
                if len(set(list(Y))) > 2:
                    grid_search = "le nombre de classe n'est pas binaire, veuillez selectionner une métrique dont l'average est différent de 'binary'"
                    return grid_search
                else:
                    grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(precision_score,average="binary", pos_label = sorted(list(set(list(Y))))[0]))
            if scoring == "precision_micro":
                grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(precision_score,average="micro"), cv=cv, n_jobs=njobs)
            if scoring == "precision_macro":
                grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(precision_score,average="macro"), cv=cv, n_jobs=njobs)
            if scoring == "precision_weighted":
                grid_search = GridSearchCV(KNeighborsClassifier(), params, scoring=make_scorer(precision_score,average="weighted"), cv=cv, n_jobs=njobs)
        grid_search = grid_search.fit(X.values,Y.values)
        return grid_search

    if clf == "KNeighborsRegressor":
        if scoring == "MAE":
            grid_search = GridSearchCV(KNeighborsRegressor(), params, scoring="neg_mean_absolute_error", cv=cv, n_jobs=njobs)
        else:
            grid_search = GridSearchCV(KNeighborsRegressor(), params, scoring="neg_mean_squared_error", cv=cv, n_jobs=njobs)
        grid_search = grid_search.fit(X.values,Y.values)
        return grid_search
    if clf == "Arbre de decision" : 
        t1 = time.time()
        grid_search = GridSearchCV(DecisionTreeClassifier(), params, scoring=scoring, cv=cv, n_jobs=njobs)
        grid_search = grid_search.fit(X,Y)
        t2 = time.time()
        diff = t2 - t1
        return [grid_search,diff]
    if clf == "Regression lineaire" : 
        t1 = time.time()
        grid_search = GridSearchCV(LinearRegression(), params, scoring=scoring, cv=cv, n_jobs=njobs)
        grid_search = grid_search.fit(X,Y)
        t2 = time.time()
        diff = t2 - t1
        return [grid_search,diff]

def binariser(df,features,target):
    df_ = pd.get_dummies(df[features])
    f = df_.columns
    df = pd.concat([df_,df[target]], axis=1)
    return [df,f]

def centrer_reduire_norm(df,features):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    X = pd.DataFrame(X,columns=features)
    return X

def build_smv(kernel,regularisation,epsilon):
    
    numerical_features = make_column_selector(dtype_include=np.number)
    categorical_features = make_column_selector(dtype_exclude=np.number)

    categorical_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(drop='first',sparse=False))
    numerical_pipeline = make_pipeline(SimpleImputer(),StandardScaler())

    preprocessor = make_column_transformer((numerical_pipeline,numerical_features),
                                            (categorical_pipeline,categorical_features))

    model = make_pipeline(preprocessor,SVR(kernel=kernel,C=regularisation,epsilon=epsilon))

    #print(sorted(model.get_params().keys()))
    #print(sorted(model.metrics.SCORERS.keys()))
    return model
    
def build_KNeighborsClassifier(n_neighbors,weights,algorithm,leaf_size,p,metric):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size,p=p,metric=metric)
    return clf

def build_KNeighborsRegressor(n_neighbors,weights,algorithm,leaf_size,p,metric):
    clf = KNeighborsRegressor(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size,p=p,metric=metric)
    return clf



def build_tree(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes):
    tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)

    return tree


def buid_linearReg(fit_intercept,copy_X,n_jobs):
    linearReg =  LinearRegression(fit_intercept=fit_intercept,copy_X=copy_X,n_jobs=n_jobs)
    return linearReg



def cross_val(clf,X,Y,cv,scoring):
    t1 = time.time()
    cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=scoring)
    if scoring == "recall_micro":
        cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="micro"))
    if scoring == "recall_macro":
        cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="macro"))
    if scoring == "recall_weighted":
        cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="weighted"))
    if scoring == "precision_micro":
        cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="micro"))
    if scoring == "precision_macro":
        cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="macro"))
    if scoring == "precision_weighted":
        cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="weighted"))
    t2 = time.time()
    diff = t2 - t1
    return [cross_val,diff]

def cross_validation(clf,X,Y,cv,scoring):
    if str(clf).startswith("DecisionTreeClassifier"):
        if scoring not in ["f1_binary","recall_binary","precision_binary","recall_micro","recall_macro","recall_weighted","precision_micro","precision_macro","precision_weighted"]:
            cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=scoring)
        else:
            if scoring == "f1_binary":
                if len(set(list(Y))) > 2:
                    cross_val = "le nombre de classe n'est pas binaire, veuillez selectionner une métrique dont l'average est différent de 'binary'"
                    return cross_val
                else:
                    cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(f1_score,average="binary", pos_label = sorted(list(set(list(Y))))[0]))
            if scoring == "recall_binary":
                if len(set(list(Y))) > 2:
                    cross_val = "le nombre de classe n'est pas binaire, veuillez selectionner une métrique dont l'average est différent de 'binary'"
                    return cross_val
                else:
                    cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="binary", pos_label = sorted(list(set(list(Y))))[0]))
            if scoring == "recall_micro":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="micro"))
            if scoring == "recall_macro":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="macro"))
            if scoring == "recall_weighted":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="weighted"))
            if scoring == "precision_binary":
                if len(set(list(Y))) > 2:
                    cross_val = "le nombre de classe n'est pas binaire, veuillez selectionner une métrique dont l'average est différent de 'binary'"
                    return cross_val
                else:
                    cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="binary", pos_label = sorted(list(set(list(Y))))[0]))
            if scoring == "precision_micro":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="micro"))
            if scoring == "precision_macro":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="macro"))
            if scoring == "precision_weighted":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="weighted"))
        return cross_val
    if str(clf).startswith("KNeighborsClassifier"):
        if scoring not in ["f1_binary","recall_binary","precision_binary","recall_micro","recall_macro","recall_weighted","precision_micro","precision_macro","precision_weighted"]:
            cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=scoring)
        else:
            if scoring == "f1_binary":
                if len(set(list(Y))) > 2:
                    cross_val = "le nombre de classe n'est pas binaire, veuillez selectionner une métrique dont l'average est différent de 'binary'"
                    return cross_val
                else:
                    cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(f1_score,average="binary", pos_label = sorted(list(set(list(Y))))[0]))
            if scoring == "recall_binary":
                if len(set(list(Y))) > 2:
                    cross_val = "le nombre de classe n'est pas binaire, veuillez selectionner une métrique dont l'average est différent de 'binary'"
                    return cross_val
                else:
                    cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="binary", pos_label = sorted(list(set(list(Y))))[0]))
            if scoring == "recall_micro":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="micro"))
            if scoring == "recall_macro":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="macro"))
            if scoring == "recall_weighted":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(recall_score,average="weighted"))
            if scoring == "precision_binary":
                if len(set(list(Y))) > 2:
                    cross_val = "le nombre de classe n'est pas binaire, veuillez selectionner une métrique dont l'average est différent de 'binary'"
                    return cross_val
                else:
                    cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="binary", pos_label = sorted(list(set(list(Y))))[0]))
            if scoring == "precision_micro":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="micro"))
            if scoring == "precision_macro":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="macro"))
            if scoring == "precision_weighted":
                cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=make_scorer(precision_score,average="weighted"))
        return cross_val

    if str(clf).startswith("KNeighborsRegressor"):
        if scoring == "MAE":
            cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring="neg_mean_absolute_error")
        else:
            cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring="neg_mean_squared_error")

        return cross_val
    if str(clf).startswith("LinearRegression"):
        cross_val = cross_val_score(estimator=clf,X=X.values,y=Y.values,cv=cv,scoring=scoring) 
        return cross_val
