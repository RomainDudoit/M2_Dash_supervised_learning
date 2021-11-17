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


from layout.svr_layout import regression_svm
from layout.log_layout import classification_log




# --- /!\ data_path = os.getcwd() +'\data\\' # WINDOWS
data_path = os.getcwd() +'/data/' # LINUX - MAC-OS
files = [f for f in os.listdir(r'%s' %data_path)]

########################################################################################################
# (INIT) Input pour définir le répertoire de travail
location_folder = dbc.Row(
    [
        dbc.Col(
            dbc.Input(
                    type="text", id="location_folder", placeholder="Chemin absolu du répertoire du répertoire de travail : C:\..",persistence=True
                ),className="mb-3"
        ),
        dbc.Col(
            dbc.Button(
                "Valider", id="validation_folder", className="me-2", n_clicks=1
            ),className="mb-3"
        )
    ]
)

########################################################################################################
# (INIT) Dropdown pour sélectionner le fichier à analyser selon le répertoire de travail choisit
dataset_selection = dbc.Row(
    [
        dbc.Label("Jeu de données sélectionné", html_for="file_selection", width=1,style={'font-weight': 'bold'}),
        dbc.Col(
            dcc.Dropdown(
                id='file_selection',
                #options=[{'label':i, 'value':i} for i in files],
                searchable=False,
                placeholder="Choisir un jeu de données",
                clearable=False,
                style={'width':'50%'},
                persistence = True
            ),
            width=10,
        ),
    ],
    className="mb-3",
)

########################################################################################################
# (INIT) Dropdown pour sélectionner la variable cible
target_selection = dbc.Row(
    [
        dbc.Label("Variable cible", html_for="target_selection", width=1,style={'color': 'red','font-weight': 'bold'}),
        dbc.Col(
            dcc.Dropdown(
                id='target_selection',
                placeholder="Sélectionner la variable cible",
                searchable=False,clearable=False,
                style={'width':'50%'},
                persistence=True,
                #persistence_type='memory'
            ),
            width=10,
        ),
    ],
    className="mb-3",
)

########################################################################################################
# (INIT) Dropdown pour sélection des variables explicatives
features_selection = dbc.Row(
    [
        dbc.Label("Variables explicatives", html_for="features_selection", width=1,style={'color': 'blue','font-weight': 'bold'}),
        dbc.Col(
            dcc.Dropdown(
                    id='features_selection',
                    searchable=False,
                    placeholder="Sélectionner les variables explicatives",
                    clearable=False,
                    multi=True,
                    style={'width':'50%'},
                    persistence=True,
                    #persistence_type='memory'
            ),
            width=10,
        ),
    ],
    className="mb-3",
)

########################################################################################################
# (Classification) Onglets

# Arbre, KNN, régression logistique

# Arbre de décision
classification_decision_tree = dbc.Card(
    children=[

            html.H2(html.B(html.P("Decision Tree", className="card-text"))),html.Br(),html.Hr(),html.Br(),

            html.Div([

                html.Div([

                    html.H3(html.B("Paramètrage")),html.Br(),html.Hr(),
                    html.H4(html.B("Optimisation des hyperparamètres :")),html.Br(),html.Br(),
                    html.B("GridSearchCV_number_of_folds "),
                    html.I("par défaut=10"),html.Br(),
                    html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée pour l'optimisation des hyperparamètres.", className="card-text"),
                    dcc.Input(
                        id="Tree_GridSearchCV_number_of_folds",
                        type="number",
                        placeholder="input with range",
                        min=1,max=100, step=1,value=10),
                    html.Br(),html.Br(),
                    html.B("GridSearchCV_scoring "),
                    html.I("par défaut = 'f1_macro'"),html.Br(),
                    html.P("Selectionnez la méthode de scoring pour l'optimisation des hyperparamètres."),
                    dcc.Dropdown(
                        id='Tree_GridSearchCV_scoring',
                        options=[
                            {'label': "accuracy", 'value': "accuracy"},
                            {'label': "balanced_accuracy", 'value': "balanced_accuracy"},
                            {'label': "neg_brier_score", 'value': "neg_brier_score"},
                            {'label': "f1_micro", 'value': "f1_micro"},
                            {'label': "f1_macro", 'value': "f1_macro"},
                            {'label': "f1_weighted", 'value': "f1_weighted"},
                            #{'label': "neg_log_loss", 'value': "neg_log_loss"},
                            {'label': "precision_micro", 'value': "precision_micro"},
                            {'label': "precision_macro", 'value': "precision_macro"},
                            {'label': "precision_weighted", 'value': "precision_weighted"},
                            {'label': "recall_micro", 'value': "recall_micro"},
                            {'label': "recall_macro", 'value': "recall_macro"},
                            {'label': "recall_weighted", 'value': "recall_weighted"},
                            {'label': "roc_auc_ovr", 'value': "roc_auc_ovr"},
                            {'label': "roc_auc_ovo", 'value': "roc_auc_ovo"},
                            {'label': "roc_auc_ovr_weighted", 'value':"roc_auc_ovr_weighted" },
                            {'label': "roc_auc_ovo_weighted", 'value': "roc_auc_ovo_weighted"}
                        ],
                        value = 'f1_macro'
                    ),html.Br(),html.Br(),
                    html.B("GridSearchCV_njobs "),html.I("par défaut=-1"),html.Br(),
                    html.P("Selectionner le nombre de coeurs (-1 = tous les coeurs)", className="card-text"),
                    dcc.Dropdown(
                        id="Tree_GridSearchCV_njobs",
                        options=[
                            {'label': -1, 'value': -1},{'label': 1, 'value': 1},{'label': 2, 'value': 2},{'label': 3, 'value': 3},{'label': 4, 'value': 4},
                            {'label': 5, 'value': 5},{'label': 6, 'value': 6},{'label': 7, 'value': 7},{'label': 8, 'value': 8},{'label': 9, 'value': 9},
                            {'label': 10, 'value': 10},{'label': 11, 'value': 11},{'label': 12, 'value': 12},{'label': 13, 'value': 13},{'label': 14, 'value': 14},
                            {'label': 15, 'value': 15},{'label': 16, 'value': 16},{'label': 17, 'value': 17},{'label': 18, 'value': 18},{'label': 19, 'value': 19},
                            {'label': 20, 'value': 20},{'label': 21, 'value': 21},{'label': 22, 'value': 22},{'label': 23, 'value': 23},{'label': 24, 'value': 24},
                            {'label': 25, 'value': 25},{'label': 26, 'value': 26},{'label': 27, 'value': 27},{'label': 28, 'value': 28},{'label': 29, 'value': 29},
                            {'label': 30, 'value': 30},{'label': 31, 'value': 31},{'label': 32, 'value': 32},{'label': 'None', 'value': 'None'}
                        ],
                        value = -1
                    ),html.Br(),html.Br(),
                    dbc.Button("valider GridSearchCV", color="info",id='Tree_button_GridSearchCV',n_clicks=0),
                    dcc.Loading(id="ls-loading-0_tree", children=[html.Div(id="ls-loading-output-0_tree")], type="default"),html.Br(),html.Hr(),html.Br(),
                    # on passe à une autre section (modification des paramètre) >> propre à la fonction
                    html.H4(html.B("Paramètres : ")),
                    html.Hr(),html.Br(),
                    html.B("criterion : ")," ",
                    dcc.Dropdown(
                        id='criterion',
                        options=[
                            {'label': 'gini', 'value': 'gini'},
                            {'label': 'entropy', 'value': 'entropy'}
                        ],
                        value='gini'
                    ),
                    html.Br(),
                    html.B("splitter : "),"",
                    dcc.Dropdown(
                        id='splitter',
                        options=[
                            {'label':'best','value':'best'},
                            {'label':'random','value':'random'}],
                        value="best",
                    ),
                    html.Br(),
                    html.B("max_depth : "),"",
                    dcc.Input(
                        id='max_depth',
                        type='number',
                        min=0,
                        max=10,
                        value=0,
                    ),
                    html.Br(),
                    html.B("min_samples_split : "),"",
                    html.Br(),
                    dcc.Input(
                        id='min_samples_split',
                        type="number",
                        value=2,
                        max=10,
                        min=1
                    ),
                    html.Br(),
                    html.B("min_samples_leaf : "),"",
                    dcc.Input(
                        id='min_samples_leaf',
                        type='number',
                        min=1,
                        max=10,
                        value=1
                    ),
                    html.Br(),
                    html.B("max_leaf_nodes : "),"",
                    dcc.Input(
                        id='max_leaf_nodes',
                        type='number',
                        max=10,
                        min=0,
                        value=0
                    ),
                    html.Br(),html.Br(),
                    dbc.Button("Valider Fit & Predict", color="danger",id='Tree_button_FitPredict',n_clicks=0),
                    dcc.Loading(id="ls-loading-1_tree", children=[html.Div(id="ls-loading-output-1_tree")], type="default"),html.Br(),html.Hr(),html.Br(),
                    # on passe à une autre section (validation croissé)
                    html.H4(html.B("validation croisée :")),html.Br(),html.Br(),
                    html.B("cv_number_of_folds "),
                    html.I("par défaut=10"),html.Br(),
                    html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée.", className="card-text"),
                    dcc.Input(
                        id="Tree_cv_number_of_folds", type="number", 
                        placeholder="input with range",min=1,max=100, 
                        step=1,value=10),
                    html.Br(),html.Br(),
                    html.B("cv_scoring "),html.I("par défaut = 'f1_macro'"),html.Br(),
                    html.P("Selectionnez la méthode de scoring pour la validation croisée."),
                    dcc.Dropdown(
                        id='Tree_cv_scoring',
                        options=[
                            {'label': "accuracy", 'value': "accuracy"},
                            {'label': "balanced_accuracy", 'value': "balanced_accuracy"},
                            #{'label': "neg_brier_score", 'value': "neg_brier_score"},
                            {'label': "f1_micro", 'value': "f1_micro"},
                            {'label': "f1_macro", 'value': "f1_macro"},
                            {'label': "f1_weighted", 'value': "f1_weighted"},
                            {'label': "neg_log_loss", 'value': "neg_log_loss"},
                            {'label': "precision_micro", 'value': "precision_micro"},
                            {'label': "precision_macro", 'value': "precision_macro"},
                            {'label': "precision_weighted", 'value': "precision_weighted"},
                            {'label': "recall_micro", 'value': "recall_micro"},
                            {'label': "recall_macro", 'value': "recall_macro"},
                            {'label': "recall_weighted", 'value': "recall_weighted"},
                            {'label': "roc_auc_ovr", 'value': "roc_auc_ovr"},
                            {'label': "roc_auc_ovo", 'value': "roc_auc_ovo"},
                            {'label': "roc_auc_ovr_weighted", 'value':"roc_auc_ovr_weighted" },
                            {'label': "roc_auc_ovo_weighted", 'value': "roc_auc_ovo_weighted"}
                        ],
                        value = 'f1_macro'
                    ),html.Br(),html.Hr(),
                    dbc.Button("Valider Decision Tree CrossValidation", color="success",id='Tree_button_CrossValidation',n_clicks=0),
                    dcc.Loading(id="ls-loading-2_tree", children=[html.Div(id="ls-loading-output-2_tree")], type="default")

                ],className="six columns",style={"margin":10}),

                html.Div([

                    html.H3(html.B("Résultats :")),
                    html.Div(id="res_Tree_GridSearchCV"),html.Br(),html.Hr(),html.Br(),
                    html.Div(id="res_Tree_FitPredict"),html.Br(),html.Hr(),html.Br(),
                    html.Div(id="res_Tree_CrossValidation"), html.Hr(),
                    html.H4("Affichage de l'arbre de décision : "),
                    dbc.Button("Affichage de l'arbre",color="success",id='tree_plot_button',n_clicks=0)


                ],className="six columns",style={"margin":10}),

            ], className="row"),
        ],body=True
)


# KNeighborsClassifier
classification_KNeighborsClassifier = dbc.Card(
    children=[

            html.H2(html.B(html.P("KNeighbors Classifier", className="card-text"))),html.Br(),html.Br(),
            html.B("centrer_reduire "),html.I("par défaut=non"),html.Br(),html.P("Si coché, on retranche à chaque donnée la moyenne de sa colonne d'appartenance et on la divise ensuite par l'écart-type de sa colonne d'appartenance."),
            dbc.Checklist(id="KNeighborsClassifier_centrer_reduire",options=[{"label":"centrer réduire","value":"yes"}]),html.Br(),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),html.Br(),

            html.Div([

                html.Div([

                    html.H3(html.B("Settings")),html.Br(),html.Hr(),html.Br(),
                    html.H4(html.B("Optimisation des hyperparamètres :")),html.Br(),html.Br(),

                    html.B("GridSearchCV_number_of_folds "),html.I("par défaut=10"),html.Br(),html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée pour l'optimisation des hyperparamètres.", className="card-text"),
                    dcc.Input(id="KNeighborsClassifier_GridSearchCV_number_of_folds", type="number", placeholder="input with range",min=1,max=100, step=1,value=10),html.Br(),html.Br(),
                    
                    html.B("GridSearchCV_scoring "),html.I("par défaut = 'f1_macro'"),html.Br(),html.P("Selectionnez la méthode de scoring pour l'optimisation des hyperparamètres."),
                    dcc.Dropdown(
                        id='KNeighborsClassifier_GridSearchCV_scoring',
                        options=[
                            {'label': "accuracy", 'value': "accuracy"},
                            {'label': "balanced_accuracy", 'value': "balanced_accuracy"},
                            {'label': "neg_brier_score", 'value': "neg_brier_score"},
                            {'label': "f1_binary", 'value': "f1_binary"},
                            {'label': "f1_micro", 'value': "f1_micro"},
                            {'label': "f1_macro", 'value': "f1_macro"},
                            {'label': "f1_weighted", 'value': "f1_weighted"},
                            #{'label': "neg_log_loss", 'value': "neg_log_loss"},
                            {'label': "precision_binary", 'value': "precision_binary"},
                            {'label': "precision_micro", 'value': "precision_micro"},
                            {'label': "precision_macro", 'value': "precision_macro"},
                            {'label': "precision_weighted", 'value': "precision_weighted"},
                            {'label': "recall_binary", 'value': "recall_binary"},
                            {'label': "recall_micro", 'value': "recall_micro"},
                            {'label': "recall_macro", 'value': "recall_macro"},
                            {'label': "recall_weighted", 'value': "recall_weighted"},
                            {'label': "roc_auc_ovr", 'value': "roc_auc_ovr"},
                            {'label': "roc_auc_ovo", 'value': "roc_auc_ovo"},
                            {'label': "roc_auc_ovr_weighted", 'value':"roc_auc_ovr_weighted" },
                            {'label': "roc_auc_ovo_weighted", 'value': "roc_auc_ovo_weighted"}
                        ],
                        value = 'f1_macro'
                    ),html.Br(),html.Br(),
                    html.B("GridSearchCV_njobs "),html.I("par défaut=-1"),html.Br(),html.P("Selectionner le nombre de coeurs (-1 = tous les coeurs)", className="card-text"),
                    dcc.Dropdown(
                        id="KNeighborsClassifier_GridSearchCV_njobs",
                        options=[
                            {'label': -1, 'value': -1},{'label': 1, 'value': 1},{'label': 2, 'value': 2},{'label': 3, 'value': 3},{'label': 4, 'value': 4},
                            {'label': 5, 'value': 5},{'label': 6, 'value': 6},{'label': 7, 'value': 7},{'label': 8, 'value': 8},{'label': 9, 'value': 9},
                            {'label': 10, 'value': 10},{'label': 11, 'value': 11},{'label': 12, 'value': 12},{'label': 13, 'value': 13},{'label': 14, 'value': 14},
                            {'label': 15, 'value': 15},{'label': 16, 'value': 16},{'label': 17, 'value': 17},{'label': 18, 'value': 18},{'label': 19, 'value': 19},
                            {'label': 20, 'value': 20},{'label': 21, 'value': 21},{'label': 22, 'value': 22},{'label': 23, 'value': 23},{'label': 24, 'value': 24},
                            {'label': 25, 'value': 25},{'label': 26, 'value': 26},{'label': 27, 'value': 27},{'label': 28, 'value': 28},{'label': 29, 'value': 29},
                            {'label': 30, 'value': 30},{'label': 31, 'value': 31},{'label': 32, 'value': 32},{'label': 'None', 'value': 'None'}
                        ],
                        value = -1
                    ),html.Br(),html.Br(),
                    dbc.Button("valider GridSearchCV",color ="info",id='KNeighborsClassifier_button_GridSearchCV',n_clicks=0),
                    dcc.Loading(id="KNeighborsClassifier-ls-loading-1", children=[html.Div(id="KNeighborsClassifier-ls-loading-output-1")], type="default"),html.Br(),html.Hr(),html.Br(),
                    html.H4(html.B("Paramètrage du modèle et Fit & Predict :")),html.Br(),html.Br(),
                    html.B("n_neighbors "),html.I("par défaut=5"),html.Br(),html.P("Nombre de voisins à utiliser par défaut pour les requêtes de voisins.", className="card-text"),
                    dcc.Input(id="KNeighborsClassifier_n_neighbors", type="number", placeholder="input with range",min=1,max=100, step=1,value=5),html.Br(),html.Br(),
                    html.B("weights "),html.I("par défaut = 'uniform'"),html.Br(),html.P("Fonction de poids utilisée dans la prédiction."),
                    dcc.Dropdown(
                        id='KNeighborsClassifier_weights',
                        options=[
                            {'label': 'uniform', 'value': 'uniform'},
                            {'label': 'distance', 'value': 'distance'},
                        ],
                        value = 'uniform'
                    ),html.Br(),html.Br(),
                    html.B("algorithm "),html.I("par défaut = 'auto'"),html.Br(),html.P("Algorithme utilisé pour calculer les voisins les plus proches."),
                    dcc.Dropdown(
                        id='KNeighborsClassifier_algorithm',
                        options=[
                            {'label': 'auto', 'value': 'auto'},
                            {'label': 'ball_tree', 'value': 'ball_tree'},
                            {'label': 'kd_tree', 'value': 'kd_tree'},
                            {'label': 'brute', 'value': 'brute'},
                        ],
                        value = 'auto'
                    ),html.Br(),html.Br(),
                    html.B("leaf_size "),html.I("par défaut=30"),html.Br(),html.P("Taille de la feuille transmise à BallTree ou KDTree. Cela peut affecter la vitesse de construction et de requête, ainsi que la mémoire requise pour stocker l'arbre. La valeur optimale dépend de la nature du problème.", className="card-text"),
                    dcc.Input(id="KNeighborsClassifier_leaf_size", type="number", placeholder="input with range",min=1,max=300, step=1,value=30),html.Br(),html.Br(),
                    html.B("p "),html.I("par défaut=2"),html.Br(),html.P("Paramètre de puissance pour la métrique de Minkowski. Lorsque p = 1, cela équivaut à utiliser manhattan_distance (l1) et euclidean_distance (l2) pour p = 2. Pour p arbitraire, minkowski_distance (l_p) est utilisé.", className="card-text"),
                    dcc.Input(id="KNeighborsClassifier_p", type="number", placeholder="input with range",min=1,max=100, step=1,value=2),html.Br(),html.Br(),
                    html.B("metric "),html.I("par défaut = 'minkowski'"),html.Br(),html.P("La métrique de distance à utiliser pour l'arbre."),
                    dcc.Dropdown(
                        id='KNeighborsClassifier_metric',
                        options=[
                            {'label': 'minkowski', 'value': 'minkowski'},
                            {'label': 'euclidean', 'value': 'euclidean'},
                            {'label': 'manhattan', 'value': 'manhattan'},
                            {'label': 'chebyshev', 'value': 'chebyshev'},
                            {'label': 'wminkowski', 'value': 'wminkowski'},
                            {'label': 'seuclidean', 'value': 'seuclidean'},
                            #{'label': 'mahalanobis', 'value': 'mahalanobis'},
                        ],
                        value = 'minkowski'
                    ),html.Br(),html.Br(),

                    html.B("test_size "),html.I("par défaut=0.3"),html.Br(),html.P("Taille du jeu de données test.", className="card-text"),
                    dcc.Input(id="KNeighborsClassifier_test_size", type="number", placeholder="input with range",min=0.1,max=0.5, step=0.1,value=0.3),html.Br(),html.Br(),

                    html.B("shuffle "),html.I("par défaut shuffle=True"),html.Br(),html.P("s'il faut ou non mélanger les données avant de les diviser.", className="card-text"),
                    dcc.Dropdown(
                        id='KNeighborsClassifier_shuffle',
                        options=[
                            {'label': 'True', 'value': 'True'},
                            {'label': 'False', 'value': 'False'},
                        ],
                        value = 'True'
                    ),html.Br(),html.Br(),

                    html.B("stratify "),html.I("par defaut stratify=False"),html.Br(),html.P("Si ce n'est pas False, les données sont divisées de manière stratifiée en utilisant les étiquettes de la classe à prédire", className="card-text"),
                    dcc.Dropdown(
                        id='KNeighborsClassifier_stratify',
                        options=[
                            {'label': 'True', 'value': 'True'},
                            {'label': 'False', 'value': 'False'},
                        ],
                        value = 'False'
                    ),html.Br(),html.Br(),

                    dbc.Button("Valider Fit & Predict", color="danger",id='KNeighborsClassifier_button_FitPredict',n_clicks=0),
                    dcc.Loading(id="KNeighborsClassifier-ls-loading-3", children=[html.Div(id="KNeighborsClassifier-ls-loading-output-3")], type="default"),html.Br(),html.Hr(),html.Br(),
                    html.H4(html.B("validation croisée :")),html.Br(),html.Br(),
                    html.B("cv_number_of_folds "),html.I("par défaut=10"),html.Br(),html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée.", className="card-text"),
                    dcc.Input(id="KNeighborsClassifier_cv_number_of_folds", type="number", placeholder="input with range",min=1,max=100, step=1,value=10),html.Br(),html.Br(),
                    html.B("cv_scoring "),html.I("par défaut = 'f1_macro'"),html.Br(),html.P("Selectionnez la méthode de scoring pour la validation croisée."),
                    dcc.Dropdown(
                        id='KNeighborsClassifier_cv_scoring',
                        options=[
                            {'label': "accuracy", 'value': "accuracy"},
                            {'label': "balanced_accuracy", 'value': "balanced_accuracy"},
                            #{'label': "neg_brier_score", 'value': "neg_brier_score"},
                            {'label': "f1_binary", 'value': "f1_binary"},
                            {'label': "f1_micro", 'value': "f1_micro"},
                            {'label': "f1_macro", 'value': "f1_macro"},
                            {'label': "f1_weighted", 'value': "f1_weighted"},
                            #{'label': "neg_log_loss", 'value': "neg_log_loss"},
                            {'label': "precision_binary", 'value': "precision_binary"},
                            {'label': "precision_micro", 'value': "precision_micro"},
                            {'label': "precision_macro", 'value': "precision_macro"},
                            {'label': "precision_weighted", 'value': "precision_weighted"},
                            {'label': "recall_binary", 'value': "recall_binary"},
                            {'label': "recall_micro", 'value': "recall_micro"},
                            {'label': "recall_macro", 'value': "recall_macro"},
                            {'label': "recall_weighted", 'value': "recall_weighted"},
                            {'label': "roc_auc_ovr", 'value': "roc_auc_ovr"},
                            {'label': "roc_auc_ovo", 'value': "roc_auc_ovo"},
                            {'label': "roc_auc_ovr_weighted", 'value':"roc_auc_ovr_weighted" },
                            {'label': "roc_auc_ovo_weighted", 'value': "roc_auc_ovo_weighted"}
                        ],
                        value = 'f1_macro'
                    ),html.Br(),html.Br(),
                    dbc.Button("Valider K-Fold Cross-Validation", color="success",id='KNeighborsClassifier_button_CrossValidation',n_clicks=0),
                    dcc.Loading(id="KNeighborsClassifier-ls-loading-2", children=[html.Div(id="KNeighborsClassifier-ls-loading-output-2")], type="default")

                ],className="six columns",style={"margin":10}),

                html.Div([

                    html.H3(html.B("Résultats :")),html.Br(),html.Hr(),html.Br(),
                    html.Div(id="res_KNeighborsClassifier_GridSearchCV"),html.Br(),html.Hr(),html.Br(),
                    html.Div(id="res_KNeighborsClassifier_FitPredict"),html.Br(),html.Hr(),html.Br(),
                    html.Div(id="res_KNeighborsClassifier_CrossValidation")


                ],className="six columns",style={"margin":10}),

            ], className="row"),
        ],body=True
)

# Ensemble des onglets de classification
classification_tabs = dbc.Tabs(
    id="classification_tab",
    children= [
        dbc.Tab(classification_log,label="Régression logistique",tab_id ='log',tab_style={'background-color':'#E4F2F2','border-color':'white'},label_style={'color':'black'}),
        dbc.Tab(classification_decision_tree,label="Arbre de décision",tab_id='decision_tree',tab_style={'background-color':'#E4F2F2','border-color':'white'},label_style={'color':'black'}),
        dbc.Tab(classification_KNeighborsClassifier,label="KNeighborsClassifier", tab_id='KNeighborsClassifier',tab_style={'background-color':'#E4F2F2','border-color':'white'},label_style={'color':'black'})
    ]
)

########################################################################################################
# (Régression) Onglets

# KNeighborsRegressor
regression_KNeighborsRegressor = dbc.Card(
    children=[

        html.H2(html.B(html.P("KNeighbors Regressor", className="card-text"))),html.Br(),html.Br(),
        html.B("centrer_reduire "),html.I("par défaut=non"),html.Br(),html.P("Si coché, on retranche à chaque donnée la moyenne de sa colonne d'appartenance et on la divise ensuite par l'écart-type de sa colonne d'appartenance."),
        dbc.Checklist(id="KNeighborsRegressor_centrer_reduire",options=[{"label":"centrer réduire","value":"yes"}]),html.Br(),html.Hr(style={'borderWidth': "0.5vh", "borderColor": "grey"}),html.Br(),

        html.Div([

            html.Div([

                html.H3(html.B("Settings")),html.Br(),html.Hr(),html.Br(),
                html.H4(html.B("Optimisation des hyperparamètres :")),html.Br(),html.Br(),

                html.B("GridSearchCV_number_of_folds "),html.I("par défaut=10"),html.Br(),html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée pour l'optimisation des hyperparamètres.", className="card-text"),
                dcc.Input(id="KNeighborsRegressor_GridSearchCV_number_of_folds", type="number", placeholder="input with range",min=1,max=100, step=1,value=10),html.Br(),html.Br(),
               
                html.B("GridSearchCV_scoring "),html.I("par défaut = 'MSE'"),html.Br(),html.P("Selectionnez la méthode de scoring pour l'optimisation des hyperparamètres."),
                dcc.Dropdown(
                    id='KNeighborsRegressor_GridSearchCV_scoring',
                    options=[
                        {'label': "MSE", 'value': "MSE"},
                        {'label': "RMSE", 'value': "RMSE"},
                        {'label': "MAE", 'value': "MAE"},
                    ],
                    value = 'MSE'
                ),html.Br(),html.Br(),
                
                html.B("GridSearchCV_njobs "),html.I("par défaut=-1"),html.Br(),html.P("Selectionner le nombre de coeurs (-1 = tous les coeurs)", className="card-text"),
                dcc.Dropdown(
                    id="KNeighborsRegressor_GridSearchCV_njobs",
                    options=[
                        {'label': -1, 'value': -1},{'label': 1, 'value': 1},{'label': 2, 'value': 2},{'label': 3, 'value': 3},{'label': 4, 'value': 4},
                        {'label': 5, 'value': 5},{'label': 6, 'value': 6},{'label': 7, 'value': 7},{'label': 8, 'value': 8},{'label': 9, 'value': 9},
                        {'label': 10, 'value': 10},{'label': 11, 'value': 11},{'label': 12, 'value': 12},{'label': 13, 'value': 13},{'label': 14, 'value': 14},
                        {'label': 15, 'value': 15},{'label': 16, 'value': 16},{'label': 17, 'value': 17},{'label': 18, 'value': 18},{'label': 19, 'value': 19},
                        {'label': 20, 'value': 20},{'label': 21, 'value': 21},{'label': 22, 'value': 22},{'label': 23, 'value': 23},{'label': 24, 'value': 24},
                        {'label': 25, 'value': 25},{'label': 26, 'value': 26},{'label': 27, 'value': 27},{'label': 28, 'value': 28},{'label': 29, 'value': 29},
                        {'label': 30, 'value': 30},{'label': 31, 'value': 31},{'label': 32, 'value': 32},{'label': 'None', 'value': 'None'}
                    ],
                    value = -1
                ),html.Br(),html.Br(),

                dbc.Button("valider GridSearchCV", color="info",id='KNeighborsRegressor_button_GridSearchCV',n_clicks=0),
                dcc.Loading(
                    id="KNeighborsRegressor-ls-loading-1", 
                    children=[html.Div(id="KNeighborsRegressor-ls-loading-output-1")], type="default"
                ),html.Br(),html.Hr(),html.Br(),
                
                html.H4(html.B("Paramètrage du modèle et Fit & Predict :")),html.Br(),html.Br(),
                html.B("n_neighbors "),html.I("par défaut=5"),html.Br(),html.P("Nombre de voisins à utiliser par défaut pour les requêtes de voisins.", className="card-text"),
                
                dcc.Input(id="KNeighborsRegressor_n_neighbors", type="number", placeholder="input with range",min=1,max=100, step=1,value=5),html.Br(),html.Br(),
                html.B("weights "),html.I("par défaut = 'uniform'"),html.Br(),html.P("Fonction de poids utilisée dans la prédiction."),
                dcc.Dropdown(
                    id='KNeighborsRegressor_weights',
                    options=[
                        {'label': 'uniform', 'value': 'uniform'},
                        {'label': 'distance', 'value': 'distance'},
                    ],
                    value = 'uniform'
                ),html.Br(),html.Br(),

                html.B("algorithm "),html.I("par défaut = 'auto'"),html.Br(),html.P("Algorithme utilisé pour calculer les voisins les plus proches."),
                dcc.Dropdown(
                    id='KNeighborsRegressor_algorithm',
                    options=[
                        {'label': 'auto', 'value': 'auto'},
                        {'label': 'ball_tree', 'value': 'ball_tree'},
                        {'label': 'kd_tree', 'value': 'kd_tree'},
                        {'label': 'brute', 'value': 'brute'},
                    ],
                    value = 'auto'
                ),html.Br(),html.Br(),
                html.B("leaf_size "),html.I("par défaut=30"),html.Br(),html.P("Taille de la feuille transmise à BallTree ou KDTree. Cela peut affecter la vitesse de construction et de requête, ainsi que la mémoire requise pour stocker l'arbre. La valeur optimale dépend de la nature du problème.", className="card-text"),
                dcc.Input(id="KNeighborsRegressor_leaf_size", type="number", placeholder="input with range",min=1,max=300, step=1,value=30),html.Br(),html.Br(),
                html.B("p "),html.I("par défaut=2"),html.Br(),html.P("Paramètre de puissance pour la métrique de Minkowski. Lorsque p = 1, cela équivaut à utiliser manhattan_distance (l1) et euclidean_distance (l2) pour p = 2. Pour p arbitraire, minkowski_distance (l_p) est utilisé.", className="card-text"),
                dcc.Input(id="KNeighborsRegressor_p", type="number", placeholder="input with range",min=1,max=100, step=1,value=2),html.Br(),html.Br(),
                html.B("metric "),html.I("par défaut = 'minkowski'"),html.Br(),html.P("La métrique de distance à utiliser pour l'arbre."),
                dcc.Dropdown(
                    id='KNeighborsRegressor_metric',
                    options=[
                        {'label': 'minkowski', 'value': 'minkowski'},
                        {'label': 'euclidean', 'value': 'euclidean'},
                        {'label': 'manhattan', 'value': 'manhattan'},
                        {'label': 'chebyshev', 'value': 'chebyshev'},
                        {'label': 'wminkowski', 'value': 'wminkowski'},
                        {'label': 'seuclidean', 'value': 'seuclidean'},
                        #{'label': 'mahalanobis', 'value': 'mahalanobis'},
                    ],
                    value = 'minkowski'
                ),html.Br(),html.Br(),

                html.B("test_size "),html.I("par défaut=0.3"),html.Br(),html.P("Taille du jeu de données test.", className="card-text"),
                dcc.Input(id="KNeighborsRegressor_test_size", type="number", placeholder="input with range",min=0.1,max=0.5, step=0.1,value=0.3),html.Br(),html.Br(),

                html.B("shuffle "),html.I("par défaut shuffle=True"),html.Br(),html.P("s'il faut ou non mélanger les données avant de les diviser.", className="card-text"),
                dcc.Dropdown(
                    id='KNeighborsRegressor_shuffle',
                    options=[
                        {'label': 'True', 'value': 'True'},
                        {'label': 'False', 'value': 'False'},
                    ],
                    value = 'True'
                ),html.Br(),html.Br(),

                dbc.Button("Valider Fit & Predict", color="danger",id='KNeighborsRegressor_button_FitPredict',n_clicks=0),
                dcc.Loading(id="KNeighborsRegressor-ls-loading-3", children=[html.Div(id="KNeighborsRegressor-ls-loading-output-3")], type="default"),html.Br(),html.Hr(),html.Br(),
                html.H4(html.B("validation croisée :")),html.Br(),html.Br(),
                html.B("cv_number_of_folds "),html.I("par défaut=10"),html.Br(),html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée.", className="card-text"),
                dcc.Input(id="KNeighborsRegressor_cv_number_of_folds", type="number", placeholder="input with range",min=1,max=100, step=1,value=10),html.Br(),html.Br(),
                html.B("cv_scoring "),html.I("par défaut = 'MSE'"),html.Br(),html.P("Selectionnez la méthode de scoring pour la validation croisée."),
                dcc.Dropdown(
                    id='KNeighborsRegressor_cv_scoring',
                    options=[
                        {'label': "MSE", 'value': "MSE"},
                        {'label': "RMSE", 'value': "RMSE"},
                        {'label': "MAE", 'value': "MAE"},
                    ],
                    value = 'MSE'
                ),html.Br(),html.Br(),
                dbc.Button("Valider K-Fold Cross-Validation", color="success",id='KNeighborsRegressor_button_CrossValidation',n_clicks=0),
                dcc.Loading(id="KNeighborsRegressor-ls-loading-2", children=[html.Div(id="KNeighborsRegressor-ls-loading-output-2")], type="default")


                ],className="six columns",style={"margin":10}),

            html.Div([

                    html.H3(html.B("Résultats :")),html.Br(),html.Hr(),html.Br(),
                    html.Div(id="res_KNeighborsRegressor_GridSearchCV"),html.Br(),html.Hr(),html.Br(),
                    html.Div(id="res_KNeighborsRegressor_FitPredict"),html.Br(),html.Hr(),html.Br(),
                    html.Div(id="res_KNeighborsRegressor_CrossValidation")

                ],className="six columns",style={"margin":10})

        ], className="row")

    ],
    body=True
)



# regression linéaire
Regession_regression_lineaire = dbc.Card(
    children=[

            html.H2(html.B(html.P("Linear Regression", className="card-text"))),html.Br(),html.Hr(),html.Br(),

            html.Div([

                html.Div([

                    html.H3(html.B("Paramètrage")),html.Br(),html.Hr(),
                    html.H4(html.B("Optimisation des hyperparamètres :")),html.Br(),html.Br(),
                    html.B("GridSearchCV_number_of_folds "),
                    html.I("par défaut=10"),html.Br(),
                    html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée pour l'optimisation des hyperparamètres.", className="card-text"),
                    dcc.Input(
                        id="Linear_GridSearchCV_number_of_folds",
                        type="number",
                        placeholder="input with range",
                        min=1,max=100, step=1,value=10),
                    html.Br(),html.Br(),
                    html.B("GridSearchCV_scoring "),
                    html.I("par défaut = 'R2'"),html.Br(),
                    html.P("Selectionnez la méthode de scoring pour l'optimisation des hyperparamètres."),
                    dcc.Dropdown(
                        id='Linear_GridSearchCV_scoring',
                        options=[
                            {'label': "MSE", 'value': "MSE"},
                            {'label': "MAE", 'value': "MAE"},
                            {'label':"RMSE",'value':'RMSE'}
                            
                        ],
                        value = 'MSE'
                    ),html.Br(),html.Br(),
                    html.B("GridSearchCV_njobs "),html.I("par défaut=-1"),html.Br(),
                    html.P("Selectionner le nombre de coeurs (-1 = tous les coeurs)", className="card-text"),
                    dcc.Dropdown(
                        id="Linear_GridSearchCV_njobs",
                        options=[
                            {'label': -1, 'value': -1},{'label': 1, 'value': 1},{'label': 2, 'value': 2},{'label': 3, 'value': 3},{'label': 4, 'value': 4},
                            {'label': 5, 'value': 5},{'label': 6, 'value': 6},{'label': 7, 'value': 7},{'label': 8, 'value': 8},{'label': 9, 'value': 9},
                            {'label': 10, 'value': 10},{'label': 11, 'value': 11},{'label': 12, 'value': 12},{'label': 13, 'value': 13},{'label': 14, 'value': 14},
                            {'label': 15, 'value': 15},{'label': 16, 'value': 16},{'label': 17, 'value': 17},{'label': 18, 'value': 18},{'label': 19, 'value': 19},
                            {'label': 20, 'value': 20},{'label': 21, 'value': 21},{'label': 22, 'value': 22},{'label': 23, 'value': 23},{'label': 24, 'value': 24},
                            {'label': 25, 'value': 25},{'label': 26, 'value': 26},{'label': 27, 'value': 27},{'label': 28, 'value': 28},{'label': 29, 'value': 29},
                            {'label': 30, 'value': 30},{'label': 31, 'value': 31},{'label': 32, 'value': 32},{'label': 'None', 'value': 'None'}
                        ],
                        value = -1
                    ),html.Br(),html.Br(),
                    dbc.Button("valider GridSearchCV", color="info",id='Linear_button_GridSearchCV',n_clicks=0),
                    dcc.Loading(id="ls-loading-0_linear", children=[html.Div(id="ls-loading-output-0_linear")], type="default"),html.Br(),html.Hr(),html.Br(),
                    # on passe à une autre section (modification des paramètre) >> propre à la fonction
                    html.H4(html.B("Paramètres : ")),
                    html.Hr(),html.Br(),
                    html.B("fit_intercept : ")," ",
                    dcc.Dropdown(
                        id='fit_intercept',
                        options=[
                            {'label': 'True', 'value': 'True'},
                            {'label': 'False', 'value': 'False'}
                        ],
                        value= 'True',
                    ),
                    html.Br(),
                    html.B("copy_X : "),"",
                    dcc.Dropdown(
                        id='copy_X',
                        options=[
                            {'label': 'True', 'value': 'True'},
                            {'label': 'False', 'value': 'False'}],
                        value= 'True',
                    ),
                    html.Br(),
                    html.B("n_jobs : "),"",
                    dcc.Input(
                        id='n_jobs',
                        type='number',
                        min=0,
                        max=10,
                        value=0,
                    ),
                    html.Br(),html.Br(),
                    dbc.Button("Valider Fit & Predict", color="danger",id='Linear_button_FitPredict',n_clicks=0),
                    dcc.Loading(id="ls-loading-1_linear", children=[html.Div(id="ls-loading-output-1_Linear")], type="default"),html.Br(),html.Hr(),html.Br(),
                    # on passe à une autre section (validation croissé)
                    html.H4(html.B("validation croisée :")),html.Br(),html.Br(),
                    html.B("cv_number_of_folds "),
                    html.I("par défaut=10"),html.Br(),
                    html.P("Selectionner le nombre de fois que vous souhaitez réaliser la validation croisée.", className="card-text"),
                    dcc.Input(
                        id="Linear_cv_number_of_folds", type="number", 
                        placeholder="input with range",min=1,max=100, 
                        step=1,value=10),
                    html.Br(),html.Br(),
                    html.B("cv_scoring "),html.I("par défaut = 'R2'"),html.Br(),
                    html.P("Selectionnez la méthode de scoring pour la validation croisée."),
                    dcc.Dropdown(
                        id='Linear_cv_scoring',
                        options=[
                            {'label': "MSE", 'value': "MSE"},
                            {'label': "MAE", 'value': "MAE"},
                            {'label':"RMSE",'value':'RMSE'}
                            
                        ],
                        value = 'MSE'
                    ),html.Br(),html.Hr(),
                    dbc.Button("Valider CrossValidation", color="success",id='Linear_button_CrossValidation',n_clicks=0),
                    dcc.Loading(id="ls-loading-2_Linear", children=[html.Div(id="ls-loading-output-2_Linear")], type="default")

                ],className="six columns",style={"margin":10}),

                html.Div([

                    html.H3(html.B("Résultats :")),
                    html.Div(id="res_Linear_GridSearchCV"),html.Br(),html.Hr(),html.Br(),
                    html.Div(id="res_Linear_FitPredict"),html.Br(),html.Hr(),html.Br(),
                    html.Div(id="res_Linear_CrossValidation"), html.Hr(),

                ],className="six columns",style={"margin":10}),

            ], className="row"),
        ],body=True
)
# Onglets pour les algos de régression -------------------------------------------------------------------------------------------------------
regression_tabs = dbc.Tabs(
    id='regression_tabs',
    children = [
        dbc.Tab(Regession_regression_lineaire,label="Régression linéaire",tab_id='reg_lin'),
        dbc.Tab(regression_svm,label="SVR",tab_id ='svr'),
        dbc.Tab(regression_KNeighborsRegressor,label="KNeighborsRegressor", tab_id='KNeighborsRegressor',tab_style={'background-color':'#E4F2F2','border-color':'white'},label_style={'color':'black'})
    ]
)


