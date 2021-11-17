import dash
from dash import dcc
from dash import html
from dash.development.base_component import Component
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
from detect_delimiter import detect
import dash_daq as daq
import cchardet as chardet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,precision_score


from layout.layout import drag_and_drop,build_tree, build_kmeans, get_pandas_dataframe, parse_contents, location_folder, dataset_selection, target_selection,features_selection, kmeans_params_and_results,decisionTree_params_and_result

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title="Machine Learning App"

form = dbc.Form([location_folder, dataset_selection,target_selection,features_selection])
form_kmeans_params_and_results = dbc.Form([kmeans_params_and_results])
form_decision_tree_params_results = dbc.Form([decisionTree_params_and_result])

regression_models = ['Régression linéaire', 'Régression polynomiale', 'Régression lasso']
classfication_models = ['Arbre de décision','SVM','KNN',"CAH","kmeans"]

def allowed_files(path,extensions):
    allowed_files=[]
    for file in os.listdir(path):
        if file.endswith(extensions):
            allowed_files.append(file)
        else:
            continue
    return allowed_files

app.layout = html.Div(children=[
        html.Div(
            [
                html.H1('Réalisation d’une interface d’analyse de données par apprentissage supervisé'),
                html.H5(['Olivier IMBAUD, Inès KARA, Romain DUDOIT'],style={'color':'white','font-weight':'bold'}),
                html.H6('Master SISE (2021-2022)',style={'color':'white','font-weight':'bold'})
            ],className='container-fluid top'
        ),
        drag_and_drop,
        html.Div(
            [
                dbc.Row(
                    [
                        form,
                        dbc.Col(
                                html.Div(id='dataset'),width="100%"
                            )
                    ]
                )
            ]
            , className='container-fluid'
        ),
        html.Div(id='output-data-upload'), # Affichage du tableau
        html.Div(
            dbc.RadioItems(
                id="model_selection",
            ),
            className='container-fluid'
        ),
        html.Br(),
        html.Div(
            dbc.Checklist(
                id="centrer_reduire"
            ),
            className='container-fluid'
        ),
        html.Br(),
        form_kmeans_params_and_results,
        dcc.Store(id='num_variables'),
        html.Br(),
        form_decision_tree_params_results
        
])

# Récupération de la liste des fichiers autorisés dans un répertoire renseigné par l'utilisateur---------------------------------------------------------------------------
@app.callback(
    Output('file_selection','options'), # mise à jour de la liste des fichiers dans le répertoire
    Input('validation_folder', 'n_clicks'), # valeur du bouton
    State(component_id="location_folder",component_property='value') #valeur de l'input
)
def update_files_list(n_clicks,data_path):
    allowed_extensions =('.csv','.xlsx','.xls')
    # Si on a appuyer sur le bouton valider alors
    if n_clicks !=0:
        # On essaie de parcourir les fichiers dans le répertoire data_path
        try :
            files = os.listdir(r'%s' %data_path)
            filtred_files=allowed_files(data_path,allowed_extensions)
        # Si le répertoire n'existe
        except:
            return dash.no_update, '{} is prime!'.format(data_path)     ######################################/!\ Exception à reprendre

        # --- /!\ return ([{'label':f, 'value':(r'%s' %(data_path+'\\'+f))} for f in filtred_files]) # WINDOWS
        return ([{'label':f, 'value':(r'%s' %(data_path+'/'+f))} for f in filtred_files]) # LINUX / MAC-OS
    else:
        raise PreventUpdate

# Lecture du fichier choisit et mise à jour de la dropdown des variables cibles possibles --------------------------------------------------------------------------
@app.callback(
    Output(component_id='target_selection', component_property='value'),
    Output(component_id='target_selection', component_property='options'),
    Output(component_id='dataset', component_property='children'),
    Output(component_id='num_variables', component_property='data'),
    Input(component_id='file_selection', component_property='value'),
)
def FileSelection(file_path):
    if file_path is None:
        raise PreventUpdate
    else:
        df = get_pandas_dataframe(file_path)
        variables = df.columns.tolist()
        num_variables = df.select_dtypes(include=np.number).columns.tolist()
        table =dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name":i,"id":i} for i in df.columns],
                fixed_rows={'headers': True},
                page_size=20,
                sort_action='native',
                sort_mode='single',
                sort_by=[],
                style_cell={'textAlign': 'left','minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
                style_table={'height': '400px', 'overflowY': 'scroll','overflowX': 'scroll'},
                style_header={'backgroundColor': 'dark','fontWeight': 'bold'},
                style_cell_conditional=[
                    {'if': {'column_id': c},'textAlign': 'center'} for c in df.columns],
            )
        return (None,[{'label':v, 'value':v} for v in variables],table,num_variables)

# Chargement des variables pour les variables explicatives à sélectionner ---------------------------------------------------------
@app.callback(
        Output(component_id='features_selection', component_property='options'),
        Output(component_id='features_selection', component_property='value'),
        Input(component_id='target_selection', component_property='value'),
        Input(component_id='target_selection', component_property='options'),
        Input(component_id='features_selection', component_property='value')
)
def TargetSelection(target,options,feature_selection_value):
    # On commence d'abord par traiter le cas lorsque l'utilisateur n'a rien sélectionné -------------------------------------------------------------
    if target is None:
        return ([{'label':"", 'value':""}],None)
    else :
        variables = [d['value'] for d in options]
        if feature_selection_value == None:
            return (
                [{'label':v, 'value':v} for v in variables if v!=target],
                [v for v in variables if v!=target]
            )
        else:
            if len(feature_selection_value) >= 2:
                return (
                    [{'label':v, 'value':v} for v in variables if v!=target],
                    [v for v in feature_selection_value if v!=target]
                )
            else:
                return (
                    [{'label':v, 'value':v} for v in variables if v!=target],
                    [v for v in variables if v!=target]
                )

# Proposition du/des modèles qu'il est possible de sélectionner selon le type de la variable cible
@app.callback(
    Output(component_id='model_selection',component_property='options'),
    Output(component_id='centrer_reduire',component_property='options'),
    Input(component_id='num_variables',component_property='data'),
    Input(component_id='target_selection',component_property='value'),
    Input(component_id='features_selection',component_property='value'),
    Input(component_id='file_selection', component_property='value'),
    Input(component_id='model_selection',component_property='value'))
def ModelSelection(num_variables,target_selection,feature_selection,file,selected_model):
    if target_selection != None:
        if target_selection in num_variables:
                return [{"label":v,"value":v} for v in regression_models],[{"label":"centrer réduire","value":"yes"}]
        else:
            if selected_model == "Arbre de décision":
                return [{"label":v,"value":v} for v in classfication_models],[]
            else:
                return [{"label":v,"value":v} for v in classfication_models],[{"label":"centrer réduire","value":"yes"}]
    else:
        raise PreventUpdate

# Affichage des paramètres du modèle (pour le moment uniquement kmeans)
@app.callback(
    Output(component_id='kmeans-container',component_property='style'),
    Output('tree-container','style'),
    Output(component_id='n_clusters',component_property='value'),
    Input(component_id='model_selection',component_property='value'),
    Input(component_id='file_selection', component_property='value'),
    Input(component_id='target_selection',component_property='value'))
def ModelParameters(model,file_path,target):
    if file_path is None:
        raise PreventUpdate
    else:
        df = get_pandas_dataframe(file_path)
        if model == "kmeans":
            return {"margin":25,"display":"block"},len(set(list(df[target])))
        else:
            raise PreventUpdate

# update des parametres surtout affichage des parametre metric 
@app.callback(
    Output(component_id='kmeans-explore-object',component_property='options'),
    Output(component_id='kmeans-explore-object',component_property='value'),
    Input(component_id='model_selection',component_property='value'),
    Input(component_id='file_selection', component_property='value'),
    Input(component_id='target_selection',component_property='value'),
    Input(component_id='features_selection',component_property='value'),
    Input(component_id='n_clusters',component_property='value'),
    Input(component_id='init',component_property='value'),
    Input(component_id='n_init',component_property='value'),
    Input(component_id='max_iter',component_property='value'),
    Input(component_id='tol',component_property='value'),
    Input(component_id='verbose',component_property='value'),
    Input(component_id='random_state',component_property='value'),
    Input(component_id='algorithm',component_property='value'),
    Input(component_id='centrer_reduire',component_property='value'),
    Input('num_variables','data'))
def ShowModelAttributes(model,file_path,target,features,n_clusters,init,n_init,max_iter,tol,verbose,random_state,algorithm,centrer_reduire,num_variables):
    if file_path is None:
        raise PreventUpdate
    else:
        df = get_pandas_dataframe(file_path)
        if model == "kmeans":

            if any(item not in num_variables for item in features) == True:
                df_ = pd.get_dummies(df.loc[:, df.columns != target])
                features = list(df_.columns)
                df = pd.concat([df_,df[target]],axis=1)

            if random_state == "None":
                random_state = None

            kmeans = build_kmeans(df[features],n_clusters,init,n_init,max_iter,tol,verbose,random_state,algorithm,centrer_reduire)
            return [{"label":v,"value":v} for v in list(kmeans.__dict__.keys())+["randscore_"] if v.endswith("_")],"randscore_"
        else:
            raise PreventUpdate

@app.callback(
    Output(component_id='kmeans-explore-object-display',component_property='children'),
    Output(component_id='kmeans-pca',component_property='figure'),
    Output(component_id='input-pca',component_property='figure'),
    Input(component_id='model_selection',component_property='value'),
    Input(component_id='file_selection', component_property='value'),
    Input(component_id='target_selection',component_property='value'),
    Input(component_id='features_selection',component_property='value'),
    Input(component_id='n_clusters',component_property='value'),
    Input(component_id='init',component_property='value'),
    Input(component_id='n_init',component_property='value'),
    Input(component_id='max_iter',component_property='value'),
    Input(component_id='tol',component_property='value'),
    Input(component_id='verbose',component_property='value'),
    Input(component_id='random_state',component_property='value'),
    Input(component_id='algorithm',component_property='value'),
    Input(component_id='centrer_reduire',component_property='value'),
    Input(component_id='kmeans-explore-object',component_property='value'),
    Input('num_variables','data'))
def ShowModelResults(model,file_path,target,features,n_clusters,init,n_init,max_iter,tol,verbose,random_state,algorithm,centrer_reduire,kmeans_object_value,num_variables):
    if file_path is None:
        raise PreventUpdate
    else:
        df = get_pandas_dataframe(file_path)
        if model == "kmeans":
            if any(item not in num_variables for item in features) == True:
                df_ = pd.get_dummies(df.loc[:, df.columns != target])
                features = list(df_.columns)
                df = pd.concat([df_,df[target]],axis=1)

            if random_state == "None":
                random_state = None
            kmeans = build_kmeans(df[features],n_clusters,init,n_init,max_iter,tol,verbose,random_state,algorithm,centrer_reduire)
            #y = list(df[target].replace({"setosa":0,"versicolor":1,"virginica":2}))
            setattr(kmeans, 'randscore_', adjusted_rand_score(kmeans.labels_,df[target]))
            pca = PCA(n_components=2)
            temp = pca.fit_transform(df[features])
            coord = pd.DataFrame(temp,columns=["PCA1","PCA2"])
            Y_pred = pd.DataFrame(list(map(str,kmeans.labels_)),columns=["kmeans_clusters"])
            result = pd.concat([coord,Y_pred,df[target]], axis=1)
            fig_kmeans = px.scatter(result, x="PCA1", y="PCA2", color="kmeans_clusters", hover_data=['kmeans_clusters'],
                             title="PCA du jeu de données {} colorié par clusters du KMeans".format(file_path.split("/")[-1]))
            fig_input_data = px.scatter(result, x="PCA1", y="PCA2", color=target, hover_data=[target],
                             title="PCA du jeu de données {} colorié en fonction de la variable à prédire".format(file_path.split("/")[-1]))
            return html.P("{}".format(getattr(kmeans, kmeans_object_value))),fig_kmeans,fig_input_data
        else:
            raise PreventUpdate
            
# affichage des résultats arbre
@app.callback(
    Output(component_id='print_result_metric', component_property='children'),
    #Output(component_id, component_property),
    Input('model_selection','value'),
    Input('target_selection','value'),
    Input('features_selection','value'),
    Input('file_selection','value'), 
    Input('criterion','value'),
    Input('splitter','value'),
    Input('max_depth','value'),
    Input('min_samples_split','value'),
    Input('tree_diff_metric','value'))
def update_result_tree(model,target,feature,file,criterion,splitter,max_depth,min_samples_split,metric):
    #creation du dataframe
    if file is None: 
        raise PreventUpdate
    else : 
        df = get_pandas_dataframe(file)
        
        #on le fait que si model == arbre decison
        if model == "Arbre de décision" : 
            # prendre en compte le parametre None
            if max_depth == 0: 
                max_depth = None
            
            # separt en test et apprentissage 
            X = df.loc[:,feature]
            y = df.loc[:,target]
            
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state=0)
            
            #creation du model 
            tree = build_tree(X_train, y_train, criterion, splitter, max_depth, min_samples_split)
            
            cross_val = cross_val_score(tree, X,y,cv=5,scoring=metric)
            # retour la moyenne des métrics choisi
            moy = np.mean(cross_val) # sert de prédiction 
            
            
            return html.P('Résult {}'.format(str(moy)))
        else : 
            raise PreventUpdate

# @app.callback(
#     Output('test','children'),
#     Input('file_selection','value'),
#     Input('target_selection','value'),
#     Input('features_selection','value')
# )
# def display(file_selection,target_selection,feature_selection):
#     ctx=dash.callback_context

#     ctx_msg = json.dumps({
#         'states': ctx.states,
#         'triggered': ctx.triggered,
#         'inputs': ctx.inputs
#     }, indent=2)

#     return html.Div([
#         html.Pre(ctx_msg)
#     ])

# Affichage du tableau après ajout d'un fichier.
# @app.callback(Output('output-data-upload', 'children'),
#               Input('upload-data', 'contents'), # les données du fichier
#               State('upload-data', 'filename'), # nom du fichier
# )
# def update_output(list_of_contents, list_of_names):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n) for c, n in
#             zip(list_of_contents, list_of_names)]
#         return children

app.css.append_css({
'external_url': './assets/style2.css' # LINUX - MAC-OS
})

if __name__=='__main__':
    app.run_server(debug=True)
