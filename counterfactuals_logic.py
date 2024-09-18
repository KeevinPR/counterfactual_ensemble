import pandas as pd
# Necesary imports
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import random
from statistics import mean 
from ensemble_counterfactuals.common_funcs import train_models
from ensemble_counterfactuals.algorithms import ga, eda, moeda, nsga2, ebna, moebna
# Función para generar los contrafactuales
def generate_counterfactuals(selected_row_index, new_class, num_models, df):
    """
    Genera contrafactuales basados en los datos proporcionados.
    
    :param selected_row_index: índice de la fila seleccionada en el dataframe.
    :param new_class: la nueva clase asignada por el usuario.
    :param num_models: número de modelos seleccionados para generar los contrafactuales.
    :param df: el dataframe completo con los datos.
    
    :return: la fila original y la fila modificada (contrafactual).
    """
    # Obtener la fila seleccionada del dataframe original
    original_row = df.iloc[selected_row_index].to_dict()

    # Crear una copia de la fila para generar el contrafactual
    counterfactual_row = original_row.copy()

    # Modificar la clase del contrafactual con la nueva clase seleccionada
    counterfactual_row['class'] = new_class

    # Aquí puedes agregar lógica adicional para modificar otras características de la fila
    # en función de los modelos seleccionados. Por ejemplo, puedes hacer que ciertos valores
    # cambien de manera sistemática para simular la influencia de otros factores.

    for i in range(num_models):
        # Simulación simple: sumar un valor a todas las características numéricas
        for key, value in counterfactual_row.items():
            if isinstance(value, (int, float)):  # Si el valor es numérico
                counterfactual_row[key] = value + (i + 1)  # Agregamos una variación simple

    # Retornar tanto la fila original como la fila con los cambios contrafactuales
    # Importing a dataset from uci machine learning repository as an example
    nursery = fetch_ucirepo(id=19) 

    X = nursery.data.features 
    y = nursery.data.targets 

    X['class'] = y['class']
    class_values = y['class'].unique()
    one_sol = False
    obj_class = 'spec_prior'

    discrete_variables = [True,True,True,True,True,True]
    #discrete_variables = None
    
    # Train and test split
    train_df, test_df = train_test_split(X, test_size=0.1,random_state=188574)
    len(test_df)
    print(test_df.values[2])
    # Train the models with the train and test
    train_models(train_df,test_df)
    # Execution example
    # Change the instance number, objective class and the line after the comment
    instance_number = 2
    obj_class = "acc"
    results = []
    dfs = []
    times = []
    print(f'Instance number: {instance_number}')
    cls_val = np.delete(class_values, np.where(class_values == test_df.values[instance_number][-1]))
    print(test_df.values[instance_number])
    # Change eda.ensemble_coiunter_eda with each possible algorithm, all have the same attributes
    df, rem_models, accuracy,time = eda.ensemble_counter_eda(X=train_df,input=test_df.values[instance_number],obj_class=obj_class,test=test_df,discrete_variables=discrete_variables,verbose=False,no_train=True)
    if df is not None:
        times.append(time)
        dfs.append(df)
        models = {}
        chg_varaibles = {}
        prediction = {}
        plausible = {}
        for e in df["model"].unique():
            if e != "input":
                models[e] = []
                if e != "baseline":
                    chg_varaibles[e] = []
                    prediction[e] = []
                    plausible[e] = []

        for row in df.values:
            if row[0] == "input":
                continue
            models[row[0]].append(float(row[-2]))
            if row[0] == "baseline":
                continue
            chg_varaibles[row[0]].append(float(row[-4]))   
            prediction[row[0]].append(float(row[-3]))
            plausible[row[0]].append(float(row[-1]))

        baseline = models['baseline'][0]
        model_distances = {}
        for model_name in models.keys():
            if model_name == "input" or model_name == "baseline":
                continue
            dict_dist = {"distance":1000000000000,"baseline":None,'distance_ind':0}
            for ind, elem in enumerate(models[model_name]):
                if dict_dist["distance"] >= elem:
                    dict_dist["distance"] = elem
                    dict_dist["distance_ind"] = ind
                if elem == baseline:
                    dict_dist["baseline"] = elem
                    dict_dist["baseline_ind"] = ind
            dict_dist["chg_variables"] = chg_varaibles[model_name][dict_dist["distance_ind"]]
            dict_dist["prediction"] = prediction[model_name][dict_dist["distance_ind"]]    
            dict_dist["plausible"] = plausible[model_name][dict_dist["distance_ind"]] 
            dict_dist["accuracy"] = accuracy[rem_models.index(model_name)]
            model_distances[model_name] = dict_dist
        results.append(model_distances)
    print(f'Media tiempos: {mean(times)}')
    # Dictionary with all the results
    print(results)
    # Getting the result dataframe with all models
    print(obj_class)
    display(df)
    
    return original_row, counterfactual_row
