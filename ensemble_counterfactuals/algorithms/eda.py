import numpy as np
import pandas as pd

import rpy2.robjects as robjects

from EDAspy.optimization import UMDAcat

from IPython.display import display
import matplotlib.pyplot as plt

import time
from statistics import mean

# Estas funciones hay que importarlas:
from ensemble_counterfactuals.codification import Codificacion
from ensemble_counterfactuals.common_funcs import ensemble_selector

def ensemble_counter_eda(X, input, obj_class, test,
                         discrete_variables=None,
                         verbose=True,
                         no_train=False):
    # == DEBUG ==> Print shape/columns
    print("** [DEBUG] ensemble_counter_eda()")
    print("   - X.shape:", X.shape, "| X.columns:", list(X.columns))
    print("   - test.shape:", test.shape, "| test.columns:", list(test.columns))
    print("   - input:", input, "| obj_class:", obj_class)
    print("   - discrete_variables:", discrete_variables)

    # Llamamos a ensemble_selector
    name_order, remaing_models, accuracy = ensemble_selector(X, input, test, no_train)

    # Si no hay modelos que clasifiquen la instancia original -> salimos
    if remaing_models[0] == 0:
        return None, None, None, None

    # == DEBUG ==> Print which models are valid
    print("** [DEBUG] remaing_models:", remaing_models)
    print("** [DEBUG] name_order:", name_order)
    print("** [DEBUG] accuracy:", accuracy)

    # Codificamos
    cod = Codificacion(X, name_order)

    # == DEBUG ==> Check cod.X shape/columns
    print("** [DEBUG] cod.X.shape:", cod.X.shape, "| cod.X.columns:", list(cod.X.columns))

    models_list = ['input']

    # Suponemos que la última columna de X es la clase,
    # y solo cogemos las columnas [0 : len(X.columns)-1] como predictores
    columns = list(X.columns)[0:len(X.columns)-1]
    columns.append('Chg Variables')
    columns.append('Prediction score')
    columns.append('Distance')
    columns.append('Plausible')
    new_df = pd.DataFrame(columns=columns)

    # Creamos primera fila representando la instancia original
    new_fila = list(input)[0:len(input)-1]
    new_fila.extend(['-'] * (len(columns) - len(new_fila)))
    new_df.loc[len(new_df)] = new_fila

    times = []

    # == DEBUG ==> We'll see which column is used as class in "cod.X"
    # Buscamos filas en cod.X cuyo "último" valor sea obj_class
    plausible_rows = []
    for idx, a in cod.X.iterrows():
        # == DEBUG ==>
        # Print the last column name + value:
        last_col_name = cod.X.columns[-1]
        last_val = a.iloc[-1]
        # The code checks: if a[-1] != obj_class -> skip
        print(f"** [DEBUG] Checking row {idx}, last_col='{last_col_name}', value='{last_val}'")
        if a.iloc[-1] != obj_class:
            continue
        plausible_rows.append(a)

    # == DEBUG ==> How many plausible rows
    print(f"** [DEBUG] #plausible_rows={len(plausible_rows)} for obj_class='{obj_class}'")

    # Recorremos cada modelo en remaing_models
    for i, model_name in enumerate(remaing_models):
        start_time = time.time()
        # Ejecutamos _eda_ensemble
        sol, score = _eda_ensemble(input, obj_class, cod, X,
                                   model_name,
                                   one_sol=False,
                                   discrete_variables=discrete_variables)
        if verbose:
            print(model_name)
            print(accuracy[i])

        if sol is None:
            continue
        df = pd.DataFrame(sol)
        # Eliminamos la última columna (coste o lo que sea)
        df = df.drop(df.columns[-1], axis=1)
        models_list.append(model_name)

        # Preparamos predict en R
        r_elem = robjects.StrVector([str(el) for el in cod.decode(sol)])
        robjects.globalenv['elem'] = r_elem
        robjects.r('''
        niveles <- lapply(r_from_pd_df, levels)
        df <- as.data.frame(matrix(elem, nrow = 1))
        colnames(df) <- colnames(r_from_pd_df)
        for (i in seq_along(df)) {
          df[[i]] <- factor(df[[i]], levels = niveles[[i]])
        }
        re <- predict(ensemble[[model_name]], df, prob = TRUE)
        ''')
        r_vector = robjects.globalenv['re']
        re = list(r_vector)
        cls = cod.cls
        # Buscamos índice de la clase deseada
        ind = list(cls).index(obj_class)
        zeros = [0]*len(cls)
        zeros[ind] = 1
        desired_prob = zeros

        # Calcular la 'prediction' como distancia
        import numpy as np
        prediction = np.linalg.norm(np.array(re) - np.array(desired_prob),
                                    ord=np.inf)

        sol_decoded = cod.decode(sol)[0:len(sol)-1]
        ch_var = 0

        # Distancia a la fila plausible más cercana
        less_dist = 1e15  # algo grande
        for a in plausible_rows:
            a_cod = cod.encode(a)
            d = 0
            for j, e in enumerate(a[0:len(a)-1]):
                if discrete_variables is not None and discrete_variables[j]:
                    # normalizamos
                    d += abs(a_cod[j]-sol[j]) / (cod.get_max()[j]-cod.get_min()[j])
                else:
                    d += (e != sol_decoded[j])
            d = d * 1/len(a)
            if d < less_dist:
                less_dist = d

        # Preparamos la fila con "Chg Variables", etc.
        x = []
        for ind2, el in enumerate(sol_decoded):
            if el != input[ind2]:
                x.append(el)
                ch_var += 1
            else:
                x.append('-')

        new_row = x
        new_row += (ch_var, prediction, score, less_dist)
        new_df.loc[len(new_df)] = new_row

        end_time = time.time()
        times.append(end_time - start_time)

    # baseline
    less_distance = 1e15
    sol_less_distance = None
    input_cod = cod.encode(input)

    # == DEBUG ==>
    print("** [DEBUG] Searching for 'baseline' solution in X, matching obj_class...")

    for idx, a in X.iterrows():
        if a.iloc[-1] != obj_class:
            continue
        a_cod = cod.encode(a)
        d = 0
        for j, e in enumerate(a_cod[0:len(a_cod)-1]):
            if discrete_variables is not None and discrete_variables[j]:
                d += abs(e - input_cod[j]) / (cod.get_max()[j] - cod.get_min()[j])
            else:
                d += (e != input_cod[j])
        d = d * 1 / len(a[0:len(a)-1])
        if d < less_distance:
            less_distance = d
            sol_less_distance = a

    changed_variables = 0
    x = []
    if sol_less_distance is not None:
        # Creamos fila baseline
        for ind, el in enumerate(sol_less_distance):
            if ind < len(input) and el != input[ind]:
                x.append(el)
                if ind != len(input)-1:
                    changed_variables += 1
            else:
                x.append('-')
        # Agregamos al final
        new_row = x[0:-1]
        new_row += [changed_variables, '-', less_distance, '-']
        new_df.loc[len(new_df)] = new_row
        models_list.append('baseline')
    else:
        # == DEBUG ==> No baseline found
        print("** [DEBUG] No baseline found for obj_class=", obj_class)

    new_df = pd.concat([pd.Series(models_list, name='model'), new_df], axis=1)

    if verbose:
        display(new_df)

    return new_df, remaing_models, accuracy, mean(times)

# Variables globales para la función de coste
_discrete_variable = None
_x_instance_cod = None
_max = None
_min = None
_codification = None
_model_name = None
_y_desired_cod = None

def _categorical_cost_function(solution):
    # Aseguramos que solution sea entero
    solution = solution.astype(int)
    global _discrete_variable, _x_instance_cod, _max, _min, _codification, _model_name, _y_desired_cod

    # == DEBUG ==>
    print("** [DEBUG] _categorical_cost_function() - solution:", solution)

    r_name = robjects.StrVector([_model_name])
    robjects.globalenv['model_name'] = r_name

    # Decodificamos la solución para enviar a R
    decoded_sol = _codification.decode(solution)
    r_elem = robjects.StrVector(decoded_sol)
    robjects.globalenv['elem'] = r_elem

    # == DEBUG ==>
    print("** [DEBUG] Decoded solution for R:", decoded_sol)

    robjects.r('''
    niveles <- lapply(r_from_pd_df, levels)
    df <- as.data.frame(matrix(elem, nrow = 1))
    colnames(df) <- colnames(r_from_pd_df)
    for (i in seq_along(df)) {
      df[[i]] <- factor(df[[i]], levels = niveles[[i]])
    }
    re <- predict(ensemble[[model_name]], df, prob = FALSE)
    ''')
    elem_round = np.round(solution[0:len(solution)-1])

    # Calculamos distancia con la instancia original
    dist = 0
    for i, e in enumerate(elem_round):
        if _discrete_variable is not None and _discrete_variable[i]:
            dist += abs(e - _x_instance_cod[i]) / (_max[i] - _min[i])
        else:
            dist += (e != _x_instance_cod[i])
    dist = dist * 1 / len(solution[0:len(solution)-1])

    r_vector = robjects.globalenv['re']
    re = list(r_vector)
    predicted_class = re[0]

    print(f"** [DEBUG] Predicted class: {predicted_class}")
    try:
        re_encoded = _codification.encode_class(predicted_class)
    except ValueError:
        print(f"** [DEBUG] Error: Predicted class '{predicted_class}' not found in codification.cls")
        dist += 1000  # Penalizamos fuertemente
        return dist

    print(f"** [DEBUG] Encoded predicted class: {re_encoded}")
    print(f"** [DEBUG] Desired class code: {_y_desired_cod}")

    # Si la clase predicha no coincide con la deseada, penalizamos
    if re_encoded != _y_desired_cod:
        dist += 100
    return dist


def _eda_ensemble(x_instance, y_desired, codification, X,
                  model_name,
                  pop_size=20,
                  n_gen=20,
                  period=10,
                  one_sol=True,
                  discrete_variables=None,
                  graphic=False):
    global _discrete_variable, _x_instance_cod, _max, _min, _codification, _model_name, _y_desired_cod

    # == DEBUG ==> Print some data
    print("** [DEBUG] _eda_ensemble() - model_name:", model_name)
    print("   x_instance:", x_instance, "| y_desired:", y_desired)
    print("   codification.X.shape:", codification.X.shape, "| X.shape:", X.shape)
    print("   pop_size:", pop_size, "n_gen:", n_gen, "period:", period)

    # Convert min y max a enteros
    min_values = np.array(codification.get_min(), dtype=int)
    max_values = np.array(codification.get_max(), dtype=int)

    _discrete_variable = discrete_variables
    _x_instance_cod = codification.encode(x_instance)
    _max = max_values
    _min = min_values
    _codification = codification
    _model_name = model_name
    _y_desired_cod = codification.encode_class(y_desired)

    # Preparamos los valores posibles para UMDAcat
    pos_values = []
    for i, e in enumerate(min_values):
        pos_values.append(list(range(e, max_values[i] + 1)))

    freqs = []
    for elem in pos_values:
        freqs.append([1 / len(elem)] * len(elem))

    # Construimos UMDAcat
    ebna = UMDAcat(
        size_gen=pop_size,
        max_iter=n_gen,
        dead_iter=period,
        n_variables=len(min_values),
        alpha=0.8,
        possible_values=pos_values,
        frequency=freqs
    )

    ebna.w_noise = 0  # Quitamos ruido

    ebna_result = ebna.minimize(_categorical_cost_function, False)

    # == DEBUG ==>
    print("** [DEBUG] UMDAcat finished. Best cost:", ebna_result.best_cost)
    print("** [DEBUG] Best individual:", ebna_result.best_ind)

    if graphic:
        algs = ebna_result.history
        time_ = list(range(0, len(algs)))
        plt.figure(figsize=(8, 6))
        plt.scatter(time_, algs, facecolors='none', edgecolors='b')
        plt.xlabel('Generation')
        plt.ylabel('Best Distance')
        plt.title(f'{model_name}')

    return ebna_result.best_ind, ebna_result.best_cost