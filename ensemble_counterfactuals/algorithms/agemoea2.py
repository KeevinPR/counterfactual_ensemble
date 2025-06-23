import numpy as np
import pandas as pd

from pymoo.optimize import minimize
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.termination.default import DefaultMultiObjectiveTermination

import rpy2.robjects as robjects

from IPython.display import display
import matplotlib.pyplot as plt

import time
from statistics import mean

from ensemble_counterfactuals.codification import Codificacion
from ensemble_counterfactuals.common_funcs import ensemble_selector
from ensemble_counterfactuals.problems import Ensemble_Problem

def ensemble_counter_AGE_MOEA2(X,input,obj_class,test,discrete_variables=None,verbose=True,no_train=False):
    input = [str(elemento) for elemento in input]
    X = X.astype(str)
    test = test.astype(str)
    name_order, remaing_models, accuracy = ensemble_selector(X,input,test,no_train)
    if remaing_models[0] == 0:
        return None, None, None, None
    cod = Codificacion(X,name_order)

    times = []

    models_list = ['input']
    columns = list(X.columns)[0:len(X.columns)-1]
    columns.append('Chg Variables')
    columns.append('Prediction score')
    columns.append('Distance')
    columns.append('Plausible')
    new_df = pd.DataFrame(columns=columns)
    new_fila = list(input)[0:len(input)-1]
    new_fila.extend(['-'] * (len(columns) - len(new_fila)))
    new_df.loc[len(new_df)] = new_fila
    for i,model_name in enumerate(remaing_models):
        start_time = time.time()
        sol,score = _AGE_MOEA2_ensemble(input,obj_class,cod,X,model_name,one_sol=False,discrete_variables=discrete_variables)
        if verbose:
            print(model_name)
            print(accuracy[i])

        if sol is None:
            # end_time = time.time()
            # times.append(end_time-start_time)
            continue
        df = pd.DataFrame(sol)
        df = df.drop(df.columns[-1], axis=1)
        df_unique = df.drop_duplicates()
        indices_filas_unicas = df_unique.index
        solutions =  [sol[idx] for idx in list(indices_filas_unicas)]
        scores = [score[idx] for idx in list(indices_filas_unicas)]

        for i,e in enumerate(solutions):
            models_list.append(model_name)
            r_elem = robjects.StrVector(cod.decode(e))
            robjects.globalenv['elem'] = r_elem
            robjects.r('''
            niveles <- lapply(r_from_pd_df, levels)
            df <- as.data.frame(matrix(elem, nrow = 1))
            colnames(df) <- colnames(r_from_pd_df)
            for (i in seq_along(df)) {
            df[[i]] <- factor(df[[i]], levels = niveles[[i]])
            }

            re <- predict(tn, df, prob = FALSE)

            ''')
            r_vector = robjects.globalenv['re']
            re = list(r_vector)
            sol_decoded = cod.decode(e)[0:len(e)-1]
            x = []
            for ind,el in enumerate(sol_decoded):
                if el != input[ind]:
                    x.append(el)
                else:
                    x.append('-')
            new_row = x
            new_row += list(scores[i])
            new_df.loc[len(new_df)] = new_row

        end_time = time.time()
        times.append(end_time-start_time)

    less_distance = 1000000000
    sol_less_distance = None
    input_cod = cod.encode(input)
    for _,a in X.iterrows():
        if a[-1] != obj_class:
            continue
        a_cod = cod.encode(a)
        d = 0
        for i,e in enumerate(a_cod[0:len(a_cod)-1]):
            if discrete_variables != None and discrete_variables[i]:
                d += abs(e-input_cod[i])/(cod.get_max()[i]-cod.get_min()[i])
            else:
                d += e!=input_cod[i]
        d = d * 1/len(a[0:len(a)-1])
        if d < less_distance:
            less_distance = d
            sol_less_distance = a


    changed_variables = 0
    x = []
    for ind,el in enumerate(sol_less_distance):
        if el != input[ind]:
            x.append(el)
            if ind != len(input)-1:
                changed_variables+=1
        else:
            x.append('-')
    new_row = x[0:-1]
    new_row += [changed_variables,'-',less_distance,'-']
    new_df.loc[len(new_df)] = new_row

    models_list.append('baseline')
    new_df = pd.concat([pd.Series(models_list,name='model'),new_df],axis=1)
    if verbose:
        display(new_df)
        
    return new_df, remaing_models, accuracy, mean(times)


def _AGE_MOEA2_ensemble(x_instance,y_desired,codification,X,model_name,pop_size=20,n_gen=20,period=10,one_sol=True,discrete_variables=None,graphic=False):

    min_values = codification.get_min()
    max_values = codification.get_max()
    problem = Ensemble_Problem(codification,x_instance,y_desired,discrete_variables,model_name,xl=min_values,xu=max_values,type_var=int,n_var=len(x_instance), n_obj=4,n_constr=1)    

    algo = AGEMOEA2(pop_size=pop_size,save_history=True)


    stop = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=0.0025,
        period=period,
        n_max_gen=n_gen,
    )

    # result = minimize(problem=problem,algorithm=algo,termination=get_termination("n_gen", n_gen),save_history=True)
    result = minimize(problem=problem,algorithm=algo,termination=stop,save_history=True)
    if result.F is None:
        return np.array([codification.encode(x_instance)]).astype(int),[[10000000000000,10000000000000,10000000000000,10000000000000]]
    if one_sol:
        gbf = result.F[0]
        sol = np.round(result.X[0]).astype(int)
        return sol,gbf
    if graphic:
        algs = result.history
        time = list(range(0,len(algs)))
        best_values = []
        for x in algs:
            l = []
            for e in x.opt:
                l.append(e.F[2])
            best_values.append(min(l))

        plt.figure(figsize=(8, 6))
        plt.scatter(time,best_values,facecolors='none', edgecolors='b')
        plt.xlabel('Generation')
        plt.ylabel('Best Distance')
        plt.title(f'{model_name}')


    gbf = result.F
    sol = np.round(result.X).astype(int)
    return sol,gbf