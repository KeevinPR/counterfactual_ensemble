import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

import rpy2.robjects as robjects

from EDAspy.optimization import UMDAcat

from IPython.display import display
import matplotlib.pyplot as plt

from typing import Union, List
import time
from statistics import mean

from ensemble_counterfactuals.codification import Codificacion
from ensemble_counterfactuals.common_funcs import ensemble_selector
from ensemble_counterfactuals.problems import Ensemble_Problem


class UMDAmod(UMDAcat):
    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 possible_values: Union[List, np.array],
                 frequency: Union[List, np.array],
                 alpha: float = 0.5,
                 elite_factor: float = 0.4,
                 disp: bool = True,
                 parallelize: bool = False,
                 init_data: np.array = None):
        super().__init__(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter,
                         n_variables=n_variables,possible_values=possible_values,frequency=frequency, alpha=alpha, 
                         elite_factor=elite_factor, disp=disp, parallelize=parallelize, init_data=init_data)
        self.bestdistance = None

    def _check_generation_no_parallel(self, objective_function: callable):
        global _codification, _problem

        individuals = []
        for x in self.generation:
            ind = Individual()
            # ind._X = _codification.encode(x)
            ind._X = x
            individuals.append(ind)

        algo = NSGA2(pop_size=50)
        algo.setup(_problem)
        eva = Evaluator()
        pop = Population(individuals=individuals)
        algo.evaluator.eval(_problem,pop,algorithm=algo)
        r = RankAndCrowding()
        _problem.evaluate
        sol = r.do(_problem,pop,n_survive=50,algorithm=algo)
        #self.evaluations = np.apply_along_axis(objective_function, 1, self.generation)
        solution = []
        for elem in sol:
            solution.append(elem.X)
        # evaluations = [solution.index(e) for e in self.generation]
        evaluations = [np.where(np.all(sublista == solution, axis=1))[0][0] for sublista in self.generation]
        for i,e in enumerate(evaluations):
            if sol[e].G[0] != 0:
                evaluations[i] +=10
            evaluations[i] += sol[e].F[2]

        self.evaluations = np.array(evaluations)
        self.bestdistance = sol[0].F[2]


def ensemble_counter_moeda(X,input,obj_class,test,discrete_variables=None,verbose=True,no_train=False):
    input = [str(elemento) for elemento in input]
    X = X.astype(str)
    test = test.astype(str)
    name_order, remaing_models, accuracy = ensemble_selector(X,input,test,no_train)
    if remaing_models[0] == 0:
        return None, None, None, None
    cod = Codificacion(X,name_order)

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

    times = []

    plausible_rows = []
    for _,a in cod.X.iterrows():
        if a[-1] != obj_class:
            continue
        plausible_rows.append(a)

    for i,model_name in enumerate(remaing_models):
        start_time = time.time()        
        sol,score = _eda_ensemble_mod(input,obj_class,cod,X,model_name,one_sol=False,discrete_variables=discrete_variables)
        if verbose:
            print(model_name)
            print(accuracy[i])
        # print(sol)
        # print(score)
        if sol is None:
            continue
        df = pd.DataFrame(sol)
        df = df.drop(df.columns[-1], axis=1)
        models_list.append(model_name)
        r_elem = robjects.StrVector(cod.decode(sol))
        robjects.globalenv['elem'] = r_elem
        robjects.r('''
        niveles <- lapply(r_from_pd_df, levels)
        df <- as.data.frame(matrix(elem, nrow = 1))
        colnames(df) <- colnames(r_from_pd_df)
        for (i in seq_along(df)) {
        df[[i]] <- factor(df[[i]], levels = niveles[[i]])
        }

        re <- predict(tn, df, prob = TRUE)

        ''')
        r_vector = robjects.globalenv['re']
        re = list(r_vector)
        cls = cod.cls
        ind = list(cls).index(obj_class)
        zeros = [0]*len(cls)
        zeros[ind] = 1
        desired_prob = zeros
        prediction = np.linalg.norm(np.array(re) - np.array(desired_prob), ord=np.inf)
        sol_decoded = cod.decode(sol)[0:len(sol)-1]
        ch_var = 0

        less_dist = 1000000000

        for a in plausible_rows:
            a_cod = cod.encode(a)
            d = 0
            for i,e in enumerate(a[0:len(a)-1]):
                if discrete_variables != None and discrete_variables[i]:
                    d += abs(a_cod[i]-sol[i])/(cod.get_max()[i]-cod.get_min()[i])
                else:
                    d += e!=sol_decoded[i]
            d = d * 1/len(a)
            if d < less_dist:
                less_dist = d
                x = []
                
        for ind,el in enumerate(sol_decoded):
            if el != input[ind]:
                x.append(el)
                ch_var+=1
            else:
                x.append('-')
        
        new_row = x
        new_row += (ch_var,prediction,score,less_dist)
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

def _categorical_cost_function(solution):
    global _discrete_variable, _x_instance_cod, _max, _min, _codification, _model_name, _y_desired_cod
    r_name = robjects.StrVector([_model_name])
    robjects.globalenv['model_name'] = r_name
    r_elem = robjects.StrVector(_codification.decode(solution))
    robjects.globalenv['elem'] = r_elem
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
    dist = 0
    for i,e in enumerate(elem_round):
        if _discrete_variable is not None and _discrete_variable[i]:
            dist += abs(e-_x_instance_cod[i])/(_max[i]-_min[i])
        else:
            dist += e!=_x_instance_cod[i]
    dist = dist * 1/len(solution[0:len(solution)-1])

    r_vector = robjects.globalenv['re']
    re = list(r_vector)
    if (re[0]-1)!=_y_desired_cod:
        dist+=100
    return dist


def _eda_ensemble_mod(x_instance,y_desired,codification,X,model_name,pop_size=20,n_gen=20,period=20,one_sol=True,discrete_variables=None,graphic=False):
    global _problem,_discrete_variable, _x_instance_cod, _max, _min, _codification, _model_name, _y_desired_cod
    min_values = codification.get_min()
    max_values = codification.get_max()

    problem = Ensemble_Problem(codification,x_instance,y_desired,discrete_variables,model_name,xl=min_values,xu=max_values,type_var=int,n_var=len(x_instance), n_obj=4,n_constr=1)    

    _discrete_variable = discrete_variables
    _x_instance_cod = codification.encode(x_instance)
    _max = max_values
    _min = min_values
    _codification = codification
    _model_name = model_name
    _y_desired_cod = codification.encode_class(y_desired)
    _problem = problem

    pos_values = []
    for i,e in enumerate(min_values):
        pos_values.append(list(range(e,max_values[i]+1)))

    freqs = []
    for elem in pos_values:
        freqs.append([1/len(elem)]*len(elem))

    ebna = UMDAmod(size_gen=pop_size, max_iter=n_gen, dead_iter=period, n_variables=len(min_values), alpha=0.5,
        possible_values=pos_values, frequency=freqs)

    ebna_result = ebna.minimize(_categorical_cost_function, False)
    if graphic:
        algs = ebna_result.history
        time = list(range(0,len(algs)))

        plt.figure(figsize=(8, 6))
        plt.scatter(time,algs,facecolors='none', edgecolors='b')
        plt.xlabel('Generation')
        plt.ylabel('Best Distance')
        plt.title(f'{model_name}')
    return ebna_result.best_ind,ebna_result.best_cost