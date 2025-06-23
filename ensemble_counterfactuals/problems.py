import numpy as np
from pymoo.core.problem import Problem

import rpy2.robjects as robjects








# Standart Problem for multi-objetive optimization
class Ensemble_Problem(Problem):
    def __init__(self,codification,x_instance,y_desired,discrete_variables,model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_instance_des = x_instance
        self.x_instance_cod = codification.encode(x_instance)
        self.y_desired_dec =  y_desired
        self.y_desired_cod =  codification.encode_class(y_desired)
        cls = codification.cls
        ind = list(cls).index(y_desired)
        zeros = [0]*len(cls)
        zeros[ind] = 1
        self.desired_prob = zeros
        self.cod = codification
        self.max = codification.get_max()
        self.min = codification.get_min()
        self.discrete_variables = discrete_variables
        self.model_name = model_name
        self.plausible_rows = []
        for _,a in self.cod.X.iterrows():
            if a[-1] != self.y_desired_dec:
                continue
            self.plausible_rows.append(a)


    def _evaluate(self, x, out, *args, **kwargs):
        obj1 = []
        obj2 = []
        obj3 = []
        obj4 = []
        rest = []
        elems = []
        for elem in x:
            a = np.array(np.round(elem[0:len(elem)-1]))
            b = np.array(self.x_instance_cod[0:len(elem)-1])
            obj1.append(np.sum(a != b))

            r_name = robjects.StrVector([self.model_name])
            robjects.globalenv['model_name'] = r_name
            r_elem = robjects.StrVector(self.cod.decode(elem))
            robjects.globalenv['elem'] = r_elem
            robjects.r('''
            niveles <- lapply(r_from_pd_df, levels)
            df <- as.data.frame(matrix(elem, nrow = 1))
            colnames(df) <- colnames(r_from_pd_df)
            for (i in seq_along(df)) {
            df[[i]] <- factor(df[[i]], levels = niveles[[i]])
            }

            sal <- predict(ensemble[[model_name]], df, prob = TRUE)
            ''')
            r_vector = robjects.globalenv['sal']
            sal = list(r_vector)

            obj2.append(np.linalg.norm(np.array(sal) - np.array(self.desired_prob), ord=np.inf))
            elem_round = np.round(elem[0:len(elem)-1])
            dist = 0
            for i,e in enumerate(elem_round):
                if self.discrete_variables is not None and self.discrete_variables[i]:
                    dist += abs(e-self.x_instance_cod[i])/(self.max[i]-self.min[i])
                else:
                    dist += e!=self.x_instance_cod[i]
            dist = dist * 1/len(elem[0:len(elem)-1])
            obj3.append(dist)
            
            elems.append(elem_round)

            robjects.r('''
            re <- predict(ensemble[[model_name]], df, prob = FALSE)
    
            ''')
            r_vector = robjects.globalenv['re']
            re = list(r_vector)
            rest.append((re[0]-1)==self.y_desired_cod)


        less_dist = [10000000000]*len(elems)

        for a in self.plausible_rows:
            a_cod = self.cod.encode(a)
            for index,element in enumerate(elems):
                d = 0
                for i,e in enumerate(a_cod[0:len(a_cod)-1]):
                    if self.discrete_variables != None and self.discrete_variables[i]:
                        d += abs(e-element[i])/(self.max[i]-self.min[i])
                    else:
                        d += e!=element[i]
                d = d * 1/len(a)
                if d < less_dist[index]:
                    less_dist[index] = d
        obj4 = less_dist
        out['F'] = np.column_stack([np.array(obj1),np.array(obj2),np.array(obj3),np.array(obj4)])

        for i,e in enumerate(rest):
            if e:
                rest[i] = 0
            else:
                rest[i] = 1

        out['G'] = np.column_stack([np.array(rest)])








# Problem for multi-objective optimization with only 2 objectives: Distance and Plausibility
class Ensemble_Problem2(Problem):
    def __init__(self,codification,x_instance,y_desired,discrete_variables,model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_instance_des = x_instance
        self.x_instance_cod = codification.encode(x_instance)
        self.y_desired_dec =  y_desired
        self.y_desired_cod =  codification.encode_class(y_desired)
        cls = codification.cls
        ind = list(cls).index(y_desired)
        zeros = [0]*len(cls)
        zeros[ind] = 1
        self.desired_prob = zeros
        self.cod = codification
        self.max = codification.get_max()
        self.min = codification.get_min()
        self.discrete_variables = discrete_variables
        self.model_name = model_name
        self.plausible_rows = []
        for _,a in self.cod.X.iterrows():
            if a[-1] != self.y_desired_dec:
                continue
            self.plausible_rows.append(a)


    def _evaluate(self, x, out, *args, **kwargs):
        obj1 = []
        obj2 = []
        obj3 = []
        obj4 = []
        rest = []
        elems = []
        for elem in x:
            r_name = robjects.StrVector([self.model_name])
            robjects.globalenv['model_name'] = r_name
            r_elem = robjects.StrVector(self.cod.decode(elem))
            robjects.globalenv['elem'] = r_elem
            robjects.r('''
            niveles <- lapply(r_from_pd_df, levels)
            df <- as.data.frame(matrix(elem, nrow = 1))
            colnames(df) <- colnames(r_from_pd_df)
            for (i in seq_along(df)) {
            df[[i]] <- factor(df[[i]], levels = niveles[[i]])
            }

            ''')

            elem_round = np.round(elem[0:len(elem)-1])
            dist = 0
            for i,e in enumerate(elem_round):
                if self.discrete_variables is not None and self.discrete_variables[i]:
                    dist += abs(e-self.x_instance_cod[i])/(self.max[i]-self.min[i])
                else:
                    dist += e!=self.x_instance_cod[i]
            dist = dist * 1/len(elem[0:len(elem)-1])
            obj3.append(dist)
            
            elems.append(elem_round)

            robjects.r('''
            re <- predict(ensemble[[model_name]], df, prob = FALSE)
    
            ''')
            r_vector = robjects.globalenv['re']
            re = list(r_vector)
            rest.append((re[0]-1)==self.y_desired_cod)


        less_dist = [10000000000]*len(elems)

        for a in self.plausible_rows:
            a_cod = self.cod.encode(a)
            for index,element in enumerate(elems):
                d = 0
                for i,e in enumerate(a_cod[0:len(a_cod)-1]):
                    if self.discrete_variables != None and self.discrete_variables[i]:
                        d += abs(e-element[i])/(self.max[i]-self.min[i])
                    else:
                        d += e!=element[i]
                d = d * 1/len(a)
                if d < less_dist[index]:
                    less_dist[index] = d
        obj4 = less_dist
        out['F'] = np.column_stack([np.array(obj3),np.array(obj4)])

        for i,e in enumerate(rest):
            if e:
                rest[i] = 0
            else:
                rest[i] = 1

        out['G'] = np.column_stack([np.array(rest)])


# Problem for single objective optimization
class GA_Problem(Problem):
    def __init__(self,codification,x_instance,y_desired,discrete_variables,model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_instance_des = x_instance
        self.x_instance_cod = codification.encode(x_instance)
        self.y_desired_dec =  y_desired
        self.y_desired_cod =  codification.encode_class(y_desired)
        cls = codification.cls
        ind = list(cls).index(y_desired)
        zeros = [0]*len(cls)
        zeros[ind] = 1
        self.desired_prob = zeros
        self.cod = codification
        self.max = codification.get_max()
        self.min = codification.get_min()
        self.discrete_variables = discrete_variables
        self.model_name = model_name
        self.plausible_rows = []
        for _,a in self.cod.X.iterrows():
            if a[-1] != self.y_desired_dec:
                continue
            self.plausible_rows.append(a)


    def _evaluate(self, x, out, *args, **kwargs):
        obj1 = []
        obj2 = []
        obj3 = []
        obj4 = []
        rest = []
        elems = []
        for elem in x:
            r_name = robjects.StrVector([self.model_name])
            robjects.globalenv['model_name'] = r_name
            r_elem = robjects.StrVector(self.cod.decode(elem))
            robjects.globalenv['elem'] = r_elem
            robjects.r('''
            niveles <- lapply(r_from_pd_df, levels)
            df <- as.data.frame(matrix(elem, nrow = 1))
            colnames(df) <- colnames(r_from_pd_df)
            for (i in seq_along(df)) {
            df[[i]] <- factor(df[[i]], levels = niveles[[i]])
            }

            ''')
            elem_round = np.round(elem[0:len(elem)-1])
            dist = 0
            for i,e in enumerate(elem_round):
                if self.discrete_variables is not None and self.discrete_variables[i]:
                    dist += abs(e-self.x_instance_cod[i])/(self.max[i]-self.min[i])
                else:
                    dist += e!=self.x_instance_cod[i]
            dist = dist * 1/len(elem[0:len(elem)-1])
            obj3.append(dist)
            
            elems.append(elem_round)

            robjects.r('''
            re <- predict(ensemble[[model_name]], df, prob = FALSE)
    
            ''')
            r_vector = robjects.globalenv['re']
            re = list(r_vector)
            rest.append((re[0]-1)==self.y_desired_cod)


        out['F'] = np.column_stack([np.array(obj3)])

        for i,e in enumerate(rest):
            if e:
                rest[i] = 0
            else:
                rest[i] = 1

        out['G'] = np.column_stack([np.array(rest)])