import os
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from contextlib import contextmanager


# Activate the pandas to R conversion and set it globally
pandas2ri.activate()

@contextmanager
def rpy2_context():
    # Activar pandas2ri y conversiones locales
    pandas2ri.activate()
    robjects.conversion.set_conversion(default_converter + pandas2ri.converter)
    
    with localconverter(default_converter + pandas2ri.converter):
        #try:
            yield
        #finally:
            # Desactivar pandas2ri después del uso
            #pandas2ri.deactivate()
            
# Variables globales para el control del modelo       
_trained_models = False
_name_order = None

def train_models(X, test):
    with rpy2_context():
        try:
            # Convert dataframes using localconverter
            r_from_pd_df = robjects.conversion.py2rpy(X)
            r_test_df = robjects.conversion.py2rpy(test)

            robjects.globalenv['r_from_pd_df'] = r_from_pd_df
            robjects.globalenv['r_test_df'] = r_test_df
            
            robjects.r('''
            # Código R
            library(bnclassify)
            columnas_texto <- sapply(r_from_pd_df, is.character)
            r_from_pd_df[columnas_texto] <- lapply(r_from_pd_df[columnas_texto], as.factor)
            
            # Crear y guardar los modelos
            ensemble <- list()
            nb <- nb(target_col, r_from_pd_df)
            nb <- lp(nb, r_from_pd_df, smooth = 0.01)
            saveRDS(nb, "nb.rds")
            ensemble$nb <- nb
                    
            tn <- tan_cl(target_col, r_from_pd_df, score = 'aic')
            tn <- lp(tn, r_from_pd_df, smooth = 0.01)
            saveRDS(tn, "tan.rds")
            ensemble$tn <- tn 
                    
            fssj <- fssj(target_col, r_from_pd_df, k=5)
            fssj <- lp(fssj, r_from_pd_df, smooth = 0.01)
            saveRDS(fssj, "fssj.rds")
            ensemble$fssj <- fssj
                    
            kdb <- kdb(target_col, r_from_pd_df, k=5)
            kdb <- lp(kdb, r_from_pd_df, smooth = 0.01)
            saveRDS(kdb, "kdb.rds")
            ensemble$kdb <- kdb
                    
            tanhc <- tan_hc(target_col, r_from_pd_df, k=5)
            tanhc <- lp(tanhc, r_from_pd_df, smooth = 0.01)
            saveRDS(tanhc, "tanhc.rds")
            ensemble$tanhc <- tanhc       
            ''')
            print("R code train_models executed successfully")
        except Exception as e:
            print(f"Error in R code: {e}")
        
def model_ensemble():
    # Verificar la existencia de los archivos .rds
    model_files = ["nb.rds", "tan.rds", "fssj.rds", "kdb.rds", "tanhc.rds"]
    missing_files = [file for file in model_files if not os.path.isfile(file)]
    if missing_files:
        print(f"Error: Los siguientes archivos de modelos faltan: {missing_files}")
        return
    with rpy2_context():
        try:
            # Cargar los modelos en R
            robjects.r('''
            ensemble <- list()
            nb <- readRDS("nb.rds")
            ensemble$nb = nb
            
            tn <- readRDS("tan.rds")
            ensemble$tn = tn

            fssj <- readRDS("fssj.rds")
            ensemble$fssj = fssj

            kdb <- readRDS("kdb.rds")
            ensemble$kdb = kdb

            tanhc <- readRDS("tanhc.rds")
            ensemble$tanhc = tanhc
            ''')
            print("Modelos cargados exitosamente.")
        except Exception as e:
            print(f"Error loading R models: {e}")

def inicialize_ensemble(X, test):
    with rpy2_context():
        try:
            # Si no existen los archivos .rds, entrenar los modelos
            if not all(os.path.isfile(file) for file in ["nb.rds", "tan.rds", "fssj.rds", "kdb.rds", "tanhc.rds"]):
                #print("Entrenando modelos...")
                train_models(X, test)
                
            r_from_pd_df = robjects.conversion.py2rpy(X)
            r_test_df = robjects.conversion.py2rpy(test)

            robjects.globalenv['r_from_pd_df'] = r_from_pd_df
            robjects.globalenv['r_test_df'] = r_test_df
            
            robjects.r('''
            # Código R
            library(bnclassify)
            columnas_texto <- sapply(r_from_pd_df, is.character)
            r_from_pd_df[columnas_texto] <- lapply(r_from_pd_df[columnas_texto], as.factor)
            ''')
            model_ensemble()
            
            robjects.r('''
            niveles <- lapply(r_from_pd_df, levels)
            n = niveles[[target_col]]
                    
            for (i in seq_along(r_test_df)) {
            r_test_df[[i]] <- factor(r_test_df[[i]], levels = niveles[[i]])
            }
            ''')
            print("R code initialize_ensemble executed successfully")
            r_vector = robjects.globalenv['n']
            name_order = list(r_vector)
            return name_order
        except Exception as e:
            print(f"Error in R code within inicialize_ensemble: {e}")
            traceback = robjects.r('geterrmessage()')
            #print(f"R traceback: {traceback[0]}")
            return None

def ensemble_selector(X,input,test,no_train):
    with rpy2_context():
        try:
            global _trained_models, _name_order
            if not no_train:
                name_order = inicialize_ensemble(X,test)
                _trained_models = False
            elif not _trained_models:
                name_order = inicialize_ensemble(X,test)
                _name_order = name_order
                _trained_models = True
            else:
                name_order = _name_order
            # model_ensemble()
            r_elem = robjects.StrVector(input)
            robjects.globalenv['elem'] = r_elem
            robjects.r('''
            niveles <- lapply(r_from_pd_df, levels)
            df <- as.data.frame(matrix(elem, nrow = 1))
            colnames(df) <- colnames(r_from_pd_df)
            for (i in seq_along(df)) {
            df[[i]] <- factor(df[[i]], levels = niveles[[i]])
            }

            outputs <- list()
            for (name in names(ensemble)){
                sal = predict(ensemble[[name]], df, prob = FALSE)
                cat("Modelo:", name, "Sol:", as.character(sal), "\n")
                if(sal==df[1,][[target_col]]){
                    outputs[[name]] = sal            
                }
            }
            
            
            accu <- list()
            for (name in names(outputs)){
                p = predict(ensemble[[name]], r_test_df, prob = FALSE)
                accu_sal <- accuracy(p,r_test_df[[target_col]])
                accu[[name]] = accu_sal
                cat("Modelo:", name, "Sol:", accu_sal, "\n")
            }        
            remaining_models = names(outputs)
                    if (is.null(remaining_models)){
                    remaining_models <- 0}
            acurracy = unlist(accu)
            ''')
            r_remaining_models = robjects.globalenv['remaining_models']
            r_accuracy = robjects.globalenv['acurracy']
            remaing_models = list(r_remaining_models)
            #print(remaing_models)
            accuracy = np.array(r_accuracy)
            #print("R code ensemble_selector executed successfully")
        except Exception as e:
            print(f"Error in R code: {e}")
    return name_order, remaing_models, accuracy

