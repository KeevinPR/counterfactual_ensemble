import numpy as np

import rpy2.robjects as robjects
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import conversion, default_converter

_trained_models = False
_name_order = None

def train_models(X,test):
    with (ro.default_converter + pandas2ri.converter).context():
        r_from_pd_df = ro.conversion.get_conversion().py2rpy(X)

    with (ro.default_converter + pandas2ri.converter).context():
            r_test_df = ro.conversion.get_conversion().py2rpy(test)

    robjects.globalenv['r_from_pd_df'] = r_from_pd_df
    robjects.globalenv['r_test_df'] = r_test_df
    robjects.r('''
    # Código R
    library(bnclassify)
    columnas_texto <- sapply(r_from_pd_df, is.character)
    r_from_pd_df[columnas_texto] <- lapply(r_from_pd_df[columnas_texto], as.factor)
    ''')
    robjects.r('''
    ensemble <- list()
    nb <- nb('class', r_from_pd_df)
    nb <- lp(nb, r_from_pd_df, smooth = 0.01)
    saveRDS(nb,"nb.rds")      
    ensemble$nb = nb
               
    tn <- tan_cl('class', r_from_pd_df, score = 'aic')
    tn <- lp(tn, r_from_pd_df, smooth = 0.01)
                   saveRDS(tn,"tan.rds")  
    ensemble$tn = tn   
               
    fssj <- fssj('class', r_from_pd_df, k=5)
    fssj <- lp(fssj, r_from_pd_df, smooth = 0.01)
                   saveRDS(fssj,"fssj.rds")  
    ensemble$fssj = fssj
               
    kdb <- kdb('class', r_from_pd_df, k=5)
    kdb <- lp(kdb, r_from_pd_df, smooth = 0.01)
                   saveRDS(kdb,"kdb.rds")  
    ensemble$kdb = kdb
               
    tanhc <- tan_hc('class', r_from_pd_df, k=5)
    tanhc <- lp(tanhc, r_from_pd_df, smooth = 0.01)
                   saveRDS(tanhc,"tanhc.rds")  
    ensemble$tanhc = tanhc         
    ''')

def model_ensemble():
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

def inicialize_ensemble(X,test):
    with conversion.localconverter(default_converter):
        with (ro.default_converter + pandas2ri.converter).context():
                r_from_pd_df = ro.conversion.get_conversion().py2rpy(X)

        with (ro.default_converter + pandas2ri.converter).context():
                r_test_df = ro.conversion.get_conversion().py2rpy(test)

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
    n = niveles$class
               
    for (i in seq_along(r_test_df)) {
    r_test_df[[i]] <- factor(r_test_df[[i]], levels = niveles[[i]])
    }
    ''')
    r_vector = robjects.globalenv['n']
    name_order = list(r_vector)
    return name_order

def ensemble_selector(X,input,test,no_train):
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
        if(sal==df[1,]$class){
            outputs[[name]] = sal            
        }
    }
    
    
    accu <- list()
    for (name in names(outputs)){
        p = predict(ensemble[[name]], r_test_df, prob = FALSE)
        accu_sal <- accuracy(p,r_test_df$class)
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
    print(remaing_models)
    accuracy = np.array(r_accuracy)

    return name_order, remaing_models, accuracy