import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter



# Activate the pandas to R conversion and set it globally
pandas2ri.activate()

_trained_models = False
_name_order = None

def train_models(X, test):
    import os

    pandas2ri.activate()
    robjects.conversion.set_conversion(default_converter + pandas2ri.converter)
     
    # Convert 'class' to categorical if not already
    X['class'] = X['class'].astype('category')
    test['class'] = test['class'].astype('category')

    try:
        # Convert dataframes using localconverter
        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
            r_from_pd_df = robjects.conversion.py2rpy(X)
            r_test_df = robjects.conversion.py2rpy(test)

        robjects.globalenv['r_from_pd_df'] = r_from_pd_df
        robjects.globalenv['r_test_df'] = r_test_df
        
        robjects.r('''
        library(bnclassify)
        r_from_pd_df[] <- lapply(r_from_pd_df, as.factor)
        niveles <- lapply(r_from_pd_df, levels)

        r_test_df[] <- Map(function(col, lev) factor(col, levels = lev), r_test_df, niveles)

        # Train models and assign to ensemble
        ensemble <- list()
        nb <- nb('class', r_from_pd_df)
        nb <- lp(nb, r_from_pd_df, smooth = 0.01)
        ensemble$nb <- nb

        tn <- tan_cl('class', r_from_pd_df, score = 'aic')
        tn <- lp(tn, r_from_pd_df, smooth = 0.01)
        ensemble$tn <- tn

        fssj <- fssj('class', r_from_pd_df, k=5)
        fssj <- lp(fssj, r_from_pd_df, smooth = 0.01)
        ensemble$fssj <- fssj

        kdb <- kdb('class', r_from_pd_df, k=5)
        kdb <- lp(kdb, r_from_pd_df, smooth = 0.01)
        ensemble$kdb <- kdb

        tanhc <- tan_hc('class', r_from_pd_df, k=5)
        tanhc <- lp(tanhc, r_from_pd_df, smooth = 0.01)
        ensemble$tanhc <- tanhc

        # Assign ensemble to global environment
        assign("ensemble", ensemble, envir = .GlobalEnv)
        ''')
        print("R code train_models executed successfully")
    except Exception as e:
        print(f"Error in R code: {e}")
    finally:
        # Desactiva el conversor global
        pandas2ri.deactivate()
        
def model_ensemble():
    pass


def inicialize_ensemble(X, test):
    
    # Convert 'class' to categorical if not already
    X['class'] = X['class'].astype('category')
    test['class'] = test['class'].astype('category')
    
    # Mantener los mismos niveles para la columna 'class'
    class_levels = X['class'].cat.categories
    X['class'] = X['class'].cat.set_categories(class_levels)
    test['class'] = test['class'].cat.set_categories(class_levels)
    
    # Convert dataframes using localconverter
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_from_pd_df = robjects.conversion.py2rpy(X)
        r_test_df = robjects.conversion.py2rpy(test)

    robjects.globalenv['r_from_pd_df'] = r_from_pd_df
    robjects.globalenv['r_test_df'] = r_test_df

    try:
        robjects.r('''
        library(bnclassify)
        # Convert all columns to factors
        r_from_pd_df[] <- lapply(r_from_pd_df, factor)
        niveles <- lapply(r_from_pd_df, levels)

        # Debug statements
        print("Names in niveles:")
        print(names(niveles))

        # Ensure r_test_df columns are factors with the same levels
        r_test_df[] <- Map(function(col, lev) factor(col, levels = lev), r_test_df, niveles)

        # Retrieve class levels
        n <- niveles[['class']]
        ''')
        print("R code initialize_ensemble executed successfully")
    except Exception as e:
        print(f"Error in R code within inicialize_ensemble: {e}")
        traceback = robjects.r('geterrmessage()')
        print(f"R traceback: {traceback[0]}")
        return None
    
    # Check if 'n' exists in R global environment
    if 'n' in robjects.globalenv:
        r_vector = robjects.globalenv['n']
        name_order = list(r_vector)
        print("Variable 'n' retrieved from R environment.")
    else:
        print("Variable 'n' not found in R environment.")
        return None
    

    return name_order



def ensemble_selector(X, input, test, no_train):
    global _trained_models, _name_order

    if not _trained_models:
        train_models(X, test)
        _trained_models = True
        name_order = inicialize_ensemble(X, test)
        _name_order = name_order
    else:
        name_order = _name_order

    # Check if name_order is None (in case of error in inicialize_ensemble)
    if name_order is None:
        print("Error initializing ensemble.")
        return None, None, None

    # Ensure input is a list of strings (original labels)
    input_str = [str(val) for val in input]  # Should already be strings
    r_elem = robjects.StrVector(input_str)
    robjects.r.assign('elem', r_elem)


    # Load ensemble models if not already loaded
    if not _trained_models:
        model_ensemble()
        _trained_models = True

    try:
        robjects.r('''
        library(bnclassify)
        niveles <- lapply(r_from_pd_df, levels)

        # Prepare input instance as data frame with correct factor levels
        df <- as.data.frame(matrix(elem, nrow = 1))
        colnames(df) <- colnames(r_from_pd_df)
        df[] <- Map(function(col, lev) factor(col, levels = lev), df, niveles)

        # Check for NAs in df after setting factor levels
        if (any(is.na(df))) {
            print("NAs found in df after setting factor levels:")
            print(df)
            print("Levels in training data:")
            print(niveles)
            stop("Input instance contains values not present in training data levels.")
        }

        # Proceed with predictions
        outputs_list <- list()
        for (name in names(ensemble)) {
            sal <- predict(ensemble[[name]], df, prob = FALSE)
            if (sal == df[1, ]$class) {
                outputs_list[[name]] <- sal
            }
        }

        accu_list <- list()
        for (name in names(outputs_list)) {
            p <- predict(ensemble[[name]], r_test_df, prob = FALSE)
            accu_sal <- bnclassify::accuracy(p, r_test_df$class)
            accu_list[[name]] <- accu_sal
        }

        remaining_models_var <- names(outputs_list)
        if (length(remaining_models_var) == 0) {
            remaining_models_var <- 0
        }
        accuracy_values <- unlist(accu_list)
        ''')
        print("R code ensemble_selector executed successfully")
    except Exception as e:
        print(f"Error in R code: {e}")
        # Retrieve detailed error message from R
        traceback = robjects.r('geterrmessage()')
        print(f"R traceback: {traceback[0]}")
        return None, None, None

    # Extract remaining models and accuracy from R
    if 'remaining_models_var' in robjects.globalenv and 'accuracy_values' in robjects.globalenv:
        r_remaining_models_var = robjects.globalenv['remaining_models_var']
        r_accuracy_values = robjects.globalenv['accuracy_values']
        remaining_models = list(r_remaining_models_var)
        accuracy = np.array(r_accuracy_values)
    else:
        print("Variables 'remaining_models_var' or 'accuracy_values' not found in R environment.")
        return None, None, None

    return name_order, remaining_models, accuracy

