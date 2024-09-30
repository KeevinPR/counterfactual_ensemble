import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, default_converter

# Activate the pandas to R conversion and set it globally
pandas2ri.activate()
robjects.conversion.set_conversion(default_converter + pandas2ri.converter)

_trained_models = False
_name_order = None

def train_models(X, test):
    print("Type of robjects.r before data conversion in train_models:", type(robjects.r))
    # Rename 'class' column to 'class_label' in X and test
    X = X.rename(columns={'class': 'class_label'})
    test = test.rename(columns={'class': 'class_label'})

    
    # Convert dataframes
    r_from_pd_df = robjects.conversion.py2rpy(X)
    r_test_df = robjects.conversion.py2rpy(test)

    robjects.globalenv['r_from_pd_df'] = r_from_pd_df
    robjects.globalenv['r_test_df'] = r_test_df
    # Verify robjects.r
    print("Type of robjects.r after data conversion in train_models:", type(robjects.r))
        
    # R code to preprocess data and train models
    robjects.r('''
    tryCatch({
        library(bnclassify)
        columnas_texto <- sapply(r_from_pd_df, is.character)
        r_from_pd_df[columnas_texto] <- lapply(r_from_pd_df[columnas_texto], as.factor)
    }, error = function(e) {
        print(e)
    })
    ''')

    robjects.r('''
    ensemble <- list()
    nb <- nb('class_label', r_from_pd_df)
    nb <- lp(nb, r_from_pd_df, smooth = 0.01)
    saveRDS(nb, "nb.rds")
    ensemble$nb <- nb

    tn <- tan_cl('class_label', r_from_pd_df, score = 'aic')
    tn <- lp(tn, r_from_pd_df, smooth = 0.01)
    saveRDS(tn, "tan.rds")
    ensemble$tn <- tn

    fssj_model <- fssj('class_label', r_from_pd_df, k=5)
    fssj_model <- lp(fssj_model, r_from_pd_df, smooth = 0.01)
    saveRDS(fssj_model, "fssj.rds")
    ensemble$fssj <- fssj_model

    kdb_model <- kdb('class_label', r_from_pd_df, k=5)
    kdb_model <- lp(kdb_model, r_from_pd_df, smooth = 0.01)
    saveRDS(kdb_model, "kdb.rds")
    ensemble$kdb <- kdb_model

    tanhc_model <- tan_hc('class_label', r_from_pd_df, k=5)
    tanhc_model <- lp(tanhc_model, r_from_pd_df, smooth = 0.01)
    saveRDS(tanhc_model, "tanhc.rds")
    ensemble$tanhc <- tanhc_model
    ''')
        
def model_ensemble():
    robjects.r('''
    ensemble <- list()
    nb <- readRDS("nb.rds")
    ensemble$nb <- nb

    tn <- readRDS("tan.rds")
    ensemble$tn <- tn

    fssj_model <- readRDS("fssj.rds")
    ensemble$fssj <- fssj_model

    kdb_model <- readRDS("kdb.rds")
    ensemble$kdb <- kdb_model

    tanhc_model <- readRDS("tanhc.rds")
    ensemble$tanhc <- tanhc_model
    ''')

def inicialize_ensemble(X, test):
    # Rename 'class' column to 'class_label' in X and test
    X = X.rename(columns={'class': 'class_label'})
    test = test.rename(columns={'class': 'class_label'})

    with robjects.conversion.localconverter(robjects.default_converter):
        with (robjects.default_converter + pandas2ri.converter).context():
            r_from_pd_df = robjects.conversion.get_conversion().py2rpy(X)

        with (robjects.default_converter + pandas2ri.converter).context():
            r_test_df = robjects.conversion.get_conversion().py2rpy(test)

    robjects.globalenv['r_from_pd_df'] = r_from_pd_df
    robjects.globalenv['r_test_df'] = r_test_df

    robjects.r('''
    library(bnclassify)
    columnas_texto <- sapply(r_from_pd_df, is.character)
    r_from_pd_df[columnas_texto] <- lapply(r_from_pd_df[columnas_texto], as.factor)
    ''')

    model_ensemble()

    robjects.r('''
    niveles <- lapply(r_from_pd_df, levels)
    n <- niveles[['class_label']]

    for (i in seq_along(r_test_df)) {
        r_test_df[[i]] <- factor(r_test_df[[i]], levels = niveles[[i]])
    }
    ''')

    r_vector = robjects.globalenv['n']
    name_order = list(r_vector)
    return name_order

def ensemble_selector(X, input, test, no_train):
    global _trained_models, _name_order

    # Rename 'class' column to 'class_label' in X and test
    X = X.rename(columns={'class': 'class_label'})
    test = test.rename(columns={'class': 'class_label'})

    if not no_train:
        name_order = inicialize_ensemble(X, test)
        _trained_models = False
    elif not _trained_models:
        name_order = inicialize_ensemble(X, test)
        _name_order = name_order
        _trained_models = True
    else:
        name_order = _name_order

    # Convert input instance to R vector
    r_elem = robjects.StrVector(input)
    robjects.globalenv['elem'] = r_elem

    with (robjects.default_converter + pandas2ri.converter).context():
        r_from_pd_df = robjects.conversion.get_conversion().py2rpy(X)
        r_test_df = robjects.conversion.get_conversion().py2rpy(test)

    robjects.globalenv['r_from_pd_df'] = r_from_pd_df
    robjects.globalenv['r_test_df'] = r_test_df

    robjects.r('''
    library(MLmetrics)
    niveles <- lapply(r_from_pd_df, levels)
    df <- as.data.frame(matrix(elem, nrow = 1))
    colnames(df) <- colnames(r_from_pd_df)

    for (i in seq_along(df)) {
        df[[i]] <- factor(df[[i]], levels = niveles[[i]])
    }

    outputs_list <- list()
    for (name in names(ensemble)) {
        sal <- predict(ensemble[[name]], df, prob = FALSE)
        if (sal == df[1, ]$class_label) {
            outputs_list[[name]] <- sal
        }
    }

    accu_list <- list()
    for (name in names(outputs_list)) {
        p <- predict(ensemble[[name]], r_test_df, prob = FALSE)
        accu_sal <- Accuracy(p, r_test_df$class_label)
        accu_list[[name]] <- accu_sal
    }

    remaining_models_var <- names(outputs_list)
    if (is.null(remaining_models_var)) {
        remaining_models_var <- 0
    }
    accuracy_values <- unlist(accu_list)
    ''')

    r_remaining_models_var = robjects.globalenv['remaining_models_var']
    r_accuracy_values = robjects.globalenv['accuracy_values']
    remaining_models = list(r_remaining_models_var)
    accuracy = np.array(r_accuracy_values)

    return name_order, remaining_models, accuracy
