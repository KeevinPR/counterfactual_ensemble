
"""
• Keeps every model artefact in a **single folder** (`artifacts/`)
• Auto‑re‑trains whenever the incoming dataset has a different set of columns
• Debug messages are kept but now in English
"""

import os
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from contextlib import contextmanager

# ----------------------------------------------------------------------
# rpy2 ⇄ pandas plumbing
# ----------------------------------------------------------------------
pandas2ri.activate()

@contextmanager
def rpy2_context():
    pandas2ri.activate()
    robjects.conversion.set_conversion(default_converter + pandas2ri.converter)
    with localconverter(default_converter + pandas2ri.converter):
        yield

# ----------------------------------------------------------------------
# Global flags and paths
# ----------------------------------------------------------------------
_trained_models = False          # internal cache flag
_name_order     = None           # keeps the class levels

BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "artifacts"))
os.makedirs(BASE_DIR, exist_ok=True)

MODEL_FILES    = ["nb.rds", "tan.rds", "fssj.rds", "kdb.rds", "tanhc.rds"]
MODEL_PATHS    = [os.path.join(BASE_DIR, f) for f in MODEL_FILES]
FEATURES_FILE  = os.path.join(BASE_DIR, "model_features.txt")

# ----------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------
def train_models(X, test):
    """Train the bnclassify ensemble and save it under BASE_DIR."""
    with rpy2_context():
        try:
            print("** [DEBUG PY] train_models()")
            print("   - X.shape:", X.shape, "| X.columns:", X.columns.tolist())
            print("   - test.shape:", test.shape, "| test.columns:",
                  test.columns.tolist())

            # send dataframes to R
            robjects.globalenv["r_from_pd_df"] = robjects.conversion.py2rpy(X)
            robjects.globalenv["r_test_df"]    = robjects.conversion.py2rpy(test)
            robjects.globalenv["MODEL_DIR"]    = BASE_DIR

            robjects.r('''
                library(bnclassify)

                cat("\\n[DEBUG R] train_models – column check\\n")
                cat("   setdiff(train , test):",
                    setdiff(colnames(r_from_pd_df), colnames(r_test_df)), "\\n")
                cat("   setdiff(test  , train):",
                    setdiff(colnames(r_test_df),  colnames(r_from_pd_df)), "\\n")

                # ensure factors
                txt_cols <- sapply(r_from_pd_df, is.character)
                r_from_pd_df[txt_cols] <- lapply(r_from_pd_df[txt_cols], as.factor)

                # build ensemble
                ensemble <- list()
                nb    <- lp(nb(target_col,  r_from_pd_df),  r_from_pd_df, smooth=.01)
                tn    <- lp(tan_cl(target_col, r_from_pd_df, score="aic"),
                            r_from_pd_df, smooth=.01)
                fssj  <- lp(fssj(target_col, r_from_pd_df, k=5),
                            r_from_pd_df, smooth=.01)
                kdb   <- lp(kdb(target_col,  r_from_pd_df, k=5),
                            r_from_pd_df, smooth=.01)
                tanhc <- lp(tan_hc(target_col, r_from_pd_df, k=5),
                            r_from_pd_df, smooth=.01)

                saveRDS(nb,    file.path(MODEL_DIR, "nb.rds"))
                saveRDS(tn,    file.path(MODEL_DIR, "tan.rds"))
                saveRDS(fssj,  file.path(MODEL_DIR, "fssj.rds"))
                saveRDS(kdb,   file.path(MODEL_DIR, "kdb.rds"))
                saveRDS(tanhc, file.path(MODEL_DIR, "tanhc.rds"))
            ''')

            # fingerprint of the column set
            with open(FEATURES_FILE, "w", encoding="utf‑8") as f:
                f.write(",".join(X.columns))
            print("R code train_models executed successfully")

        except Exception as e:
            print("Error in R code (train_models):", e)
            print("[DEBUG PY] R traceback:",
                  robjects.r('geterrmessage()'))

def model_ensemble():
    """Load every .rds into an R list called `ensemble`."""
    missing = [p for p in MODEL_PATHS if not os.path.isfile(p)]
    if missing:
        print("Error: missing model files:", missing)
        return
    with rpy2_context():
        try:
            robjects.globalenv["MODEL_DIR"] = BASE_DIR
            robjects.r('''
                ensemble <- list()
                ensemble$nb    <- readRDS(file.path(MODEL_DIR, "nb.rds"))
                ensemble$tn    <- readRDS(file.path(MODEL_DIR, "tan.rds"))
                ensemble$fssj  <- readRDS(file.path(MODEL_DIR, "fssj.rds"))
                ensemble$kdb   <- readRDS(file.path(MODEL_DIR, "kdb.rds"))
                ensemble$tanhc <- readRDS(file.path(MODEL_DIR, "tanhc.rds"))
            ''')
            print("Models loaded successfully.")
        except Exception as e:
            print("Error loading R models:", e)

def inicialize_ensemble(X, test):
    """Ensure the ensemble is consistent with the current dataframe."""
    global _trained_models
    with rpy2_context():
        try:
            print("** [DEBUG PY] inicialize_ensemble()")
            print("   - X.shape:", X.shape, "| X.columns:", X.columns.tolist())

            # 1. Decide if retraining is needed
            retrain = True
            if os.path.exists(FEATURES_FILE):
                with open(FEATURES_FILE, encoding="utf-8") as f:
                    retrain = (f.read().strip().split(",") != list(X.columns))

            if retrain:
                print("[DEBUG PY] Column set changed → retraining")
                # remove old artefacts
                for p in MODEL_PATHS:
                    if os.path.exists(p):
                        os.remove(p)
                train_models(X, test)
                _trained_models = False   # will be set True in ensemble_selector
            else:
                # if any model is missing, retrain as well
                if any(not os.path.exists(p) for p in MODEL_PATHS):
                    print("[DEBUG PY] Some model files are missing → retraining")
                    train_models(X, test)
                    _trained_models = False

            # 2. Expose train/test to R for later prediction
            robjects.globalenv["r_from_pd_df"] = robjects.conversion.py2rpy(X)
            robjects.globalenv["r_test_df"]    = robjects.conversion.py2rpy(test)

            robjects.r('''
                library(bnclassify)
                txt_cols <- sapply(r_from_pd_df, is.character)
                r_from_pd_df[txt_cols] <- lapply(r_from_pd_df[txt_cols], as.factor)
            ''')
            model_ensemble()

            robjects.r('''
                levels_list <- lapply(r_from_pd_df, levels)
                n <- levels_list[[target_col]]
                for (i in seq_along(r_test_df)) {
                    r_test_df[[i]] <- factor(r_test_df[[i]],
                                             levels = levels_list[[i]])
                }
            ''')
            name_order = list(robjects.globalenv['n'])
            print("R code initialize_ensemble executed successfully")
            return name_order

        except Exception as e:
            print("Error in R code within inicialize_ensemble:", e)
            print("[DEBUG PY] R traceback:", robjects.r('geterrmessage()'))
            return None

def ensemble_selector(X, inp, test, no_train):
    """Return (class_levels, remaining_models, accuracies)"""
    global _trained_models, _name_order
    with rpy2_context():
        try:
            # check current column fingerprint
            if os.path.exists(FEATURES_FILE):
                with open(FEATURES_FILE, encoding="utf-8") as f:
                    if f.read().strip().split(",") != list(X.columns):
                        _trained_models = False

            print("** [DEBUG PY] ensemble_selector()")
            print("   - input:", inp, "| no_train:", no_train)

            if not no_train or not _trained_models:
                name_order = inicialize_ensemble(X, test)
                _name_order      = name_order
                _trained_models  = True
            else:
                name_order = _name_order

            robjects.globalenv["elem"] = robjects.StrVector(inp)

            robjects.r('''
                library(bnclassify)
                lvl <- lapply(r_from_pd_df, levels)
                df  <- as.data.frame(matrix(elem, nrow = 1))
                colnames(df) <- colnames(r_from_pd_df)
                for (i in seq_along(df))
                    df[[i]] <- factor(df[[i]], levels = lvl[[i]])

                # -------- model by model --------
                outputs <- list()
                for (name in names(ensemble)) {
                    needed  <- features(ensemble[[name]])
                    missing <- setdiff(needed, colnames(df))
                    if (length(missing)) {
                        cat("Model:", name, "→ missing:",
                            paste(missing, collapse=", "), "\\n")
                        next
                    }
                    df_m <- df[, needed, drop = FALSE]
                    pred <- predict(ensemble[[name]], df_m, prob = FALSE)
                    cat("Model:", name, "Pred:", as.character(pred), "\\n")
                    if (pred == df[1, ][[target_col]])
                        outputs[[name]] <- pred
                }

                # -------- accuracies --------
                acc <- list()
                for (name in names(outputs)) {
                    needed <- features(ensemble[[name]])
                    p      <- predict(ensemble[[name]],
                                      r_test_df[, needed, drop = FALSE],
                                      prob = FALSE)
                    acc[[name]] <- accuracy(p, r_test_df[[target_col]])
                }

                remaining_models <- names(outputs)
                if (is.null(remaining_models)) remaining_models <- 0
                acurracy <- unlist(acc)
            ''')
            remaining = list(robjects.globalenv['remaining_models'])
            acc       = np.array(robjects.globalenv['acurracy'])

        except Exception as e:
            print("Error in R code:", e)
            print("[DEBUG PY] R traceback:", robjects.r('geterrmessage()'))
            remaining = []
            acc       = np.array([])

    return name_order, remaining, acc