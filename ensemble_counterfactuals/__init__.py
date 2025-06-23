import warnings
import os
from .common_funcs import train_models, model_ensemble, inicialize_ensemble, ensemble_selector
from .codification import Codificacion
from . import algorithms

# Configurar la variable de entorno R_HOME
#os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.2'
os.environ['R_HOME'] = '/home/ubuntu/miniconda3/envs/dash_counterfactuals/lib/R'  # For Ubuntu
warnings.filterwarnings('ignore')