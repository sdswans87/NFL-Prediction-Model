# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 12:19:07 2024

@author: swan0
"""
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator 
from h2o.estimators import H2ORandomForestEstimator 
from h2o.estimators.glm import H2OGeneralizedLinearEstimator 
from h2o.estimators.deeplearning import H2ODeepLearningEstimator 
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator 
import matplotlib.pyplot as plt
import nfl_data_py as nfl
import numpy as np
import pandas as pd

import nfl_methods as nfl_obj

class ensemble_model_actual_points():
    def __init__(self, src_obj, gm_obj, mod_obj):

        # start runtime
        start_run = time.time()


        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("import data runtime: " + str(end_run) + " minutes.")