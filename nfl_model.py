# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 06:25:03 2023

@author: swan0
"""
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator 
from h2o.estimators import H2ORandomForestEstimator 
from h2o.estimators.glm import H2OGeneralizedLinearEstimator 
from h2o.estimators.deeplearning import H2ODeepLearningEstimator 
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator 
import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_squared_error
import time


class model_data_class():
    def __init__(self, import_obj, epa_obj, eff_obj):
        
        # start runtime
        start_run = time.time()

        # ensemble column lists
        off_columns = ['game_id','year', 'week', 'away_team', 'home_team','posteam',
                       'n_pass','n_rush','n_plays', 'home_final','away_final',
                       'total_tds','total_fgs_att','total_pat','total_effective_pts',
                       'total_score','pt_diff']
        def_columns = ['game_id','year', 'week', 'away_team', 'home_team','defteam',
                       'n_pass','n_rush','n_plays', 'home_final','away_final',
                       'total_tds','total_fgs_att','total_pat','total_effective_pts',
                       'total_score','pt_diff']
        
        # prep for model calculation
        self.ensemble_offense = self.prep_data_func(eff_obj.total_off_efficiency, off_columns)
        self.ensemble_defense = self.prep_data_func(eff_obj.total_def_efficiency, def_columns)

        # ensemble model offensive efficiency/actual points scored
        h2o.init()
        self.offensive_train_h2o = self.ensemble_func(self.ensemble_offense, 0)
        self.offensive_test_h2o = self.ensemble_func(self.ensemble_offense, 1)
        self.defensive_train_h2o = self.ensemble_func(self.ensemble_defense, 0)
        self.defensive_test_h2o = self.ensemble_func(self.ensemble_defense, 1)

        # train gradient boosting
        self.offensive_gbm = self.gbm_model_func(self.offensive_train_h2o, 'poss_score', 
                                                 set(self.offensive_train_h2o.columns) - 
                                                 set(['poss_score']), 5, 5)
        self.defensive_gbm = self.gbm_model_func(self.defensive_train_h2o, 'score_allowed', 
                                                 set(self.defensive_train_h2o.columns) - 
                                                 set(['score_allowed']), 5, 5)

        # train random forest
        self.offensive_rf = self.rf_model_func(self.offensive_train_h2o, 'poss_score', 
                                               set(self.offensive_train_h2o.columns) - 
                                               set(['poss_score']), 5, 5)
        self.defensive_rf = self.rf_model_func(self.defensive_train_h2o, 'score_allowed', 
                                               set(self.defensive_train_h2o.columns) - 
                                               set(['score_allowed']), 5, 5)

        # train linear regression
        self.offensive_lr = self.lr_model_func(self.offensive_train_h2o, 'poss_score', 
                                               set(self.offensive_train_h2o.columns) - 
                                               set(['poss_score']), 5, 5)
        self.defensive_lr = self.lr_model_func(self.defensive_train_h2o, 'score_allowed', 
                                               set(self.defensive_train_h2o.columns) -
                                               set(['score_allowed']), 5, 5)

        # train neural net
        self.offensive_nn = self.nn_model_func(self.offensive_train_h2o, 'poss_score', 
                                               set(self.offensive_train_h2o.columns) - 
                                               set(['poss_score']), 5, 5)
        self.defensive_nn = self.nn_model_func(self.defensive_train_h2o, 'score_allowed', 
                                               set(self.defensive_train_h2o.columns) - 
                                               set(['score_allowed']), 5, 5)
        
        # train stacked random forest ensemble using gradient boosting/random forest/linear regression
        self.offensive_ensemble = self.ensemble_model_func(self.offensive_lr, self.offensive_rf, self.offensive_nn, 
                                                           self.offensive_gbm, self.offensive_train_h2o, 
                                                           'poss_score', set(self.offensive_train_h2o.columns) - 
                                                           set(['poss_score']))
        self.defensive_ensemble = self.ensemble_model_func(self.defensive_lr, self.defensive_rf, self.defensive_nn, 
                                                           self.defensive_gbm, self.defensive_train_h2o, 
                                                           'score_allowed', set(self.defensive_train_h2o.columns) - 
                                                           set(['score_allowed']))
        
        # view gbm performance
        self.offensive_gbm_test = self.model_performance_func(self.offensive_gbm)
        self.defensive_gbm_test = self.model_performance_func(self.defensive_gbm)

        # view random fores performance
        self.offensive_rf_test = self.model_performance_func(self.offensive_rf)
        self.defensive_rf_test = self.model_performance_func(self.defensive_rf)

        # view linear regression performance
        self.offensive_lr_test = self.model_performance_func(self.offensive_lr)
        self.defensive_lr_test = self.model_performance_func(self.defensive_lr)

        # view deep laerning performance
        self.offensive_nn_test = self.model_performance_func(self.offensive_nn)
        self.defensive_nn_test = self.model_performance_func(self.defensive_nn)

        # view ensemble performance
        self.offensive_ensemble_test = self.model_performance_func(self.offensive_ensemble)
        self.defensive_ensemble_test = self.model_performance_func(self.defensive_ensemble)
        self.offensive_ensemble_test.rmse()
        self.defensive_ensemble_test.rmse()
        min([self.offensive_gbm_test.rmse(), self.offensive_rf_test.rmse(), self.offensive_lr_test.rmse(), self.offensive_nn_test.rmse()])
        min([self.defensive_gbm_test.rmse(), self.defensive_rf_test.rmse(), self.defensive_lr_test.rmse(), self.defensive_nn_test.rmse()])

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("model data runtime: " + str(end_run) + " minutes.")


    def prep_data_func(self, eff_data, eff_columns):
        df = eff_data.drop(columns=eff_columns)
        smp_size = int(0.80 * df.shape[0])
        np.random.seed(5)
        train_test = np.random.choice(df.index, size=smp_size, replace=False)
        df_train = df.loc[train_test]
        df_test = df.drop(train_test)
        return [df_train, df_test, df]


    def ensemble_func(self, eff_data, index):
        ensemble_h2o = h2o.H2OFrame(eff_data[index])
        return ensemble_h2o
    

    def gbm_model_func(self, df, y, x, nfolds, seed):
        m = H2OGradientBoostingEstimator(nfolds=nfolds, keep_cross_validation_predictions=True, seed=seed)
        out = m.train(x=x, y=y, training_frame=df)
        return out
    

    def rf_model_func(self, df, y, x, nfolds, seed):
        m = H2ORandomForestEstimator(nfolds=nfolds, keep_cross_validation_predictions=True, seed=seed)
        out = m.train(x=x, y=y, training_frame=df)
        return out
    

    def lr_model_func(self, df, y, x, nfolds, seed):
        m = H2OGeneralizedLinearEstimator(nfolds=nfolds, keep_cross_validation_predictions=True, seed=seed)
        out = m.train(x=x, y=y, training_frame=df)
        return out
    
    
    def nn_model_func(self, df, y, x, nfolds, seed):
        m = H2ODeepLearningEstimator(nfolds=nfolds, keep_cross_validation_predictions=True, seed=seed)
        out = m.train(x=x, y=y, training_frame=df)
        return out
    
    
    def ensemble_model_func(self, mod, mod2, mod3, mod4, df, y, x):
        m = H2OStackedEnsembleEstimator(metalearner_algorithm="glm", base_models=[mod, mod2, mod3, mod4])
        out = m.train(x=x, y=y, training_frame=df)
        return out
    

    def model_performance_func(self, df):
        out = H2OGradientBoostingEstimator.model_performance(df)
        return out
    

class model_effective_points():
    def __init__(self, import_obj, epa_obj, eff_obj, mod_obj):
        
        # start runtime
        start_time = time.time()

        # model efficiency columns
        off_columns = ['game_id','year', 'week', 'away_team', 'home_team','posteam',
                       'n_pass','n_rush','n_plays', 'home_final','away_final',
                       'total_tds','total_fgs_att','total_pat','total_score',
                       'pt_diff']
        def_columns = ['game_id','year', 'week', 'away_team', 'home_team','defteam',
                       'n_pass','n_rush','n_plays', 'home_final','away_final',
                       'total_tds','total_fgs_att','total_pat','total_score',
                       'pt_diff']

        # build models off of effective points
        self.ensemble_off_eff = mod_obj.prep_data_func(eff_obj.total_off_efficiency, off_columns)
        self.ensemble_def_eff = mod_obj.prep_data_func(eff_obj.total_def_efficiency, def_columns)
        self.ensemble_off_eff_train = mod_obj.ensemble_func(self.ensemble_off_eff, 0)
        self.ensemble_off_eff_test = mod_obj.ensemble_func(self.ensemble_off_eff, 1)
        self.ensemble_def_eff_train = mod_obj.ensemble_func(self.ensemble_def_eff, 0)
        self.ensemble_def_eff_test = mod_obj.ensemble_func(self.ensemble_def_eff, 1)

        # train gradient boosting
        self.offensive_eff_gbm = mod_obj.gbm_model_func(self.ensemble_off_eff_train, 'total_effective_pts',
                                                        set(self.ensemble_off_eff_train.columns) - 
                                                        set(['total_effective_pts']), 5, 5)
        self.defensive_eff_gbm = mod_obj.gbm_model_func(self.ensemble_def_eff_train, 'total_effective_pts', 
                                                        set(self.ensemble_def_eff_train.columns) - 
                                                        set(['total_effective_pts']), 5, 5)
        
        # train random forest
        self.offensive_eff_rf = mod_obj.rf_model_func(self.ensemble_off_eff_train, 'total_effective_pts',
                                                      set(self.ensemble_off_eff_train.columns) - 
                                                      set(['total_effective_pts']), 5, 5)
        self.defensive_eff_rf = mod_obj.rf_model_func(self.ensemble_def_eff_train, 'total_effective_pts', 
                                                      set(self.ensemble_def_eff_train.columns) - 
                                                      set(['total_effective_pts']), 5, 5)
        
        # train linear regression
        self.offensive_eff_lr = mod_obj.lr_model_func(self.ensemble_off_eff_train, 'total_effective_pts',
                                                      set(self.ensemble_off_eff_train.columns) - 
                                                      set(['total_effective_pts']), 5, 5)
        self.defensive_eff_lr = mod_obj.lr_model_func(self.ensemble_def_eff_train, 'total_effective_pts',
                                                      set(self.ensemble_def_eff_train.columns) - 
                                                      set(['total_effective_pts']), 5, 5)
        
        # train neural net
        self.offensive_eff_nn = mod_obj.nn_model_func(self.ensemble_off_eff_train, 'total_effective_pts',
                                                      set(self.ensemble_off_eff_train.columns) - 
                                                      set(['total_effective_pts']), 5, 5)
        self.defensive_eff_nn = mod_obj.nn_model_func(self.ensemble_def_eff_train, 'total_effective_pts',
                                                      set(self.ensemble_def_eff_train.columns) - 
                                                      set(['total_effective_pts']), 5, 5)

        # train stacked random forest ensemble using gradient boosting/random forest/linear regression
        self.offensive_eff_ensemble = mod_obj.ensemble_model_func(self.offensive_eff_lr, self.offensive_eff_rf, self.offensive_eff_nn, 
                                                                  self.offensive_eff_gbm, self.ensemble_off_eff_train, 
                                                                  'total_effective_pts', set(self.ensemble_off_eff_train.columns) - 
                                                                  set(['total_effective_pts']))
        self.defensive_eff_ensemble = mod_obj.ensemble_model_func(self.defensive_eff_lr, self.defensive_eff_rf, self.defensive_eff_nn, 
                                                                  self.defensive_eff_gbm, self.ensemble_def_eff_train, 
                                                                  'total_effective_pts', set(self.ensemble_def_eff_train.columns) - 
                                                                  set(['total_effective_pts']))
        
        # check model performance
        self.offensive_eff_gbm_test = mod_obj.model_performance_func(self.offensive_eff_gbm)
        self.defensive_eff_gbm_test = mod_obj.model_performance_func(self.defensive_eff_gbm)
        self.offensive_eff_rf_test = mod_obj.model_performance_func(self.offensive_eff_rf)
        self.defensive_eff_rf_test = mod_obj.model_performance_func(self.defensive_eff_rf)
        self.offensive_eff_lr_test = mod_obj.model_performance_func(self.offensive_eff_lr)
        self.defensive_eff_lr_test = mod_obj.model_performance_func(self.defensive_eff_lr)
        self.offensive_eff_nn_test = mod_obj.model_performance_func(self.offensive_eff_nn)
        self.defensive_eff_nn_test = mod_obj.model_performance_func(self.defensive_eff_nn)
        self.offensive_eff_ensemble_test = mod_obj.model_performance_func(self.offensive_eff_ensemble)
        self.defensive_eff_ensemble_test = mod_obj.model_performance_func(self.defensive_eff_ensemble)
        
        # calculate mins
        min([self.offensive_eff_gbm_test.rmse(), self.offensive_eff_rf_test.rmse(), self.offensive_eff_lr_test.rmse(), 
             self.offensive_eff_nn_test.rmse()])
        min([self.defensive_eff_gbm_test.rmse(), self.defensive_eff_rf_test.rmse(), self.defensive_eff_lr_test.rmse(), 
             self.defensive_eff_nn_test.rmse()])
        self.offensive_eff_ensemble_test.rmse()
        self.defensive_eff_ensemble_test.rmse()