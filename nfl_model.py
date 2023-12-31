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
        self.off_columns = ['game_id','year', 'week', 'away_team', 'home_team','posteam',
                            'n_pass','n_rush','n_plays', 'home_final','away_final',
                            'total_tds','total_fgs_att','total_pat','total_effective_pts',
                            'total_score','pt_diff']
        self.def_columns = ['game_id','year', 'week', 'away_team', 'home_team','defteam',
                            'n_pass','n_rush','n_plays', 'home_final','away_final',
                            'total_tds','total_fgs_att','total_pat','total_effective_pts',
                            'total_score','pt_diff']
        
        # prep for model calculation
        self.ensemble_offense = self.prep_data_func(eff_obj.total_off_efficiency, self.off_columns)
        self.ensemble_defense = self.prep_data_func(eff_obj.total_def_efficiency, self.def_columns)

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
        print(min([self.offensive_gbm_test.rmse(), self.offensive_rf_test.rmse(), self.offensive_lr_test.rmse(), self.offensive_nn_test.rmse()]))
        print(min([self.defensive_gbm_test.rmse(), self.defensive_rf_test.rmse(), self.defensive_lr_test.rmse(), self.defensive_nn_test.rmse()]))

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
        columns = ['game_id','year', 'week', 'away_team', 'home_team','posteam',
                   'n_pass','n_rush','n_plays', 'home_final','away_final',
                   'total_tds','total_fgs_att','total_pat','total_score',
                   'pt_diff']

        # build models off of effective points
        self.ensemble_off_eff = mod_obj.prep_data_func(eff_obj.total_off_efficiency, columns)
        self.ensemble_def_eff = mod_obj.prep_data_func(eff_obj.total_off_efficiency, columns)
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
        print(min([self.offensive_eff_gbm_test.rmse(), self.offensive_eff_rf_test.rmse(), self.offensive_eff_lr_test.rmse(), 
                   self.offensive_eff_nn_test.rmse()]))
        print(min([self.defensive_eff_gbm_test.rmse(), self.defensive_eff_rf_test.rmse(), self.defensive_eff_lr_test.rmse(), 
                   self.defensive_eff_nn_test.rmse()]))
        self.offensive_eff_ensemble_test.rmse()
        self.defensive_eff_ensemble_test.rmse()

        # create final offense/defense efficiency 
        self.final_off = eff_obj.total_offense_eff[["game_id", "posteam", "total_effective_pts"]]
        self.final_def = eff_obj.total_defense_eff[["game_id", "defteam", "total_effective_pts"]]

        # run off models on newer data
        ensemble_off = mod_obj.prep_data_func(eff_obj.total_offense_eff, mod_obj.off_columns + ["poss_score"])
        self.ensemble_off = mod_obj.ensemble_func(ensemble_off, 2)

        # run off models on newer data
        ensemble_def = mod_obj.prep_data_func(eff_obj.total_defense_eff, mod_obj.def_columns + ["score_allowed"])
        self.ensemble_def = mod_obj.ensemble_func(ensemble_def, 2)

        # set offensive targets and predictors
        preds_off = self.offensive_eff_ensemble.predict(self.ensemble_off)
        self.preds_off = preds_off.as_data_frame()
        
        # set defensive targets and predictors
        preds_def = self.defensive_eff_ensemble.predict(self.ensemble_def)
        self.preds_def = preds_def.as_data_frame()

        # set final targets and predictors
        self.final_offense = pd.concat([self.final_off, self.preds_off], axis=1)
        self.final_defense = pd.concat([self.final_def, self.preds_def], axis=1)
        self.final_offense.columns.values[3] = 'predicted_points'
        self.final_defense.columns.values[3] = 'predicted_points_conceded'

        # calculate offensvie adjustment
        self.rmse_offense = np.sqrt(mean_squared_error(self.final_offense['total_effective_pts'], self.final_offense['predicted_points']))
        self.rmse_defense = np.sqrt(mean_squared_error(self.final_defense['total_effective_pts'], self.final_defense['predicted_points_conceded']))

        # correct for kicking efficiency
        self.nfl_fg_stats = self.nfl_fg_efficiency_func(eff_obj.model_data, 'posteam')
        self.nfl_kickers_2022 = self.nfl_kickers_func(eff_obj.model_data)
        self.final_off = self.final_offensive_func(self.final_offense, self.nfl_fg_stats)

        # schedule adjustment
        self.epa_offense = self.epa_offense_func(eff_obj.model_data)
        self.epa_defense = self.epa_defense_func(eff_obj.model_data)

  
    def nfl_fg_efficiency_func(self, df, pos_def):
        df2 = df[df['field_goal_result'].notna()]
        df2['exp_pts'] = df2['fg_prob'] * 3
        df2['act_pts'] = np.where(df2['field_goal_result'] == 'made', 3, 0)
        df2['added_pts'] = df2['act_pts'] - df2['exp_pts']
        out = df2.groupby(['season', pos_def]).agg(
            total_added_pts=('added_pts', 'sum'),
            total_kicks=('field_goal_result', 'count'),
            kicks_per_game=('field_goal_result', lambda x: len(x) / 17),
            avg_kick_exp_pts=('exp_pts', 'mean'),
            avg_pts_per_kick=('act_pts', 'mean'),
            add_pts_per_kick=('added_pts', 'mean'),
            add_pts_per_game=('added_pts', lambda x: sum(x) / 17)
        ).reset_index()
        return out


    def nfl_kickers_func(self, temp_df):
        nfl_kickers_2022 = temp_df[temp_df['field_goal_attempt'] == 1][['posteam', 'kicker_player_name', 'kicker_player_id']]
        nfl_kickers_2022 = nfl_kickers_2022.dropna(subset=['kicker_player_id']).drop_duplicates()
        return nfl_kickers_2022
    
    
    def final_offensive_func(self, df1, df2):
        df1['kick_pts_add'] = df1['posteam'].map(df2.set_index('posteam')['add_pts_per_game'])
        df1['adj_pts'] = df1['total_effective_pts'] + df1['kick_pts_add']
        return df1


    def epa_offense_func(self, temp_df):
        epa_off = temp_df[(~temp_df['epa'].isna()) & (~temp_df['posteam'].isna()) & (temp_df['play_type'].isin(['pass', 'run']))]
        epa_off = epa_off.groupby(['game_id', 'week', 'posteam']).agg(off_epa=('epa', 'mean'), off_plays=('play_type', 'count')).reset_index()
        epa_off['off_epa'] = epa_off.groupby('posteam')['off_epa'].transform(lambda x: x.rolling(window=14, min_periods=1).mean().shift())
        epa_off['off_plays'] = epa_off.groupby('posteam')['off_plays'].transform(lambda x: x.rolling(window=14, min_periods=1).mean().shift())
        epa_off = epa_off.sort_values('week').reset_index(drop=True)
        return epa_off
    
 
    def epa_defense_func(self, temp_df):
        epa_def = temp_df[(~temp_df['epa'].isna()) & (~temp_df['defteam'].isna()) & (temp_df['play_type'].isin(['pass', 'run']))]
        epa_def = epa_def.groupby(['game_id', 'week', 'defteam']).agg(def_epa=('epa', 'mean'), def_plays=('play_type', 'count')).reset_index()
        epa_def['def_epa'] = epa_def.groupby('defteam')['def_epa'].transform(lambda x: x.rolling(window=14, min_periods=1).mean().shift())
        epa_def['def_plays'] = epa_def.groupby('defteam')['def_plays'].transform(lambda x: x.rolling(window=14, min_periods=1).mean().shift())
        epa_def = epa_def.sort_values('week').reset_index(drop=True)
        return epa_def