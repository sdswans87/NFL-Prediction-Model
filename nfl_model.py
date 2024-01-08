# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 05:10:30 2024

@author: swan0
"""
import h2o
import nfl_data_py as nfl
import numpy as np
from sklearn.metrics import mean_squared_error
import time
import warnings
warnings.simplefilter(action='ignore', category=Warning)

import nfl_methods as nfl_obj


class import_data():
    def __init__(self):

        # start runtime
        start_run = time.time()

        # import data
        self.imp_game_data = nfl_obj.import_game_data(2005, 2024)
        self.imp_schedule_data = nfl_obj.import_schedule_data(2005, 2024)
        self.imp_rookie_data = nfl_obj.import_rookie_data(2005)
        self.imp_quarterback_data = nfl_obj.import_quarterback_data()
        
        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("import data runtime: " + str(end_run) + " minutes.")


class source_data():
    def __init__(self, imp_obj):

        # start runtime
        start_run = time.time()

        # source game datadata
        self.src_game_data = nfl_obj.source_game_data(imp_obj.imp_game_data, range(2016,2024))

        # source quarterback data
        self.src_quarterback_data = nfl_obj.source_quarterback_data(imp_obj.imp_quarterback_data)
        self.src_passing_data = nfl_obj.source_game_data(imp_obj.imp_game_data, range(2021,2024))
        self.src_offense_data = nfl_obj.source_offense_data(imp_obj.imp_game_data, range(2022,2024))
    
        # source rookie data 
        self.src_rookie_data = nfl_obj.source_rookie_data(imp_obj.imp_rookie_data)  
        self.src_rookie_pass_data = nfl_obj.source_rookie_pass_data(imp_obj.imp_game_data, self.src_rookie_data) 
        self.src_rookie_run_data = nfl_obj.source_rookie_run_data(imp_obj.imp_game_data, self.src_rookie_data) 

        # source schedule data
        self.src_schedule_data = imp_obj.imp_schedule_data[imp_obj.imp_schedule_data['season'] != 2020]
        
        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("source data runtime: " + str(end_run) + " minutes.")


class epa_data():
    def __init__(self, src_obj):

        # start runtime
        start_run = time.time()

        # epa position columns
        posteam_cols = ['passer_player_id', 'passer_player_name', 'posteam', 'qb_sum_epa']
        defteam_cols = ['passer_player_id', 'passer_player_name', 'posteam', 'defteam','qb_sum_epa']

        # epa cumulative data
        self.epa_cumulative = nfl_obj.epa_cumulative_data(src_obj.src_game_data, posteam_cols)
        self.epa_cumulative_allowed = nfl_obj.epa_cumulative_data(src_obj.src_game_data, defteam_cols)

        # epa season data 
        self.epa_season = nfl_obj.epa_season_data(src_obj.src_game_data)
        self.epa_season_allowed = nfl_obj.epa_season_data(src_obj.src_game_data)

        # epa passing data
        self.epa_pass_efficiency = nfl_obj.epa_pass_efficiency_data(src_obj.src_offense_data, 'posteam')
        self.epa_pass_efficiency_allowed = nfl_obj.epa_pass_efficiency_data(src_obj.src_offense_data, 'defteam')

        # epa rushing data
        self.epa_rush_efficiency = nfl_obj.epa_rush_efficiency_data(src_obj.src_offense_data, 'posteam')
        self.epa_rush_efficiency_allowed = nfl_obj.epa_rush_efficiency_data(src_obj.src_offense_data, 'defteam')

        # epa passing/rushing combined data
        self.epa_total_efficiency = nfl_obj.epa_total_efficiency_data(self.epa_pass_efficiency, self.epa_rush_efficiency, 'posteam')
        self.epa_total_efficiency_allowed = nfl_obj.epa_total_efficiency_data(self.epa_pass_efficiency_allowed, self.epa_rush_efficiency_allowed, 'defteam')

        # epa passing value data
        self.epa_passing_value = nfl_obj.epa_passing_value_data(src_obj.src_passing_data)

        # epa team data
        self.epa_team_qb = nfl_obj.epa_quarterback_data(self.epa_passing_value[1],  [2022,2023])

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("epa data runtime: " + str(end_run) + " minutes.")


class rookie_data():
    def __init__(self, src_obj):

        # start runtime
        start_run = time.time()

        # rookie play value
        self.rookie_play = nfl_obj.rookie_play_data(src_obj.src_rookie_pass_data, src_obj.src_rookie_run_data)

        # rookie season epa
        self.rookie_epa = nfl_obj.rookie_epa_data(src_obj.src_rookie_data, self.rookie_play)

        # rookie average epa
        self.rookie_mean = nfl_obj.rookie_mean_data(self.rookie_epa)

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("rookie data runtime: " + str(end_run) + " minutes.")
        

class quarterback_data():
    def __init__(self, src_obj, epa_obj, rk_obj):

        # start runtime
        start_run = time.time()

        # quarterback starter data
        self.qb_starters = nfl_obj.qb_starters_data(src_obj.src_game_data, season=2023, week=16)

        # quarterback efficiency rankings data
        self.qb_rankings = nfl_obj.qb_rankings_data(epa_obj.epa_passing_value)

        # quarterback starter rankings data
        self.qb_starter_rankings = nfl_obj.qb_starter_rankings_data(self.qb_starters, self.qb_rankings)

        # quarterback rankings adjusments (add next years rookie data here)
        self.qb_rankings_adj = nfl_obj.qb_rankings_adj_data(epa_obj.epa_team_qb, self.qb_starter_rankings, rk_obj.rookie_mean)

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("quarterback data runtime: " + str(end_run) + " minutes.")


class game_efficiency_data():
    def __init__(self, src_obj, epa_obj):

        # start runtime
        start_run = time.time()

        # class setup
        self.src_obj = src_obj
        self.epa_obj = epa_obj

        # game efficiency per drop back basis data
        self.game_efficiency = nfl_obj.game_efficiency_data(src_obj.src_game_data, epa_obj.epa_cumulative)
        self.game_efficiency_allowed = nfl_obj.game_efficiency_data(src_obj.src_game_data, epa_obj.epa_cumulative_allowed)

        # game efficiency per drop back basis by season data
        self.game_efficiency_by_season = nfl_obj.game_efficiency_by_season_data(src_obj.src_game_data, epa_obj.epa_season)

        # game efficiency by side of ball data
        self.game_efficiency_offense = nfl_obj.game_efficiency_ball_side_data(src_obj.src_game_data, 'posteam')
        self.game_efficiency_defense = nfl_obj.game_efficiency_ball_side_data(src_obj.src_game_data, 'defteam')

        # game efficiency passing data
        self.game_efficiency_pass = nfl_obj.game_efficiency_pass_data(src_obj.src_game_data, 'posteam')
        self.game_efficiency_pass_allowed = nfl_obj.game_efficiency_pass_data(src_obj.src_game_data, 'defteam')

        # game efficiency running data
        self.game_efficiency_run = nfl_obj.game_efficiency_run_data(src_obj.src_game_data, 'posteam')
        self.game_efficiency_run_allowed = nfl_obj.game_efficiency_run_data(src_obj.src_game_data, 'defteam')

        # game efficiency join passing + running data
        self.game_efficiency_pass_run = nfl_obj.game_efficiency_pass_run_data(self.game_efficiency_pass, self.game_efficiency_run,'posteam')
        self.game_efficiency_pass_run_allowed = nfl_obj.game_efficiency_pass_run_data(self.game_efficiency_pass_allowed,
                                                                                      self.game_efficiency_run_allowed,'defteam')

        # game efficiency points per data
        self.game_efficiency_pts_game = nfl_obj.game_efficiency_pts_game_data(src_obj.src_game_data, 'posteam')
        self.game_efficiency_pts_game_allowed = nfl_obj.game_efficiency_pts_game_data(src_obj.src_game_data, 'defteam')  

        # game efficiency combined data
        self.game_efficiency_combined = nfl_obj.game_efficiency_combined_data(self.game_efficiency_pass_run, 
                                                                              self.game_efficiency_pts_game, 'posteam')
        self.game_efficiency_combined_allowed = nfl_obj.game_efficiency_combined_data(self.game_efficiency_pass_run_allowed, 
                                                                                      self.game_efficiency_pts_game_allowed,'defteam')
        
        # game efficiency epa data
        self.game_efficiency_epa = nfl_obj.game_efficiency_epa_data(src_obj.src_game_data, self.game_efficiency_combined,'posteam')
        self.game_efficiency_epa_allowed = nfl_obj.game_efficiency_epa_data(src_obj.src_game_data, self.game_efficiency_combined_allowed,'defteam')

        # game efficiency graph epa
        self.game_efficiency_pass_epa_graph = nfl_obj.game_efficiency_pass_epa_graph_data(self.game_efficiency_epa)
        self.game_efficiency_run_epa_graph = nfl_obj.game_efficiency_run_epa_graph_data(self.game_efficiency_epa_allowed)

        # game efficiency by down 
        self.game_efficiency_by_down_data()

        # game efficiency total
        self.game_efficiency_offense_total = nfl_obj.game_efficiency_total_data(self.game_efficiency_epa, self.down_pass_off,'posteam')
        self.game_efficiency_defense_total = nfl_obj.game_efficiency_total_data(self.game_efficiency_epa_allowed, self.down_pass_def,'defteam')

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("game efficiency data runtime: " + str(end_run) + " minutes.")


    def game_efficiency_by_down_data(self):

        # game efficiency 1st down data
        self.down_1st_pass_off = nfl_obj.game_efficiency_down_data(self.src_obj.src_game_data, 1, 'pass', 'posteam')
        self.down_1st_run_off = nfl_obj.game_efficiency_down_data(self.src_obj.src_game_data, 1, 'run', 'posteam')
        self.down_1st_pass_def = nfl_obj.game_efficiency_down_data(self.src_obj.src_game_data, 1, 'pass', 'defteam')
        self.down_1st_run_def = nfl_obj.game_efficiency_down_data(self.src_obj.src_game_data, 1, 'run', 'defteam')

        # game efficiency 2nd down data
        self.down_2nd_pass_off = nfl_obj.game_efficiency_down_data(self.src_obj.src_game_data, 2, 'pass', 'posteam')
        self.down_2nd_run_off = nfl_obj.game_efficiency_down_data(self.src_obj.src_game_data, 2, 'run', 'posteam')
        self.down_2nd_pass_def = nfl_obj.game_efficiency_down_data(self.src_obj.src_game_data, 2, 'pass', 'defteam')
        self.down_2nd_run_def = nfl_obj.game_efficiency_down_data(self.src_obj.src_game_data, 2, 'run', 'defteam')

        # game efficiency 3rd down data
        self.down_3rd_pass_off = nfl_obj.game_efficiency_down_data(self.src_obj.src_game_data, 3, 'pass', 'posteam')
        self.down_3rd_run_off = nfl_obj.game_efficiency_down_data(self.src_obj.src_game_data, 3, 'run', 'posteam')
        self.down_3rd_pass_def = nfl_obj.game_efficiency_down_data(self.src_obj.src_game_data, 3, 'pass', 'defteam')
        self.down_3rd_run_def = nfl_obj.game_efficiency_down_data(self.src_obj.src_game_data, 3, 'run', 'defteam')

        # game efficiency all down data
        self.down_pass_off = nfl_obj.game_efficiency_all_down_data(self.down_1st_pass_off, self.down_1st_run_off, 
                                                                   self.down_2nd_pass_off, self.down_2nd_run_off,
                                                                   self.down_3rd_pass_off, self.down_3rd_run_off, "posteam")
        self.down_pass_def = nfl_obj.game_efficiency_all_down_data(self.down_1st_pass_def, self.down_1st_run_def, 
                                                                   self.down_2nd_pass_def, self.down_2nd_run_def,
                                                                   self.down_3rd_pass_def, self.down_3rd_run_def, "defteam")
        

class model_efficiency_data():
    def __init__(self, src_obj):

        # start runtime
        start_run = time.time()

        # class setup
        self.src_obj = src_obj

        # model efficiency pass epa
        self.model_efficiency_pass = nfl_obj.game_efficiency_pass_data(src_obj.src_offense_data, 'posteam')
        self.model_efficiency_pass_allowed = nfl_obj.game_efficiency_pass_data(src_obj.src_offense_data, 'defteam')

        # model efficiency running data
        self.model_efficiency_run = nfl_obj.game_efficiency_run_data(src_obj.src_offense_data, 'posteam')
        self.model_efficiency_run_allowed = nfl_obj.game_efficiency_run_data(src_obj.src_offense_data, 'defteam')

        # model efficiency join passing + running data
        self.model_efficiency_pass_run = nfl_obj.game_efficiency_pass_run_data(self.model_efficiency_pass, 
                                                                               self.model_efficiency_run, 'posteam')
        self.model_efficiency_pass_run_allowed = nfl_obj.game_efficiency_pass_run_data(self.model_efficiency_pass_allowed,
                                                                                       self.model_efficiency_run_allowed,'defteam')
        
        # model efficiency points per game data
        self.model_efficiency_pts_game = nfl_obj.game_efficiency_pts_game_data(src_obj.src_offense_data, 'posteam')
        self.model_efficiency_pts_game_allowed = nfl_obj.game_efficiency_pts_game_data(src_obj.src_offense_data, 'defteam') 

        # model efficiency combined data
        self.model_efficiency_combined = nfl_obj.game_efficiency_combined_data(self.model_efficiency_pass_run, 
                                                                               self.model_efficiency_pts_game, 'posteam')
        self.model_efficiency_combined_allowed = nfl_obj.game_efficiency_combined_data(self.model_efficiency_pass_run_allowed, 
                                                                                       self.model_efficiency_pts_game_allowed,'defteam')
        
        # model efficiency epa data
        self.model_efficiency_epa = nfl_obj.game_efficiency_epa_data(src_obj.src_offense_data, self.model_efficiency_combined,'posteam')
        self.model_efficiency_epa_allowed = nfl_obj.game_efficiency_epa_data(src_obj.src_offense_data, self.model_efficiency_combined_allowed,'defteam')
        
        # model efficiency by down 
        self.model_efficiency_by_down_data()
        
        # model efficiency total
        self.model_efficiency_offense_total = nfl_obj.game_efficiency_total_data(self.model_efficiency_epa, self.down_pass_off,'posteam')
        self.model_efficiency_defense_total = nfl_obj.game_efficiency_total_data(self.model_efficiency_epa_allowed, self.down_pass_def,'defteam')

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("model efficiency data runtime: " + str(end_run) + " minutes.")


    def model_efficiency_by_down_data(self):

        # model efficiency 1st down data
        self.down_1st_pass_off = nfl_obj.game_efficiency_down_data(self.src_obj.src_offense_data, 1, 'pass', 'posteam')
        self.down_1st_run_off = nfl_obj.game_efficiency_down_data(self.src_obj.src_offense_data, 1, 'run', 'posteam')
        self.down_1st_pass_def = nfl_obj.game_efficiency_down_data(self.src_obj.src_offense_data, 1, 'pass', 'defteam')
        self.down_1st_run_def = nfl_obj.game_efficiency_down_data(self.src_obj.src_offense_data, 1, 'run', 'defteam')

        # model efficiency 2nd down data
        self.down_2nd_pass_off = nfl_obj.game_efficiency_down_data(self.src_obj.src_offense_data, 2, 'pass', 'posteam')
        self.down_2nd_run_off = nfl_obj.game_efficiency_down_data(self.src_obj.src_offense_data, 2, 'run', 'posteam')
        self.down_2nd_pass_def = nfl_obj.game_efficiency_down_data(self.src_obj.src_offense_data, 2, 'pass', 'defteam')
        self.down_2nd_run_def = nfl_obj.game_efficiency_down_data(self.src_obj.src_offense_data, 2, 'run', 'defteam')

        # model efficiency 3rd down data
        self.down_3rd_pass_off = nfl_obj.game_efficiency_down_data(self.src_obj.src_offense_data, 3, 'pass', 'posteam')
        self.down_3rd_run_off = nfl_obj.game_efficiency_down_data(self.src_obj.src_offense_data, 3, 'run', 'posteam')
        self.down_3rd_pass_def = nfl_obj.game_efficiency_down_data(self.src_obj.src_offense_data, 3, 'pass', 'defteam')
        self.down_3rd_run_def = nfl_obj.game_efficiency_down_data(self.src_obj.src_offense_data, 3, 'run', 'defteam')

        # model efficiency all down data
        self.down_pass_off = nfl_obj.game_efficiency_all_down_data(self.down_1st_pass_off, self.down_1st_run_off, 
                                                                   self.down_2nd_pass_off, self.down_2nd_run_off,
                                                                   self.down_3rd_pass_off, self.down_3rd_run_off, "posteam")
        self.down_pass_def = nfl_obj.game_efficiency_all_down_data(self.down_1st_pass_def, self.down_1st_run_def, 
                                                                   self.down_2nd_pass_def, self.down_2nd_run_def,
                                                                   self.down_3rd_pass_def, self.down_3rd_run_def, "defteam")
        

class ensemble_model_actual_points():
    def __init__(self, gm_obj):

        # start runtime
        start_run = time.time()

        # ensemble column lists
        self.ensemble_off_columns = ['game_id','year', 'week', 'away_team', 'home_team','posteam',
                                     'n_pass','n_rush','n_plays', 'home_final','away_final',
                                     'total_tds','total_fgs_att','total_pat','total_effective_pts',
                                     'total_score','pt_diff']
        self.ensemble_def_columns = ['game_id','year', 'week', 'away_team', 'home_team','defteam',
                                     'n_pass','n_rush','n_plays', 'home_final','away_final',
                                     'total_tds','total_fgs_att','total_pat','total_effective_pts',
                                     'total_score','pt_diff']
        
        # ensemble data prep for model input
        self.ensemble_offense = nfl_obj.ensemble_prep_data(gm_obj.game_efficiency_offense_total, self.ensemble_off_columns)
        self.ensemble_defense = nfl_obj.ensemble_prep_data(gm_obj.game_efficiency_defense_total, self.ensemble_def_columns)

        # ensemble offensive model on actual points 
        h2o.init()
        self.ensemble_offense_train = nfl_obj.ensemble_model_data(self.ensemble_offense, 0)
        self.ensemble_offensive_test = nfl_obj.ensemble_model_data(self.ensemble_offense, 1)

        # ensemble defensive model on actual points 
        self.ensemble_defense_train = nfl_obj.ensemble_model_data(self.ensemble_defense, 0)
        self.ensemble_defensive_test = nfl_obj.ensemble_model_data(self.ensemble_defense, 1)

        # ensemble train gradient boosting
        self.ensemble_offense_gbm = nfl_obj.ensemble_model_train_gbm(self.ensemble_offense_train, 'poss_score', 
                                                                     set(self.ensemble_offense_train.columns) - 
                                                                     set(['poss_score']), 5, 5)
        self.ensemble_defense_gbm = nfl_obj.ensemble_model_train_gbm(self.ensemble_defense_train, 'score_allowed', 
                                                                     set(self.ensemble_defense_train.columns) - 
                                                                     set(['score_allowed']), 5, 5)
        
        # builds a distributed random forest on a parsed dataset, for regression or classification
        # ensemble train random forest
        self.ensemble_offense_rf = nfl_obj.ensemble_model_train_rf(self.ensemble_offense_train, 'poss_score', 
                                                                   set(self.ensemble_offense_train.columns) - 
                                                                   set(['poss_score']), 5, 5)
        self.ensemble_defense_rf = nfl_obj.ensemble_model_train_rf(self.ensemble_defense_train, 'score_allowed', 
                                                                   set(self.ensemble_defense_train.columns) - 
                                                                   set(['score_allowed']), 5, 5)
        
        # fits a generalized linear model, specified by a response variable, a set of predictors, and a
        # description of the error distribution.
        # ensemble train linear regression
        self.ensemble_offense_lr = nfl_obj.ensemble_model_train_lr(self.ensemble_offense_train, 'poss_score', 
                                                                   set(self.ensemble_offense_train.columns) - 
                                                                   set(['poss_score']), 5, 5)
        self.ensemble_defense_lr = nfl_obj.ensemble_model_train_lr(self.ensemble_defense_train, 'score_allowed', 
                                                                   set(self.ensemble_defense_train.columns) -
                                                                   set(['score_allowed']), 5, 5)

        # build a deep neural network model using cpus
        # builds a feed-forward multilayer artificial neural network on an h2o frame
        # ensemble train nearual net
        self.ensemble_offense_nn = nfl_obj.ensemble_model_train_nn(self.ensemble_offense_train, 'poss_score', 
                                                                   set(self.ensemble_offense_train.columns) - 
                                                                   set(['poss_score']), 5, 5)
        self.ensemble_defense_nn = nfl_obj.ensemble_model_train_nn(self.ensemble_defense_train, 'score_allowed', 
                                                                   set(self.ensemble_defense_train.columns) - 
                                                                   set(['score_allowed']), 5, 5)
        
        # builds a stacked ensemble machine learning method that uses two
        # or more H2O learning algorithms to improve predictive performance - it is a loss-based
        # supervised learning method that finds the optimal combination of a collection of prediction
        # algorithms - this method supports regression and binary classification
        # ensemble train stacked estimator 
        self.ensemble_offense_stacked = nfl_obj.ensemble_model_stacked_estimator(self.ensemble_offense_lr, self.ensemble_offense_rf, 
                                                                                 self.ensemble_offense_nn, self.ensemble_offense_gbm, 
                                                                                 self.ensemble_offense_train, 'poss_score', 
                                                                                 set(self.ensemble_offense_train.columns) - 
                                                                                 set(['poss_score']))
        self.ensemble_defense_stacked = nfl_obj.ensemble_model_stacked_estimator(self.ensemble_defense_lr, self.ensemble_defense_rf, 
                                                                                 self.ensemble_defense_nn, self.ensemble_defense_gbm, 
                                                                                 self.ensemble_defense_train, 'score_allowed', 
                                                                                 set(self.ensemble_defense_train.columns) - 
                                                                                 set(['score_allowed']))

        # ensemble gbm performance
        self.ensemble_off_gbm_test = nfl_obj.ensemble_model_performance(self.ensemble_offense_gbm)
        self.ensemble_def_gbm_test = nfl_obj.ensemble_model_performance(self.ensemble_defense_gbm)

        # ensemble random forest performance
        self.ensemble_off_rf_test = nfl_obj.ensemble_model_performance(self.ensemble_offense_rf)
        self.ensemble_def_rf_test = nfl_obj.ensemble_model_performance(self.ensemble_defense_rf)

        # ensemble linear regression performance
        self.ensemble_off_lr_test = nfl_obj.ensemble_model_performance(self.ensemble_offense_lr)
        self.ensemble_def_lr_test = nfl_obj.ensemble_model_performance(self.ensemble_defense_lr)

        # ensemble neural net performance
        self.ensemble_off_nn_test = nfl_obj.ensemble_model_performance(self.ensemble_offense_nn)
        self.ensemble_def_nn_test = nfl_obj.ensemble_model_performance(self.ensemble_defense_nn)

        # ensemble stacked performance
        self.ensemble_off_stacked_test = nfl_obj.ensemble_model_performance(self.ensemble_offense_stacked)
        self.ensemble_def_stacked_test = nfl_obj.ensemble_model_performance(self.ensemble_defense_stacked)
        self.ensemble_off_stacked_test.rmse()
        self.ensemble_def_stacked_test.rmse()

        # ensemble performance mins
        print(min([self.ensemble_off_gbm_test.rmse(), self.ensemble_off_rf_test.rmse(), self.ensemble_off_lr_test.rmse(), 
                   self.ensemble_off_nn_test.rmse()]))
        print(min([self.ensemble_def_gbm_test.rmse(), self.ensemble_def_rf_test.rmse(), self.ensemble_def_lr_test.rmse(),
                   self.ensemble_def_nn_test.rmse()]))

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("ensemble model on actual points runtime: " + str(end_run) + " minutes.")


# ensemble model on effective points with smaller sample size and recalc models 
# integrate adjusted fg efficiency
# integrate schedule difficulty & home field advantage
# create weekly predictor engine & output odds
class ensemble_model_effective_points():
    def __init__(self, src_obj, qb_obj, gm_obj, mod_obj, act_obj):

        # start runtime
        start_run = time.time()

        # ensemble column list 
        self.ensemble_off_columns = ['game_id','year', 'week', 'away_team', 'home_team','posteam',
                                     'n_pass','n_rush','n_plays', 'home_final','away_final',
                                     'total_tds','total_fgs_att','total_pat','total_score',
                                     'pt_diff']
        self.ensemble_def_columns = ['game_id','year', 'week', 'away_team', 'home_team','defteam',
                                     'n_pass','n_rush','n_plays', 'home_final','away_final',
                                     'total_tds','total_fgs_att','total_pat','total_score',
                                     'pt_diff']
        
        # ensemble data prep for model input
        self.ensemble_offense = nfl_obj.ensemble_prep_data(gm_obj.game_efficiency_offense_total, self.ensemble_off_columns)
        self.ensemble_defense = nfl_obj.ensemble_prep_data(gm_obj.game_efficiency_offense_total, self.ensemble_off_columns)

        # ensemble offensive model on total effective points 
        h2o.init()
        self.ensemble_offense_train = nfl_obj.ensemble_model_data(self.ensemble_offense, 0)
        self.ensemble_offensive_test = nfl_obj.ensemble_model_data(self.ensemble_offense, 1)

        # ensemble defensive model on total effective points  
        self.ensemble_defense_train = nfl_obj.ensemble_model_data(self.ensemble_defense, 0)
        self.ensemble_defensive_test = nfl_obj.ensemble_model_data(self.ensemble_defense, 1)

        # ensemble train gradient boosting
        self.ensemble_offense_gbm = nfl_obj.ensemble_model_train_gbm(self.ensemble_offense_train, 'total_effective_pts', 
                                                                     set(self.ensemble_offense_train.columns) - 
                                                                     set(['total_effective_pts']), 5, 5)
        self.ensemble_defense_gbm = nfl_obj.ensemble_model_train_gbm(self.ensemble_defense_train, 'total_effective_pts', 
                                                                     set(self.ensemble_defense_train.columns) - 
                                                                     set(['total_effective_pts']), 5, 5)
        
        # builds a distributed random forest on a parsed dataset, for regression or classification
        # ensemble train random forest
        self.ensemble_offense_rf = nfl_obj.ensemble_model_train_rf(self.ensemble_offense_train, 'total_effective_pts', 
                                                                   set(self.ensemble_offense_train.columns) - 
                                                                   set(['total_effective_pts']), 5, 5)
        self.ensemble_defense_rf = nfl_obj.ensemble_model_train_rf(self.ensemble_defense_train, 'total_effective_pts', 
                                                                   set(self.ensemble_defense_train.columns) - 
                                                                   set(['total_effective_pts']), 5, 5)
        
        # fits a generalized linear model, specified by a response variable, a set of predictors, and a
        # description of the error distribution.
        # ensemble train linear regression
        self.ensemble_offense_lr = nfl_obj.ensemble_model_train_lr(self.ensemble_offense_train, 'total_effective_pts', 
                                                                   set(self.ensemble_offense_train.columns) - 
                                                                   set(['total_effective_pts']), 5, 5)
        self.ensemble_defense_lr = nfl_obj.ensemble_model_train_lr(self.ensemble_defense_train, 'total_effective_pts', 
                                                                   set(self.ensemble_defense_train.columns) -
                                                                   set(['total_effective_pts']), 5, 5)

        # build a deep neural network model using cpus
        # builds a feed-forward multilayer artificial neural network on an h2o frame
        # ensemble train nearual net
        self.ensemble_offense_nn = nfl_obj.ensemble_model_train_nn(self.ensemble_offense_train, 'total_effective_pts', 
                                                                   set(self.ensemble_offense_train.columns) - 
                                                                   set(['total_effective_pts']), 5, 5)
        self.ensemble_defense_nn = nfl_obj.ensemble_model_train_nn(self.ensemble_defense_train, 'total_effective_pts', 
                                                                   set(self.ensemble_defense_train.columns) - 
                                                                   set(['total_effective_pts']), 5, 5)
        
        # builds a stacked ensemble machine learning method that uses two
        # or more H2O learning algorithms to improve predictive performance - it is a loss-based
        # supervised learning method that finds the optimal combination of a collection of prediction
        # algorithms - this method supports regression and binary classification
        # ensemble train stacked estimator 
        self.ensemble_offense_stacked = nfl_obj.ensemble_model_stacked_estimator(self.ensemble_offense_lr, self.ensemble_offense_rf, 
                                                                                 self.ensemble_offense_nn, self.ensemble_offense_gbm, 
                                                                                 self.ensemble_offense_train, 'total_effective_pts', 
                                                                                 set(self.ensemble_offense_train.columns) - 
                                                                                 set(['total_effective_pts']))
        self.ensemble_defense_stacked = nfl_obj.ensemble_model_stacked_estimator(self.ensemble_defense_lr, self.ensemble_defense_rf, 
                                                                                 self.ensemble_defense_nn, self.ensemble_defense_gbm, 
                                                                                 self.ensemble_defense_train, 'total_effective_pts', 
                                                                                 set(self.ensemble_defense_train.columns) - 
                                                                                 set(['total_effective_pts']))

        # ensemble gbm performance
        self.ensemble_off_gbm_test = nfl_obj.ensemble_model_performance(self.ensemble_offense_gbm)
        self.ensemble_def_gbm_test = nfl_obj.ensemble_model_performance(self.ensemble_defense_gbm)

        # ensemble random forest performance
        self.ensemble_off_rf_test = nfl_obj.ensemble_model_performance(self.ensemble_offense_rf)
        self.ensemble_def_rf_test = nfl_obj.ensemble_model_performance(self.ensemble_defense_rf)

        # ensemble linear regression performance
        self.ensemble_off_lr_test = nfl_obj.ensemble_model_performance(self.ensemble_offense_lr)
        self.ensemble_def_lr_test = nfl_obj.ensemble_model_performance(self.ensemble_defense_lr)

        # ensemble neural net performance
        self.ensemble_off_nn_test = nfl_obj.ensemble_model_performance(self.ensemble_offense_nn)
        self.ensemble_def_nn_test = nfl_obj.ensemble_model_performance(self.ensemble_defense_nn)

        # ensemble stacked performance
        self.ensemble_off_stacked_test = nfl_obj.ensemble_model_performance(self.ensemble_offense_stacked)
        self.ensemble_def_stacked_test = nfl_obj.ensemble_model_performance(self.ensemble_defense_stacked)
        self.ensemble_off_stacked_test.rmse()
        self.ensemble_def_stacked_test.rmse()

        # ensemble performance mins
        print(min([self.ensemble_off_gbm_test.rmse(), self.ensemble_off_rf_test.rmse(), self.ensemble_off_lr_test.rmse(), 
                   self.ensemble_off_nn_test.rmse()]))
        print(min([self.ensemble_def_gbm_test.rmse(), self.ensemble_def_rf_test.rmse(), self.ensemble_def_lr_test.rmse(),
                   self.ensemble_def_nn_test.rmse()]))
        
        # ensemble final offense/defense efficiency on data with shorter sample size
        self.ensemble_final_offense = mod_obj.model_efficiency_offense_total[["game_id", "posteam", "total_effective_pts"]]
        self.ensemble_final_defense = mod_obj.model_efficiency_defense_total[["game_id", "defteam", "total_effective_pts"]]

        # ensemble final inputs for model recalc
        self.ensemble_offense_efficiency = nfl_obj.ensemble_final_efficiency_data(mod_obj.model_efficiency_offense_total, 
                                                                                  act_obj.ensemble_off_columns + ["poss_score"])
        self.ensemble_defense_efficiency = nfl_obj.ensemble_final_efficiency_data(mod_obj.model_efficiency_defense_total, 
                                                                                  act_obj.ensemble_def_columns + ["score_allowed"])
        
        # ensemble targets and predictors
        self.ensemble_offensive_predictors = nfl_obj.ensemble_predictors_data(self.ensemble_offense_stacked, self.ensemble_offense_efficiency)
        self.ensemble_defensive_predictors = nfl_obj.ensemble_predictors_data(self.ensemble_defense_stacked, self.ensemble_defense_efficiency)

        # ensemble final epa with predictors
        self.ensemble_offense_epa = nfl_obj.ensemble_epa_data(self.ensemble_final_offense, self.ensemble_offensive_predictors, "predicted_points")
        self.ensemble_defense_epa = nfl_obj.ensemble_epa_data(self.ensemble_final_defense, self.ensemble_defensive_predictors, "predicted_points_conceded")
        print(np.sqrt(mean_squared_error(self.ensemble_offense_epa['total_effective_pts'], self.ensemble_offense_epa['predicted_points'])))
        print(np.sqrt(mean_squared_error(self.ensemble_defense_epa['total_effective_pts'], self.ensemble_defense_epa['predicted_points_conceded'])))

        # ensemble kicking efficiency adjustment 
        self.ensemble_kicker_data = nfl_obj.ensemble_starting_kicker_data(src_obj.src_offense_data)
        self.ensemble_kicking_efficiency = nfl_obj.ensemble_kicking_efficiency_data(src_obj.src_offense_data, 'posteam')
        self.ensemble_offense_epa_adj = nfl_obj.ensemble_adjusted_points_data(self.ensemble_offense_epa, self.ensemble_kicking_efficiency)

        # ensemble schedule adjustment
        self.ensemble_off_sched_adj = nfl_obj.ensemble_schedule_off_adjustment(src_obj.src_offense_data)
        self.ensemble_def_sched_adj = nfl_obj.ensemble_schedule_def_adjustment(src_obj.src_offense_data)

        # ensemble epa league averages
        self.ensemble_epa_lg_avg = nfl_obj.ensemble_epa_lg_avg_data(src_obj.src_offense_data)
        self.ensemble_epa_off_avg = nfl_obj.ensemble_epa_off_avg_data(self.ensemble_off_sched_adj, self.ensemble_epa_lg_avg)
        self.ensemble_epa_def_avg = nfl_obj.ensemble_epa_def_avg_data(self.ensemble_def_sched_adj, self.ensemble_epa_lg_avg)

        # ensemble schedule difficulty
        self.ensemble_sched_diff_off = nfl_obj.ensemble_sched_diff_off_data(self.ensemble_offense_epa_adj, self.ensemble_epa_off_avg)
        self.ensemble_sched_diff_def = nfl_obj.ensemble_sched_diff_def_data(self.ensemble_defense_epa, self.ensemble_epa_def_avg)

        # ensemble weighted averages
        self.ensemble_weighted_offense = nfl_obj.ensemble_weighted_offense_data(self.ensemble_offense_epa_adj, self.ensemble_sched_diff_def)
        self.ensemble_weighted_defense = nfl_obj.ensemble_weighted_defense_data(self.ensemble_defense_epa, self.ensemble_sched_diff_off)

        # ensemble team qb performance
        self.ensemble_qb_update = nfl_obj.ensemble_qb_update_data(qb_obj.qb_rankings_adj, self.ensemble_weighted_offense)

        # ensemble team power rankings
        self.ensemble_power_rankings = nfl_obj.ensemble_power_rankings_data(self.ensemble_qb_update, self.ensemble_weighted_offense, 
                                                                            self.ensemble_weighted_defense)
        
        # ensemble home field advantage 
        self.ensemble_hfa = nfl_obj.ensemble_hfa_data(src_obj.src_schedule_data)

        # ensemble current schedule
        self.ensemble_curr_schedule = nfl_obj.ensemble_curr_schedule_data(src_obj.src_schedule_data)

        # ensemble prediction engine 
        self.predict_wk_16 = nfl_obj.predict_odds_engine(self.ensemble_curr_schedule, 16, self.ensemble_hfa, 
                                                         self.ensemble_power_rankings, self.ensemble_qb_update)
        self.predict_wk_17 = nfl_obj.predict_odds_engine(self.ensemble_curr_schedule, 17, self.ensemble_hfa, 
                                                         self.ensemble_power_rankings, self.ensemble_qb_update)
        self.predict_wk_18 = nfl_obj.predict_odds_engine(self.ensemble_curr_schedule, 18, self.ensemble_hfa, 
                                                         self.ensemble_power_rankings, self.ensemble_qb_update)

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("ensemble model on total effective points runtime: " + str(end_run) + " minutes.")


start_run = time.time()
imp_obj = import_data()
src_obj = source_data(imp_obj)
epa_obj = epa_data(src_obj)
rok_obj = rookie_data(src_obj)
qbk_obj = quarterback_data(src_obj, epa_obj, rok_obj)
gme_obj = game_efficiency_data(src_obj, epa_obj)
mod_obj = model_efficiency_data(src_obj)
act_obj = ensemble_model_actual_points(gme_obj)
eff_obj = ensemble_model_effective_points(src_obj, qbk_obj, gme_obj, mod_obj, act_obj)
end_run = time.time()
end_run = (end_run - start_run)/60
print("nfl prediction model on total runtime: " + str(end_run) + " minutes.")