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
import seaborn as sns
from sklearn.metrics import mean_squared_error
import time


class model_data_class():
    def __init__(self, eff_obj):
        
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
        self.final_offense = self.final_offensive_func(self.final_offense, self.nfl_fg_stats)

        # schedule adjustment
        self.epa_offense = self.epa_offense_func(eff_obj.model_data)
        self.epa_defense = self.epa_defense_func(eff_obj.model_data)

        # calculate league average epa
        self.epa_league_avg = self.epa_league_avg_func(eff_obj.model_data)
        self.epa_offense_avg = self.epa_offense_avg_func(self.epa_offense, self.epa_league_avg)
        self.epa_defense_avg = self.epa_defense_avg_func(self.epa_defense, self.epa_league_avg)

        # calculate schedule difficulty
        self.schedule_adj_off = self.offensive_schedule_diff_func(self.final_offense, self.epa_offense_avg)
        self.schedule_adj_def = self.defensive_schedule_diff_func(self.final_defense, self.epa_defense_avg)

        # group by teams
        self.final_offense[['year','week','home','away']] = self.final_offense['game_id'].str.split('_',expand=True)
        self.final_offense = self.final_offense.drop(columns=['year', 'home', 'away'])
        self.final_defense[['year','week','home','away']] = self.final_defense['game_id'].str.split('_',expand=True)
        self.final_defense = self.final_defense.drop(columns=['year', 'home', 'away'])

        # calculate weighted average
        self.weighted_offense = self.weighted_avg_off_func(self.final_offense, self.schedule_adj_def)
        self.weighted_defense = self.weighted_avg_def_func(self.final_defense, self.schedule_adj_off)
        
        # adjust team QB performance from above vs defenses they played
        self.qb_update_2023 = self.qb_performance_func(epa_obj.qb_rankings_adj, self.weighted_offense)

        # team power rankings
        self.power_rankings = self.power_rankings_fun(self.qb_update_2023, self.weighted_offense, self.weighted_defense)
        
        # home field advantage integration
        self.hfa_adj = self.home_field_advantage_func(import_obj.schedule_data)
        
        # predict weekly odds
        self.sched_2023 = self.schedule_func(import_obj.schedule_data)
        self.week_sixteen = self.weekly_odds_engine(self.sched_2023, 16, self.hfa_adj, self.power_rankings, self.qb_update_2023)
        self.week_seventeen = self.weekly_odds_engine(self.sched_2023, 17, self.hfa_adj, self.power_rankings, self.qb_update_2023)
  

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
    

    def epa_league_avg_func(self, epa_data):
        epa_lg_avg = epa_data.groupby('week').agg(league_mean=('epa', 'mean')).reset_index()
        epa_lg_avg['league_mean'] = epa_lg_avg['league_mean'].rolling(window=14, min_periods=1).mean().shift()
        return epa_lg_avg
    

    def epa_offense_avg_func(self, off_epa, league_epa):
        epa_off = pd.merge(off_epa, league_epa, on='week', how='left').tail(32)
        epa_off['adj_off_epa'] = epa_off['off_epa'] - epa_off['league_mean']
        epa_off['adj_off_pts'] = epa_off['adj_off_epa'] * epa_off['off_plays']
        epa_off = epa_off[['posteam', 'adj_off_epa', 'adj_off_pts']]
        epa_off.columns = ['team', 'adj_off_epa', 'adj_off_pts']
        return epa_off
    
    
    def epa_defense_avg_func(self, def_epa, league_epa):
        epa_def = pd.merge(def_epa, league_epa, on='week', how='left').tail(32)
        epa_def['adj_def_epa'] = epa_def['def_epa'] - epa_def['league_mean']
        epa_def['adj_def_pts'] = epa_def['adj_def_epa'] * epa_def['def_plays']
        epa_def = epa_def[['defteam', 'adj_def_epa', 'adj_def_pts']]
        epa_def.columns = ['team', 'adj_def_epa', 'adj_def_pts']
        return epa_def
    

    def offensive_schedule_diff_func(self, off_data, epa_data):
        adj_off = off_data[['posteam', 'game_id']].copy()
        adj_off[['year', 'week', 'away', 'home']] = adj_off['game_id'].str.split('_', expand=True)
        adj_off['opposition'] = np.where(adj_off['posteam'] == adj_off['away'], adj_off['home'], adj_off['away'])
        adj_off = adj_off.drop(columns=['year', 'week', 'away', 'home'])
        adj_off['adjust'] = adj_off['opposition'].map(epa_data.set_index('team')['adj_off_pts'])
        adj_off = adj_off.rename(columns={'posteam': 'team'})
        adj_off = adj_off.groupby('team').agg(off_vs_avg=('adjust', 'mean'))
        return adj_off
    
    
    def defensive_schedule_diff_func(self, def_data, epa_data):
        adj_def = def_data[['defteam', 'game_id']].copy()
        adj_def[['year', 'week', 'away', 'home']] = adj_def['game_id'].str.split('_', expand=True)
        adj_def['opposition'] = np.where(adj_def['defteam'] == adj_def['away'], adj_def['home'], adj_def['away'])
        adj_def = adj_def.drop(columns=['year', 'week', 'away', 'home'])
        adj_def['adjust'] = adj_def['opposition'].map(epa_data.set_index('team')['adj_def_pts'])
        adj_def = adj_def.rename(columns={'defteam': 'team'})
        adj_def = adj_def.groupby('team').agg(def_vs_avg=('adjust', 'mean'))
        return adj_def
    

    def weighted_avg_off_func(self, final_off, final_def_adj):
        final_def_adj = final_def_adj.reset_index()
        weighted_offense = final_off.groupby('posteam').apply(lambda x: x.rolling(window=10, on='predicted_points', win_type='boxcar').mean().iloc[-1])
        weighted_offense = weighted_offense.reset_index()[['posteam', 'predicted_points']]
        weighted_offense['sched_adj'] = weighted_offense['posteam'].map(final_def_adj.set_index('team')['def_vs_avg'])
        weighted_offense['adj_off'] = weighted_offense['predicted_points'] - weighted_offense['sched_adj']
        return weighted_offense
        

    def weighted_avg_def_func(self, final_def, final_off_adj):
        final_off_adj = final_off_adj.reset_index()
        weighted_defense = final_def.groupby('defteam').apply(lambda x: x.rolling(window=10, on='predicted_points_conceded', win_type='boxcar').mean().iloc[-1])
        weighted_defense = weighted_defense.reset_index()[['defteam', 'predicted_points_conceded']]
        weighted_defense['sched_adj'] = weighted_defense['defteam'].map(final_off_adj.set_index('team')['off_vs_avg'])
        weighted_defense['adj_def'] = weighted_defense['predicted_points_conceded'] - weighted_defense['sched_adj']
        return weighted_defense
    

    def qb_performance_func(self, qb_update_2023, weighted_offense):
        adj_qb_remaining = sum(range(1, 18))
        adj_qb_gone = sum(range(9, 18))
        adj_qb_ratio = (adj_qb_remaining - adj_qb_gone) / adj_qb_remaining
        qb_update_2023['sched_adj'] = qb_update_2023['team'].map(weighted_offense.set_index('posteam')['sched_adj'])
        qb_update_2023['offense_update'] = qb_update_2023.apply(lambda row: float(row['wt_avg'] * adj_qb_ratio) 
                                                                - (float(row['wt_avg_team']) - (row['sched_adj'] * adj_qb_ratio)), axis=1)
        qb_no_change = ['BUF', 'SF', 'NE', 'CIN', 'DAL', 'DET', 'JAX', 'KC', 'LAC', 'MIN',
                        'SEA', 'NYJ','CAR', 'ARI', 'PHI', 'WAS', 'GB', 'ATL', 'NO', 'LV', 
                        'CHI', 'CLE', 'HOU', 'LA', 'MIA', 'TEN', 'NYG', 'PIT', 'TB', 'DEN']
        qb_update_2023['offense_update'] = qb_update_2023.apply(lambda row: 0 if row['team'] in qb_no_change else row['offense_update'], axis=1)
        return qb_update_2023
    

    def power_rankings_fun(self, qb_update_2023, weighted_offense, weighted_defense):
        qb_update_2023 = qb_update_2023.drop_duplicates(subset=['team'])
        power_rank = weighted_offense[['posteam']].copy()
        power_rank.columns = ['team']
        power_rank['adj_off'] = power_rank['team'].map(weighted_offense.set_index('posteam')['adj_off'])
        power_rank['adj_def'] = power_rank['team'].map(weighted_defense.set_index('defteam')['adj_def'])
        power_rank['pts_vs_avg'] = power_rank['adj_off'] - power_rank['adj_def']
        power_rank['qb_adj'] = power_rank['team'].map(qb_update_2023.set_index('team')['offense_update'])
        power_rank['final_ranking'] = power_rank['pts_vs_avg'] + power_rank['qb_adj']
        power_rank.to_csv('power_rank.csv', index=False)
        return power_rank
    
    
    def home_field_advantage_func(self,  nfl_schedules):
        nfl_hfa = nfl_schedules[nfl_schedules['season'] != 2020]
        nfl_hfa = nfl_hfa[nfl_hfa['season'] != 2023]
        nfl_hfa = nfl_hfa[nfl_hfa['game_type'] == 'REG']
        nfl_hfa = nfl_hfa.drop(['away_moneyline', 'home_moneyline', 'away_spread_odds', 
                                'home_spread_odds', 'old_game_id', 'gsis', 'nfl_detail_id', 
                                'pfr', 'pff', 'espn', 'over_odds', 'under_odds'], axis=1)
        home_spread = nfl_hfa.groupby('season').agg(home_line=('spread_line', 'mean'), result_avg=('result', 'mean')).reset_index()
        line_spread = sns.lmplot(data=home_spread, y='home_line', x='season', ci=None)
        line_spread.set_axis_labels('Season', 'Home Line')
        line_result = sns.lmplot(data=home_spread, y='result_avg', x='season', ci=None)
        line_result.set_axis_labels('Season', 'Result Average')
        hfa_adj = home_spread['home_line'].tail(1).values[0]
        return hfa_adj
 

    def schedule_func(self, nfl_schedules):
        sched_2023 = nfl_schedules[nfl_schedules['season'] == 2023]
        return sched_2023
    
    
    def weekly_odds_engine(self, df, wk, hfa, power_rank, qb_update_2023):
        qb_update_2023 = qb_update_2023.drop_duplicates(subset=['team'])
        df = df[df['week'] == wk]
        df = df[['game_id', 'gameday', 'weekday', 'home_team', 'away_team', 'away_rest', 'home_rest', 'spread_line', 'total_line']]
        df['home_ranking'] = df['home_team'].map(power_rank.set_index('team')['final_ranking'])
        df['away_ranking'] = df['away_team'].map(power_rank.set_index('team')['final_ranking'])
        df['starting_home_qb'] = df['home_team'].map(qb_update_2023.set_index('team')['passer_player_name'])
        df['starting_away_qb'] = df['away_team'].map(qb_update_2023.set_index('team')['passer_player_name'])
        df['unregressed'] = df['home_ranking'] - df['away_ranking'] + hfa
        df['regressed_number'] = 0.5 * df['unregressed'] + 0.5 * df['spread_line']
        df.to_csv(rf'week_{wk}.csv', index=False)
        return df