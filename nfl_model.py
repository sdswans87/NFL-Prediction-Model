# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:08:57 2023

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
import random
import seaborn as sns
import time
import warnings
warnings.simplefilter(action='ignore', category=Warning)

class import_data_class():
    def __init__(self):

        # start runtime
        start_run = time.time()

        # import nfl data past 20 years
        self.nfl_df = self.import_games_func()

        # import quarterback data for 2023 
        self.quarterbacks_df = self.import_qb_func()

        # import rookie data
        self.rookies_df = self.import_rookies_func()

        # import nfl schedule data
        self.schedules_df = self.import_schedules_func()

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("import data runtime: " + str(end_run) + " minutes.")

    def import_games_func(self):
        df = nfl.import_pbp_data(range(2003,2023))
        return df
    
    def import_qb_func(self):
        df = nfl.import_players()
        return df

    def import_rookies_func(self):
        df = nfl.import_draft_picks()
        return df
    
    def import_schedules_func(self):
        df = nfl.import_schedules(range(2002, 2024))
        return df
    

class prep_data_class():
    def __init__(self, import_obj):

        # start runtime
        start_run = time.time()

        # create close game data 
        self.game_data_df = self.clean_data_func(import_obj.nfl_df, range(2016,2024))

        # calculate game epa
        self.game_epa_df = self.sum_epa_func(self.game_data_df, "posteam")
        self.game_epa_allowed_df = self.sum_epa_func(self.game_data_df, "defteam")

        # calcuate season epa
        self.seas_epa_df = self.season_epa_func(self.game_data_df)
        self.seas_epa_allowed_df = self.season_epa_func(self.game_data_df)

        # adjust for passing value
        self.epa_value_list = self.qb_value_adj_func(import_obj.nfl_df, range(2021,2024))

        # calculate qb points ratings
        self.qb_rankings_df = self.qb_ranking_func(self.epa_value_list)

        # active qbs
        self.active_qbs_df = self.active_qb_func(import_obj.quarterbacks_df)

        # identify starting qbs
        self.qb_one_df = self.qb_one_func(self.active_qbs_df, self.qb_rankings_df)

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("prep data runtime: " + str(end_run) + " minutes.")


    def clean_data_func(self, nfl_df, nfl_range): 
        
        # filter to selected seasons 
        clean_df = nfl_df.loc[(nfl_df['season'] >= min(nfl_range)) & 
                              (nfl_df['season'] <= max(nfl_range))]
        
        # fix aaron rodgers name
        clean_df["passer_player_name"] = clean_df["passer_player_name"].apply(
                                        lambda x: "A.Rodgers" if x == "Aa.Rodgers" else x)
        
        # drop columns 
        clean_df = clean_df.drop(columns=['nflverse_game_id', 'possession_team','offense_formation', 
                                          'offense_personnel', 'defenders_in_box','defense_personnel',
                                          'number_of_pass_rushers', 'players_on_play', 'offense_players',
                                          'defense_players','n_offense', 'n_defense'])
        
        # exclude playoffs, qb_kneel, qb_spike
        clean_df = clean_df[clean_df['season_type'] == 'REG']
        clean_df = clean_df[clean_df['qb_kneel'] == 0]
        clean_df = clean_df[clean_df['qb_spike'] == 0]
        
        # create score delta
        clean_df['current_score_differential'] = clean_df['posteam_score'] - clean_df['defteam_score']
        
        # create blowout column
        clean_df['blow_out'] = np.where((clean_df['qtr'] == 4) & (np.abs(clean_df['current_score_differential'])
                                                                  > 13.5), 1, 0)
        clean_df['blow_out'] = np.where((clean_df['qtr'] > 2) & (abs(clean_df['current_score_differential'])
                                                                 > 27.5), 1, clean_df['blow_out'])
        return clean_df
    

    def sum_epa_func(self, nfl_df, pos_side):
        if pos_side == "posteam":
            columns = ['passer_player_id', 'passer_player_name', 'posteam', 'qb_sum_epa']
        if pos_side == "defteam":
            columns = ['passer_player_id', 'passer_player_name', 'posteam', 'defteam','qb_sum_epa']
        df = nfl_df.groupby(['passer_player_id', 'passer_player_name']) \
                   .apply(lambda x: x.assign(qb_sum_epa = x['qb_epa'].sum())) \
                   .reset_index(drop=True) \
                   .loc[:, columns] \
                   .dropna(subset=['passer_player_name']) \
                   .drop_duplicates() \
                   .dropna() 
        return df
    

    def season_epa_func(self, nfl_df):
        df = nfl_df.groupby(['passer_player_id', 'season']) \
                   .apply(lambda x: x.assign(qb_sum_epa = x['qb_epa'].sum())) \
                   .reset_index(drop=True) \
                   .loc[:, ['season','passer_player_id','passer_player_name','posteam','qb_sum_epa']] \
                   .dropna(subset=['passer_player_name']) \
                   .drop_duplicates() \
                   .dropna()
        return df
    

    def qb_value_adj_func(self, nfl_df, nfl_range):
        
        # clean qb data
        qb_df = self.clean_data_func(nfl_df, nfl_range)

        # create passers dataframe
        passers = pd.DataFrame(qb_df['passer_player_id'])
        passers.columns = ['IDs']
        passers = passers[~passers['IDs'].isna()]
        passers = passers.groupby('IDs').size().reset_index(name='passes_thrown')
        
        # filter less than 100 completions
        passers = passers[passers['passes_thrown'] > 50]
        passers = passers.drop(columns=['passes_thrown'])
        passers = passers.drop_duplicates()
    
        # set pass/run dataframes
        pass_df = qb_df[qb_df['play_type'] == 'pass']
        run_df = qb_df[(qb_df['play_type'] == 'run') & (qb_df['rusher_player_id'].isin(passers['IDs']))]
    
        # concat pass/run
        out = pd.concat([pass_df, run_df])
        out['passer_player_id'] = np.where(out['play_type'] == 'run', out['rusher_player_id'], 
                                           out['passer_player_id'])
        
        # create qb epa
        out2 = out.groupby(['game_id', 'passer_player_id', 'passer_player_name', 'posteam', 
                            'season', 'week']).agg(qb_sum_epa=('epa', 'sum')).reset_index()
        out2 = out2[~out2['passer_player_name'].isna()].drop_duplicates()

        # count qb games played
        qb_count = out2.groupby('passer_player_id').size().reset_index(name='game_count')
        out2['game_count'] = out2['passer_player_id'].map(qb_count.set_index('passer_player_id')['game_count'])

        # create team epa
        out3 = out.groupby(['game_id','posteam','season','week']).agg(team_qb_epa=('epa','sum')).reset_index()
        out3 = out3.drop_duplicates()
        return [out2, out3]
    

    def qb_ranking_func(self, quarterback_df): 

        # groupby passer player id
        qb_n = quarterback_df[0].groupby('passer_player_id').size().reset_index(name='games')

        # create qb list 
        qb_list = pd.DataFrame({'passer_player_id': quarterback_df[0]['passer_player_id'].unique()})
        qb_list.columns = ['passer_player_id']
        qb_list = qb_list.merge(qb_n, on='passer_player_id', how='left')
        qb_list.columns = ['passer_player_id', 'games']
        qb_list = qb_list[qb_list['games'] >= 4]

        # calculate qb rankings
        player_ids = qb_list["passer_player_id"].unique()
        out = pd.DataFrame()
        for temp_ids in player_ids:
            df2 = quarterback_df[0][quarterback_df[0]['passer_player_id'] == temp_ids]
            df2 = df2.tail(17)
            x = min(15, len(df2) - 2)
            df2['wt_avg'] = df2["qb_sum_epa"].rolling(x).mean()
            # df2['wt_avg'] = df2["qb_sum_epa"].transform(lambda x: x.rolling(window=x_window, min_periods=1).mean().shift())
            out = out.append(df2.tail(1))
        out = out[['posteam', 'passer_player_id', 'passer_player_name', 'wt_avg']]
        return out
    

    def active_qb_func(self, players_df):
        qbs_2023 = players_df[players_df['status'] == "ACT"]
        qbs_2023 = qbs_2023[qbs_2023['position'] == 'QB']
        qbs_2023 = qbs_2023[(qbs_2023['status'] == 'ACT') | (qbs_2023['status'] == 'INA')]
        qbs_2023 = qbs_2023[['status', 'team_abbr', 'position', 'first_name', 'last_name', 'gsis_id']]
        qbs_2023.rename(columns={'gsis_id': 'passer_player_id'}, inplace=True)
        return qbs_2023
    

    def qb_one_func(self, quarterback_df, rankings_df):
        df = pd.merge(quarterback_df, rankings_df, on='passer_player_id', how='left')
        df = df.dropna(subset=['wt_avg'])
        df = df.drop(columns=['position'])
        df = df.reset_index(drop=True)
        qb_starters = df.drop([0,3,4,5,6,8,9,11,13,21,24,27,31,35,36,37,39,42,46,47,48,50,51,52,55,57])
        qb_starters.rename(columns={'team_abbr': 'team'}, inplace=True)
        return qb_starters