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

        # import nfl data past 22 years
        self.nfl_df = self.import_games()

        # import quarterback data for 2023 
        self.quarterbacks_df = self.import_quarterbacks()

        # import rookie data
        self.rookies_df = self.import_rookies()

        # import nfl schedule data
        self.schedules_df = self.import_schedules()

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("import data runtime: " + str(end_run) + " minutes.")

    def import_games(self):
        df = nfl.import_pbp_data(range(2002,2024))
        return df
    
    def import_quarterbacks(self):
        df = nfl.import_players()
        return df

    def import_rookies(self):
        df = nfl.import_draft_picks()
        return df
    
    def import_schedules(self):
        df = nfl.import_schedules(range(2002, 2024))
        return df
    

class prep_data_class():
    def __init__(self, import_obj):

        # start runtime
        start_run = time.time()

        # create close game data 
        self.close_gm_df = self.clean_data(import_obj.nfl_df, range(2016,2022))

        # calculate cumulative epa
        self.sum_epa_df = self.sum_epa(self.close_gm_df, "posteam")
        self.sum_epa_allowed_df = self.sum_epa(self.close_gm_df, "defteam")

        # calcuate per season epa
        self.seas_epa_df = self.season_epa(self.close_gm_df)
        self.seas_epa_allowed_df = self.season_epa(self.close_gm_df)

        # pull qb data for passing value calculation
        self.qb_data_df = self.clean_data(import_obj.nfl_df, range(2021,2023))

        # adjust for qb passing value
        self.qb_value_df = self.qb_value_adjustment(self.qb_data_df)

        # calculate qb points ratings
        self.qb_ratings_df = self.calc_qb_ratings(self.qb_value_df)

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("prep data runtime: " + str(end_run) + " minutes.")


    def clean_data(self, nfl_df, nfl_range): 
        
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
        
        # identify nan values by quarter
        clean_df['nans_3_qtr'] = np.where((clean_df['qtr'] == 3) &
                                           clean_df['current_score_differential'].isnull(), 1, 0)
        clean_df['nans_4_qtr'] = np.where((clean_df['qtr'] == 4) & 
                                           clean_df['current_score_differential'].isnull(), 1, 0)
        clean_df['nans_5_qtr'] = np.where((clean_df['qtr'] == 5) & 
                                           clean_df['current_score_differential'].isnull(), 1, 0)
        
        # set blanks to zero
        clean_df = clean_df[clean_df['nans_3_qtr'] == 0]
        clean_df = clean_df[clean_df['nans_4_qtr'] == 0]
        clean_df = clean_df[clean_df['nans_5_qtr'] == 0]
        clean_df = clean_df[clean_df['blow_out'] == 0]

        # drop columns
        clean_df = clean_df.drop('nans_3_qtr', axis=1)
        clean_df = clean_df.drop('nans_4_qtr', axis=1)
        clean_df = clean_df.drop('nans_5_qtr', axis=1)
        return clean_df
    
    
    def sum_epa(self, nfl_df, pos_side):
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
    

    def season_epa(self, nfl_df):
        df = nfl_df.groupby(['passer_player_id', 'season']) \
                   .apply(lambda x: x.assign(qb_sum_epa = x['qb_epa'].sum())) \
                   .reset_index(drop=True) \
                   .loc[:, ['season','passer_player_id','passer_player_name','posteam','qb_sum_epa']] \
                   .dropna(subset=['passer_player_name']) \
                   .drop_duplicates() \
                   .dropna()
        return df
    

    def qb_value_adjustment(self, nfl_df):
        
        # create passers dataframe
        passers = pd.DataFrame(nfl_df['passer_player_id'])
        passers.columns = ['IDs']
        passers = passers[~passers['IDs'].isna()]
        passers = passers.groupby('IDs').size().reset_index(name='passes_thrown')
        
        # filter less than 100 completions
        passers = passers[passers['passes_thrown'] > 100]
        passers = passers.drop(columns=['passes_thrown'])
        passers = passers.drop_duplicates()
    
        # set pass/run dataframes
        pass_df = nfl_df[nfl_df['play_type'] == 'pass']
        run_df = nfl_df[(nfl_df['play_type'] == 'run') & (nfl_df['rusher_player_id'].isin(passers['IDs']))]
    
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
    

    def calc_qb_ratings(self, quarterback_df): 

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