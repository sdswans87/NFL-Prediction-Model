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
        
        # filter out nans 
        clean_df['nans_3_qtr'] = np.where((clean_df['qtr'] == 3) &
                                           clean_df['current_score_differential'].isnull(), 1, 0)
        clean_df['nans_4_qtr'] = np.where((clean_df['qtr'] == 4) & 
                                           clean_df['current_score_differential'].isnull(), 1, 0)
        clean_df['nans_5_qtr'] = np.where((clean_df['qtr'] == 5) & 
                                           clean_df['current_score_differential'].isnull(), 1, 0)
        clean_df = clean_df[clean_df['nans_3_qtr'] == 0]
        clean_df = clean_df[clean_df['nans_4_qtr'] == 0]
        clean_df = clean_df[clean_df['nans_5_qtr'] == 0]
        clean_df = clean_df[clean_df['blow_out'] == 0]
        clean_df = clean_df.drop('nans_3_qtr', axis=1)
        clean_df = clean_df.drop('nans_4_qtr', axis=1)
        clean_df = clean_df.drop('nans_5_qtr', axis=1)
        return clean_df