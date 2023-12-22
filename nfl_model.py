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


class nfl_model_class():
    def __init__(self, nfl_week):

        # start runtime
        start_run = time.time()

        # set weekly odds to predict
        self.nfl_week = nfl_week

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
        print("nfl_model.py runtime: " + str(end_run) + " minutes.")


    def import_games(self):
        games_df = nfl.import_pbp_data(range(2002,2024))
        return games_df
    

    def import_quarterbacks(self):
        qbs_df = nfl.import_players()
        qbs_df = qbs_df[qbs_df['status'] == "ACT"]
        qbs_df = qbs_df[qbs_df['position'] == 'QB']
        qbs_df = qbs_df[(qbs_df['status'] == 'ACT') | (qbs_df['status'] == 'INA')]
        qbs_df = qbs_df[['status', 'team_abbr', 'position', 'first_name', 'last_name', 'gsis_id']]
        qbs_df.rename(columns={'gsis_id': 'passer_player_id'}, inplace=True)
        return qbs_df


    def import_rookies(self):
        rookies_df = nfl.import_draft_picks()
        rookies_df = rookies_df[rookies_df['season'] > 2002]
        rookies_df = rookies_df[rookies_df['position'] == 'QB']
        rookies_df = rookies_df[~rookies_df['gsis_id'].isna()]
        rookies_df = rookies_df[rookies_df['round'] == 1]
        return rookies_df
    

    def import_schedules(self):
        schedule_df = nfl.import_schedules(range(2002, 2024))
        return schedule_df