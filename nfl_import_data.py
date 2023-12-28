# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:08:57 2023

@author: swan0
"""
import nfl_data_py as nfl
import time
import warnings
warnings.simplefilter(action='ignore', category=Warning)

class import_data_class():
    def __init__(self):

        # start runtime
        start_run = time.time()

        # import nfl data past 20 years
        self.nfl_data = self.import_games_func()

        # import quarterback data for 2023 
        self.quarterback_data = self.import_qb_func()

        # import rookie data
        self.rookie_data = self.import_rookies_func()

        # import nfl schedule data
        self.schedule_data = self.import_schedules_func()

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("import data runtime: " + str(end_run) + " minutes.")

    def import_games_func(self):
        df = nfl.import_pbp_data(range(2005,2024))
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