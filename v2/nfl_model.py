# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 05:10:30 2024

@author: swan0
"""
import nfl_data_py as nfl
import time
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import nfl_methods as nfl_obj


class import_nfl_data():
    def __init__(self):

        # start runtime
        start_run = time.time()

        # input range for nfl data pull
        game_range_start = 2005
        game_range_end = 2024

        # input range for schedule data pull 
        sched_range_start = 2010
        sched_range_end = 2024

        # import nfl data 
        self.nfl_data = nfl_obj.import_game_data(game_range_start, game_range_end)

        # import quarterback data for 2023 
        self.quarterback_data = nfl_obj.import_quarterback_data()

        # import rookie data
        self.rookie_data = nfl_obj.import_rookie_data()

        # import nfl schedule data
        self.schedule_data = nfl_obj.import_schedule_data(sched_range_start, sched_range_end)

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("import data runtime: " + str(end_run) + " minutes.")