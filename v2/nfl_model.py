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
    def __init__(self, import_object):

        # start runtime
        start_run = time.time()

        # source data
        self.src_game_data = nfl_obj.source_game_data(import_object.imp_game_data, range(2016,2024))
        self.src_passing_data = nfl_obj.source_game_data(import_object.imp_game_data, range(2021,2024))
        self.src_quarterback_data = nfl_obj.source_quarterback_data(import_object.imp_quarterback_data)
        self.src_rookie_data = nfl_obj.source_rookie_data(import_object.imp_rookie_data)  
        self.src_rookie_pass_data = nfl_obj.source_rookie_pass_data(import_object.imp_game_data, self.src_rookie_data) 
        self.src_rookie_run_data = nfl_obj.source_rookie_run_data(import_object.imp_game_data, self.src_rookie_data) 
        self.src_schedule_data = import_object.imp_schedule_data[import_object.imp_schedule_data['season'] != 2020]
        
        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("source data runtime: " + str(end_run) + " minutes.")