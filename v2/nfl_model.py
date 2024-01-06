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
    def __init__(self, imp_obj):

        # start runtime
        start_run = time.time()

        # source game datadata
        self.src_game_data = nfl_obj.source_game_data(imp_obj.imp_game_data, range(2016,2024))

        # source quarterback data
        self.src_quarterback_data = nfl_obj.source_quarterback_data(imp_obj.imp_quarterback_data)
        self.src_passing_data = nfl_obj.source_game_data(imp_obj.imp_game_data, range(2021,2024))
        self.src_offense_data = nfl_obj.source_game_data(imp_obj.imp_game_data, range(2022,2024))
    
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