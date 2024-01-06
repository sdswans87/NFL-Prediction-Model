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