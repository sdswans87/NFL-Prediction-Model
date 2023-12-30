# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 06:24:49 2023

@author: swan0
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

class efficiency_data_class():
    def __init__(self, import_obj, epa_obj):

        # start runtime
        start_run = time.time()

        # calculate game epa per drop back basis
        self.game_efficiency = self.game_efficiency_func(epa_obj.close_game_data, epa_obj.cumulative_epa)
        self.game_efficiency_allowed = self.game_efficiency_func(epa_obj.close_game_data, epa_obj.cumulative_epa_allowed)

        # season epa per drop back 
        self.season_efficiency = self.season_efficiency_func(epa_obj.close_game_data, epa_obj.season_epa)

        # offense/defense efficiency
        self.off_efficiency = self.pos_efficiency_func(epa_obj.close_game_data, 'posteam')
        self.def_efficiency = self.pos_efficiency_func(epa_obj.close_game_data, 'defteam')

        # pass efficiency
        self.pass_efficiency = self.pass_efficiency_func(epa_obj.close_game_data, 'posteam')
        self.pass_efficiency_allowed = self.pass_efficiency_func(epa_obj.close_game_data, 'defteam')

        # run efficiency
        self.run_efficiency = self.run_efficiency_func(epa_obj.close_game_data, 'posteam')
        self.run_efficiency_allowed = self.run_efficiency_func(epa_obj.close_game_data, 'defteam')

        # join pass/run effciency
        self.off_epa_efficiency = self.pos_epa_efficiency_func(self.pass_efficiency, self.run_efficiency, 'posteam')
        self.def_epa_efficiency = self.pos_epa_efficiency_func(self.pass_efficiency_allowed, self.run_efficiency_allowed,'defteam')

        # points per game
        self.points_per_gm = self.points_per_game_func(epa_obj.close_game_data, 'posteam')
        self.points_per_gm_allowed = self.points_per_game_func(epa_obj.close_game_data, 'defteam')        

        # join offensive data with points data
        self.off_epa_efficiency = self.join_eff_points_func(self.off_epa_efficiency, self.points_per_gm, 'posteam')
        self.def_epa_efficiency = self.join_eff_points_func(self.def_epa_efficiency, self.points_per_gm_allowed, 'defteam')

        # add to epa stats
        self.off_efficiency_adj = self.cpoe_func(epa_obj.close_game_data, self.off_epa_efficiency, 'posteam')
        self.def_efficiency_adj = self.cpoe_func(epa_obj.close_game_data, self.def_epa_efficiency, 'defteam')

        # graph pass/run epa
        self.pass_epa_graph(self.off_efficiency_adj)
        self.run_epa_graph(self.off_efficiency_adj)

        # first down offensive efficiency 
        self.first_down_off_pass = self.down_efficiency_func(epa_obj.close_game_data, 1, 'pass', 'posteam')
        self.first_down_off_rush = self.down_efficiency_func(epa_obj.close_game_data, 1,'run', 'posteam')

        # first down defensive efficiency
        self.first_down_def_pass = self.down_efficiency_func(epa_obj.close_game_data, 1, 'pass', 'defteam')
        self.first_down_def_rush = self.down_efficiency_func(epa_obj.close_game_data, 1, 'run', 'defteam')
        
        # second down offensive efficiency
        self.second_down_off_pass = self.down_efficiency_func(epa_obj.close_game_data, 2, 'pass', 'posteam')
        self.second_down_off_rush = self.down_efficiency_func(epa_obj.close_game_data, 2, 'run', 'posteam')
        
        # second down defensive efficiency
        self.second_down_def_pass = self.down_efficiency_func(epa_obj.close_game_data, 2, 'pass', 'defteam')
        self.second_down_def_rush = self.down_efficiency_func(epa_obj.close_game_data, 2, 'run', 'defteam')

        # third down offensive efficiency
        self.third_down_off_pass = self.down_efficiency_func(epa_obj.close_game_data, 3, 'pass', 'posteam')
        self.third_down_off_rush = self.down_efficiency_func(epa_obj.close_game_data, 3, 'run', 'posteam')
        
        # third down defensive efficiency
        self.third_down_def_pass = self.down_efficiency_func(epa_obj.close_game_data, 3, 'pass', 'defteam')
        self.third_down_def_rush = self.down_efficiency_func(epa_obj.close_game_data, 3, 'run', 'defteam')

        # all down efficiency
        self.all_down_off_eff = self.all_down_efficiency_func(self.first_down_off_pass, self.first_down_off_rush, 
                                                              self.second_down_off_pass, self.second_down_off_rush,
                                                              self.third_down_off_pass, self.third_down_off_rush,
                                                              "posteam")
        self.all_down_def_eff = self.all_down_efficiency_func(self.first_down_def_pass, self.first_down_def_rush, 
                                                              self.second_down_def_pass, self.second_down_def_rush,
                                                              self.third_down_def_pass, self.third_down_def_rush,
                                                              "defteam")
        
        # total efficiency
        self.total_off_efficiency = self.total_efficiency_func(self.off_efficiency_adj, self.all_down_off_eff, 'posteam')
        self.total_def_efficiency = self.total_efficiency_func(self.def_efficiency_adj, self.all_down_def_eff, 'defteam')

        # model prep - gather data
        self.model_data = self.prep_model_data_func(import_obj.nfl_data, epa_obj)

        # model prep - passing epa
        self.pass_epa = self.pass_efficiency_func(self.model_data, 'posteam')
        self.pass_epa_allowed = self.pass_efficiency_func(self.model_data, 'defteam')

        # model prep - run epa
        self.run_epa = self.run_efficiency_func(self.model_data, 'posteam')
        self.run_epa_allowed = self.run_efficiency_func(self.model_data, 'defteam')

        # model prep - total epa
        self.total_epa = self.pos_epa_efficiency_func(self.pass_epa, self.run_epa, 'posteam')
        self.total_epa_allowed = self.pos_epa_efficiency_func(self.pass_epa_allowed, self.run_epa_allowed, 'defteam')

        # model prep - points a game
        self.points_game = self.points_per_game_func(self.model_data, 'posteam')
        self.points_game_allowed = self.points_per_game_func(self.model_data, 'defteam')

        # model prep - off/def epa 
        self.offensive_epa = self.join_eff_points_func(self.total_epa, self.points_game, 'posteam')
        self.defensive_epa = self.join_eff_points_func(self.total_epa_allowed, self.points_game_allowed, 'defteam')

        # model prep - off/def efficiency
        self.offensive_efficiency = self.cpoe_func(self.model_data, self.offensive_epa, 'posteam')
        self.defensive_efficiency = self.cpoe_func(self.model_data, self.defensive_epa, 'defteam')

        # model prep - first down offensive efficiency 
        self.first_down_off_passing = self.down_efficiency_func(self.model_data, 1, 'pass', 'posteam')
        self.first_down_off_rushing = self.down_efficiency_func(self.model_data, 1,'run', 'posteam')

        # model prep - first down defensive efficiency
        self.first_down_def_passing = self.down_efficiency_func(self.model_data, 1, 'pass', 'defteam')
        self.first_down_def_rushing = self.down_efficiency_func(self.model_data, 1, 'run', 'defteam')
        
        # model prep - second down offensive efficiency
        self.second_down_off_passing = self.down_efficiency_func(self.model_data, 2, 'pass', 'posteam')
        self.second_down_off_rushing = self.down_efficiency_func(self.model_data, 2, 'run', 'posteam')
        
        # model prep - second down defensive efficiency
        self.second_down_def_passing = self.down_efficiency_func(self.model_data, 2, 'pass', 'defteam')
        self.second_down_def_rushing = self.down_efficiency_func(self.model_data, 2, 'run', 'defteam')

        # model prep - third down offensive efficiency
        self.third_down_off_passing = self.down_efficiency_func(self.model_data, 3, 'pass', 'posteam')
        self.third_down_off_rushing = self.down_efficiency_func(self.model_data, 3, 'run', 'posteam')
        
        # model prep - third down defensive efficiency
        self.third_down_def_passing = self.down_efficiency_func(self.model_data, 3, 'pass', 'defteam')
        self.third_down_def_rushing = self.down_efficiency_func(self.model_data, 3, 'run', 'defteam')

        # model prep - all down efficiency
        self.all_down_offense = self.all_down_efficiency_func(self.first_down_off_passing, self.first_down_off_rushing, 
                                                              self.second_down_off_passing, self.second_down_off_rushing,
                                                              self.third_down_off_passing, self.third_down_off_rushing,
                                                              "posteam")
        self.all_down_defense = self.all_down_efficiency_func(self.first_down_def_passing, self.first_down_def_rushing, 
                                                              self.second_down_def_passing, self.second_down_def_rushing,
                                                              self.third_down_def_passing, self.third_down_def_rushing,
                                                              "defteam")
        
        # model prep - total efficiency
        self.total_offense_eff = self.total_efficiency_func(self.offensive_efficiency, self.all_down_offense, 'posteam')
        self.total_defense_eff = self.total_efficiency_func(self.defensive_efficiency, self.all_down_defense, 'defteam')

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("efficiency data runtime: " + str(end_run) + " minutes.")


    def game_efficiency_func(self, nfl_data, epa_data):
        qb_plays = nfl_data.groupby('passer_player_id').size().reset_index(name='n_passes')
        qb_plays = qb_plays.dropna()
        qb_plays = qb_plays.drop_duplicates()
        epa_data['n_passes'] = epa_data['passer_player_id'].map(qb_plays.set_index('passer_player_id')['n_passes'])
        out = epa_data[epa_data['n_passes'] > 10].copy()
        out['epa_play'] = out['qb_sum_epa'] / out['n_passes']
        return out


    def season_efficiency_func(self, nfl_data, epa_data):
        qb_plays = nfl_data.groupby(['passer_player_id', 'passer_player_name', 'season']).size().reset_index(name='n_passes')
        qb_plays = qb_plays.dropna(subset=['passer_player_name'])
        qb_plays = qb_plays.drop_duplicates()
        out = pd.merge(epa_data, qb_plays, on=['season', 'passer_player_id', 'passer_player_name'], how='left')
        out = out[out['n_passes'] > 10]
        out = out.dropna()
        out['epa_play'] = out['qb_sum_epa'] / out['n_passes']
        return out
    

    def pos_efficiency_func(self, nfl_data, pos_def):
        efficiency = nfl_data.groupby(['game_id', pos_def]).agg(game_epa=('epa', 'sum'), 
                                                                avg_cpoe=('cpoe', 'mean'),
                                                                sum_cpoe=('cpoe', 'sum')).reset_index()
        efficiency = efficiency.dropna()
        return efficiency
    

    def pass_efficiency_func(self, nfl_data, pos_def):
        pass_eff = nfl_data[nfl_data['pass'] == 1].groupby(['game_id', pos_def]).agg(
                   pass_epa_game=('epa', 'sum'), 
                   n_pass=('pass', 'count'), 
                   pass_epa_dropback=('epa', 'mean'),
                   succ_pass_pct=('success', 'mean')
                   ).reset_index().dropna()
        return pass_eff
    

    def run_efficiency_func(self, nfl_data, pos_def):
        run_eff = nfl_data[nfl_data['pass'] == 0].groupby(['game_id', pos_def]).agg(
                  rush_epa_game=('epa', 'sum'),
                  n_rush=('rush', 'count'),
                  rush_epa_dropback=('epa', 'mean'),
                  succ_rush_pct=('success', 'mean')
                  ).reset_index().dropna()
        return run_eff
    

    def pos_epa_efficiency_func(self, pass_efficiency, run_efficiency, pos_def):
        pos_epa = pd.merge(pass_efficiency, run_efficiency, on=['game_id', pos_def], how='left')
        pos_epa = pos_epa.dropna()
        pos_epa['n_plays'] = pos_epa['n_pass'] + pos_epa['n_rush']
        pos_epa[['year', 'week', 'away_team', 'home_team']] = pos_epa['game_id'].str.split('_', expand=True)
        temp_dict = {"OAK":"LV", "SD":"LAC"}
        for key,value in temp_dict.items():
            pos_epa.loc[pos_epa["home_team"] == key, "home_team"] = value
            pos_epa.loc[pos_epa["away_team"] == key, "away_team"] = value
        return pos_epa
    

    def points_per_game_func(self, nfl_data, pos_def):
        nfl_data['eff_fg_pts'] = nfl_data['fg_prob'] * 3 * nfl_data['field_goal_attempt']
        pts_per_game = nfl_data.groupby(['game_id', pos_def]).agg(
                                home_final=('home_score', 'max'),
                                away_final=('away_score', 'max'),
                                home_team=('home_team', "max"),
                                away_team=('away_team', "max"),
                                total_tds=('touchdown', 'sum'),
                                total_fgs_att=('field_goal_attempt', 'sum'),
                                total_pat=('extra_point_attempt', 'sum'),
                                total_eff_fg_pts=('eff_fg_pts', 'sum')
                                ).reset_index()
        pts_per_game["total_effective_pts"] = pts_per_game["total_tds"] * 6 + pts_per_game["total_eff_fg_pts"]
        pts_per_game['total_score'] = pts_per_game['home_final'] + pts_per_game['away_final']
        pts_per_game['pt_diff'] = pts_per_game['home_final'] - pts_per_game['away_final']
        pts_per_game = pts_per_game.loc[pts_per_game[pos_def].notna()].drop_duplicates()
        return pts_per_game
    

    def join_eff_points_func(self, eff_data, points_data, pos_def):
        eff_points = pd.merge(eff_data, points_data, on=['game_id', pos_def, 'home_team', 'away_team'], how='left')
        if pos_def == 'posteam':
            eff_points['poss_score'] = pd.NA
            for row in range(len(eff_points)):
                if eff_points['posteam'][row] == eff_points['home_team'][row]:
                    eff_points['poss_score'][row] = eff_points['home_final'][row]
                else:
                    eff_points['poss_score'][row] = eff_points['away_final'][row]
        if pos_def == 'defteam':
            eff_points['score_allowed'] = pd.NA
            for row in range(len(eff_points)):
                if eff_points['defteam'][row] == eff_points['home_team'][row]:
                    eff_points['score_allowed'][row] = eff_points['home_final'][row]
                else:
                    eff_points['score_allowed'][row] = eff_points['away_final'][row]
        return eff_points


    def cpoe_func(self, nfl_data, epa_data, pos_def):
        epa = nfl_data[nfl_data['cpoe'].notna()]
        epa = epa.groupby(['game_id', pos_def]).agg(avg_cpoe=('cpoe', 'mean'), total_cpoe=('cpoe', 'sum')).reset_index()
        out = pd.merge(epa_data, epa, on=['game_id', pos_def], how='left').drop_duplicates()
        return out
    

    def pass_epa_graph(self, temp_df):
        plt.hist(temp_df['pass_epa_game'], bins=range(-40, 41, 1))
        plt.xlim(-40, 40)
        plt.title('Pass EPA Per Game')
        plt.show()
        
    def run_epa_graph(self, temp_df):
        plt.hist(temp_df['rush_epa_game'], bins=range(-25, 26, 1))
        plt.xlim(-25, 25)
        plt.title('Run EPA Per Game')
        plt.show()


    def down_efficiency_func(self, nfl_data, dwn, pass_run, pos_def):
        down = nfl_data[nfl_data['down'] == dwn]
        down_eff = down[down['play_type'] == pass_run].groupby(['game_id', pos_def]).agg(
            sum_epa_play_type=('epa', 'sum'),
            total_success=('success', 'sum'),
            n_play=('play', 'count')
            ).reset_index()
        down_eff["epa_per_play"] = down_eff["sum_epa_play_type"] / down_eff["n_play"]
        down_eff["succ_rate_play"] = down_eff["total_success"] / down_eff["n_play"]
        down_eff.rename(columns={'sum_epa_play_type': 'sum_epa_play_type_' + pass_run + "_" + str(dwn),
                                 'total_success': 'total_success_' + pass_run + "_" + str(dwn),
                                 'n_play': 'n_play_' + pass_run + "_" + str(dwn),
                                 'epa_per_play': 'epa_per_play_' + pass_run + "_" + str(dwn),
                                 'succ_rate_play': 'succ_rate_play_' + pass_run + "_" + str(dwn)}, inplace=True)
        return down_eff
    

    def all_down_efficiency_func(self, df1, df2, df3, df4, df5, df6, pos_def):
        all_down_eff = df1.merge(df2, on = ['game_id',pos_def], how = 'left') 
        all_down_eff = all_down_eff.merge(df3, on = ['game_id',pos_def], how = 'left') 
        all_down_eff = all_down_eff.merge(df4, on = ['game_id',pos_def], how = 'left') 
        all_down_eff = all_down_eff.merge(df5, on = ['game_id',pos_def], how = 'left') 
        all_down_eff = all_down_eff.merge(df6, on = ['game_id',pos_def], how = 'left') 
        return all_down_eff
    

    def total_efficiency_func(self, df, df2, pos_def):
        out = pd.merge(df, df2, on=['game_id', pos_def], how='left')
        out['pass_rate'] = out['n_pass'] / out['n_plays']
        out['run_rate'] = 1 - out['pass_rate'] 
        out['pass_rate_first'] = out['n_play_pass_1'] / out['n_play_pass_1'] + out['n_play_run_1']
        out['pass_rate_second'] = out['n_play_pass_2'] / out['n_play_pass_2'] + out['n_play_run_2']
        return out
    

    def prep_model_data_func(self, nfl_data, epa_obj):
        final_offense = epa_obj.clean_data_func(nfl_data, range(2022,2024))
        final_offense['week'] = np.where((final_offense['week'] == 1) & (final_offense['season'] == 2023), 19, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 2) & (final_offense['season'] == 2023), 20, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 3) & (final_offense['season'] == 2023), 21, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 4) & (final_offense['season'] == 2023), 22, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 5) & (final_offense['season'] == 2023), 23, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 6) & (final_offense['season'] == 2023), 24, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 7) & (final_offense['season'] == 2023), 25, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 8) & (final_offense['season'] == 2023), 26, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 9) & (final_offense['season'] == 2023), 27, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 10) & (final_offense['season'] == 2023), 28, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 11) & (final_offense['season'] == 2023), 29, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 12) & (final_offense['season'] == 2023), 30, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 13) & (final_offense['season'] == 2023), 31, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 14) & (final_offense['season'] == 2023), 32, final_offense['week'])
        final_offense['week'] = np.where((final_offense['week'] == 15) & (final_offense['season'] == 2023), 33, final_offense['week'])
        final_offense['season'] = 2023
        final_offense = final_offense[final_offense['week'] > 10]
        final_offense['week'] = final_offense['week'] - 10
        final_offense['week'].value_counts()
        return final_offense