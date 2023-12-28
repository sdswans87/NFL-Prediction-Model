# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 06:24:49 2023

@author: swan0
"""
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
        return pos_epa