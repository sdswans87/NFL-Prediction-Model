# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 12:19:07 2024

@author: swan0
"""
import pandas as pd
import time


class game_efficiency_data():
    def __init__(self, src_obj, epa_obj, rk_obj, qb_obj):

        # start runtime
        start_run = time.time()

        # game epa per drop back basis
        self.game_efficiency = self.game_efficiency_data(src_obj.src_game_data, epa_obj.epa_cumulative)
        self.game_efficiency_allowed = self.game_efficiency_data(src_obj.src_game_data, epa_obj.epa_cumulative_allowed)

        # game epa per drop back basis by season
        self.season_efficiency = self.game_efficiency_by_season_data(src_obj.src_game_data, epa_obj.epa_season)

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("game efficiency data runtime: " + str(end_run) + " minutes.")


    def game_efficiency_data(self, game_data, epa_data):
        qb_plays = game_data.groupby('passer_player_id').size().reset_index(name='n_passes')
        qb_plays = qb_plays.dropna()
        qb_plays = qb_plays.drop_duplicates()
        epa_data['n_passes'] = epa_data['passer_player_id'].map(qb_plays.set_index('passer_player_id')['n_passes'])
        out = epa_data[epa_data['n_passes'] > 10].copy()
        out['epa_play'] = out['qb_sum_epa'] / out['n_passes']
        return out
    

    def game_efficiency_by_season_data(self, game_data, epa_data):
        qb_plays = game_data.groupby(['passer_player_id', 'passer_player_name', 'season']).size().reset_index(name='n_passes')
        qb_plays = qb_plays.dropna(subset=['passer_player_name'])
        qb_plays = qb_plays.drop_duplicates()
        out = pd.merge(epa_data, qb_plays, on=['season', 'passer_player_id', 'passer_player_name'], how='left')
        out = out[out['n_passes'] > 10]
        out = out.dropna()
        out['epa_play'] = out['qb_sum_epa'] / out['n_passes']
        return out
    

