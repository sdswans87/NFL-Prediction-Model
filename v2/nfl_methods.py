# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 05:10:30 2024

@author: swan0
"""
import nfl_data_py as nfl
import numpy as np
import pandas as pd


# import game data
def import_game_data(start_range, end_range):
    df = nfl.import_pbp_data(range(start_range, end_range))
    return df


# import schedule data
def import_schedule_data(start_range, end_range):
    df = nfl.import_schedules(range(start_range, end_range))
    return df


# import rookie data
def import_rookie_data(start_range):
    df = nfl.import_draft_picks()
    df = df[df['season'] > start_range]
    return df


# import quarterback data
def import_quarterback_data():
    df = nfl.import_players()
    return df


# source close game data
def source_game_data(game_data, nfl_range): 
    df = game_data.loc[(game_data['season'] >= min(nfl_range)) & (game_data['season'] <= max(nfl_range))]
    df["passer_player_name"] = df["passer_player_name"].apply(lambda x: "A.Rodgers" if x == "Aa.Rodgers" else x)
    df = df[df['season_type'] == 'REG']
    df = df[df['qb_kneel'] == 0]
    df = df[df['qb_spike'] == 0]
    df['current_score_differential'] = df['posteam_score'] - df['defteam_score']
    df['blow_out'] = np.where((df['qtr'] == 4) & (np.abs(df['current_score_differential']) > 13.5), 1, 0)
    df['blow_out'] = np.where((df['qtr'] > 2) & (abs(df['current_score_differential']) > 27.5), 1, df['blow_out'])
    return df


# source active quarterback data
def source_quarterback_data(quarterback_data):
    df = quarterback_data[quarterback_data['status'] == "ACT"]
    df = df[df['position'] == 'QB']
    df = df[(df['status'] == 'ACT') | (df['status'] == 'INA')]
    df = df[['status', 'team_abbr', 'position', 'first_name', 'last_name', 'gsis_id']]
    df.rename(columns={'gsis_id': 'passer_player_id'}, inplace=True)
    return df


# source rookie data
def source_rookie_data(rookie_data):
    df = rookie_data[rookie_data['position'] == 'QB']
    df = df[~df['gsis_id'].isna()]
    df = df[df['round'] == 1]
    df = df[['gsis_id', 'season', 'pfr_player_name']]
    df.rename(columns={'gsis_id': 'passer_player_id'}, inplace=True)
    return df