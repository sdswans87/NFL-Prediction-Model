# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 05:10:30 2024

@author: swan0
"""
import nfl_data_py as nfl
import numpy as np
import pandas as pd


# pull game data
def import_game_data(start_range, end_range):
    nfl_df = nfl.import_pbp_data(range(start_range, end_range))
    return nfl_df


# pull quarterback data
def import_quarterback_data():
    qb_df = nfl.import_players()
    return qb_df


# pull rookie data
def import_rookie_data():
    rookie_df = nfl.import_draft_picks()
    return rookie_df


# pull schedule data
def import_schedule_data(start_range, end_range):
    schedule_df = nfl.import_schedules(range(start_range, end_range))
    return schedule_df
