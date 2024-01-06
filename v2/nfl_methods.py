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


# source rookie pass data
def source_rookie_pass_data(game_data, rookie_data):
    df = game_data[game_data['play_type'] == 'pass']
    df = df[df['passer_player_id'].isin(rookie_data['passer_player_id'])]  
    return df


# source rookie run data
def source_rookie_run_data(game_data, rookie_data):
    df = game_data[game_data['play_type'] == 'run']
    df = df[df['rusher_player_id'].isin(rookie_data['passer_player_id'])]  
    return df


# epa cumulative data
def epa_cumulative_data(game_data, pos_columns):
    df = game_data.groupby(['passer_player_id', 'passer_player_name']) \
                  .apply(lambda x: x.assign(qb_sum_epa = x['qb_epa'].sum())) \
                  .reset_index(drop=True) \
                  .loc[:, pos_columns] \
                  .dropna(subset=['passer_player_name']) \
                  .drop_duplicates() \
                  .dropna() 
    return df


# epa season data
def epa_season_data(game_data):
    df = game_data.groupby(['passer_player_id', 'season']) \
                  .apply(lambda x: x.assign(qb_sum_epa = x['qb_epa'].sum())) \
                  .reset_index(drop=True) \
                  .loc[:, ['season','passer_player_id','passer_player_name','posteam','qb_sum_epa']] \
                  .dropna(subset=['passer_player_name']) \
                  .drop_duplicates() \
                  .dropna()
    return df


# epa passing value data 
def epa_passing_value_data(game_data):
    passers = pd.DataFrame(game_data['passer_player_id'])
    passers.columns = ['IDs']
    passers = passers[~passers['IDs'].isna()]
    passers = passers.groupby('IDs').size().reset_index(name='passes_thrown')
    passers = passers[passers['passes_thrown'] > 50]
    passers = passers.drop(columns=['passes_thrown'])
    passers = passers.drop_duplicates()
    pass_df = game_data[game_data['play_type'] == 'pass']
    run_df = game_data[(game_data['play_type'] == 'run') & (game_data['rusher_player_id'].isin(passers['IDs']))]
    out = pd.concat([pass_df, run_df])
    out['passer_player_id'] = np.where(out['play_type'] == 'run', out['rusher_player_id'], 
                                        out['passer_player_id'])
    out2 = out.groupby(['game_id', 'passer_player_id', 'passer_player_name', 'posteam', 
                        'season', 'week']).agg(qb_sum_epa=('epa', 'sum')).reset_index()
    out2 = out2[~out2['passer_player_name'].isna()].drop_duplicates()
    qb_count = out2.groupby('passer_player_id').size().reset_index(name='game_count')
    out2['game_count'] = out2['passer_player_id'].map(qb_count.set_index('passer_player_id')['game_count'])
    out3 = out.groupby(['game_id','posteam','season','week']).agg(team_qb_epa=('epa','sum')).reset_index()
    out3 = out3.drop_duplicates()
    return [out2, out3]


# epa passing efficiency data
def epa_pass_efficiency_data(game_data, side):
    df = game_data[game_data['pass'] == 1].groupby(['game_id', side]).agg(
                   pass_epa_game=('epa', 'sum'), 
                   n_pass=('pass', 'count'), 
                   pass_epa_dropback=('epa', 'mean'),
                   succ_pass_pct=('success', 'mean')
                   ).reset_index().dropna()
    return df


# epa rushing efficiency data
def epa_rush_efficiency_data(game_data, side):
    df = game_data[game_data['pass'] == 0].groupby(['game_id', side]).agg(
                   rush_epa_game=('epa', 'sum'),
                   n_rush=('rush', 'count'),
                   rush_epa_dropback=('epa', 'mean'),
                   succ_rush_pct=('success', 'mean')
                   ).reset_index().dropna()
    return df


# epa passing/rushing combined data
def epa_total_efficiency_data(pass_data, rush_data, side):
    df = pd.merge(pass_data, rush_data, on=['game_id', side], how='left')
    df = df.dropna()
    df['n_plays'] = df['n_pass'] + df['n_rush']
    df[['year', 'week', 'away_team', 'home_team']] = df['game_id'].str.split('_', expand=True)
    temp_dict = {"OAK":"LV", "SD":"LAC"}
    for key,value in temp_dict.items():
        df.loc[df["home_team"] == key, "home_team"] = value
        df.loc[df["away_team"] == key, "away_team"] = value
    return df


# epa quarterback data
def epa_quarterback_data(pass_data, nfl_seasons):
    tms = pass_data['posteam'].unique()
    out = pd.DataFrame()
    epa_passing = pass_data[pass_data['season'].isin(nfl_seasons)]
    for tm in tms:
        df2 = epa_passing[epa_passing['posteam'] == tm]
        x = df2.shape[0] - 2
        df2['wt_avg_team'] = df2["team_qb_epa"].rolling(x).mean()
        out = pd.concat([out, df2.tail(1)])
    out = out[['posteam', 'wt_avg_team']]
    out.columns = ['team', 'wt_avg_team']
    return out


# rookie play data
def rookie_play_data(pass_df, run_df):
    rookie_plays = pd.concat([pass_df, run_df])
    rookie_plays = rookie_plays.groupby(['season','game_id', 'passer_player_id']).apply(
                                            lambda x: x['epa'].sum()).reset_index()
    rookie_plays.columns = ['season','game_id', 'passer_player_id', 'qb_epa']
    rookie_plays = rookie_plays.drop_duplicates()
    return rookie_plays


# rookie epa data
def rookie_epa_data(rookie_df, rookie_plays):
    rookie_epa = pd.merge(rookie_df, rookie_plays, on=['passer_player_id', 'season'], how='left')
    return rookie_epa


# rookie mean data
def rookie_mean_data(rookie_df):
    rookie_mean = rookie_df['qb_epa'].mean()
    return rookie_mean


# quarterback starter data
def qb_starters_data(game_data, season, week):
    df = game_data[game_data['season'] == season]
    df = df[df['week'] == week]
    df = df.groupby(['passer_player_id', 'passer_player_name', 'posteam']) \
            .apply(lambda x: x.assign(sum_epa = x['epa'].sum())) \
            .loc[:, ['passer_player_name', 'passer_player_id', 'posteam']] \
            .dropna(subset=['passer_player_name']) \
            .drop_duplicates() \
            .dropna() \
            .reset_index(drop=True) 
    df = df.drop([12,11,23,26,31,4,36,39])   
    df = df.reset_index(drop=True)     
    return df


# quarterback rankings data
def qb_rankings_data(qb_data): 
    qb_n = qb_data[0].groupby('passer_player_id').size().reset_index(name='games')
    qb_list = pd.DataFrame({'passer_player_id': qb_data[0]['passer_player_id'].unique()})
    qb_list.columns = ['passer_player_id']
    qb_list = qb_list.merge(qb_n, on='passer_player_id', how='left')
    qb_list.columns = ['passer_player_id', 'games']
    qb_list = qb_list[qb_list['games'] >= 4]

    player_ids = qb_list["passer_player_id"].unique()
    out = pd.DataFrame()
    for temp_ids in player_ids:
        df2 = qb_data[0][qb_data[0]['passer_player_id'] == temp_ids]
        df2 = df2.tail(17)
        x = min(15, len(df2) - 2)
        df2['wt_avg'] = df2["qb_sum_epa"].rolling(x).mean()
        out = out.append(df2.tail(1))
    out = out[['posteam', 'passer_player_id', 'passer_player_name', 'wt_avg']]
    return out


# quarterback starter ranking data
def qb_starter_rankings_data(qb_starters, qb_rankings):
    df = pd.merge(qb_starters, qb_rankings, on=['passer_player_name','passer_player_id', 'posteam'])
    df = df.reset_index(drop=True)
    return df


# quarterback adjustment to rankings
def qb_rankings_adj_data(epa_data, starter_data, rookie_mean):
    starter_data.rename(columns={'posteam': 'team'}, inplace=True)
    qb_adj = epa_data.merge(starter_data, on='team', how='left')
    # qb_adj.at[7,"wt_avg"] = float(qb_adj.at[7,"wt_avg_team"]) * .85
    # qb_adj.at[2,"wt_avg"] = float(qb_adj.at[2,"wt_avg_team"]) + 5
    return qb_adj