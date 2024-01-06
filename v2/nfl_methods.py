# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 05:10:30 2024

@author: swan0
"""
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def import_game_data(start_range, end_range):
    df = nfl.import_pbp_data(range(start_range, end_range))
    return df


def import_schedule_data(start_range, end_range):
    df = nfl.import_schedules(range(start_range, end_range))
    return df


def import_rookie_data(start_range):
    df = nfl.import_draft_picks()
    df = df[df['season'] > start_range]
    return df


def import_quarterback_data():
    df = nfl.import_players()
    return df


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


def source_quarterback_data(quarterback_data):
    df = quarterback_data[quarterback_data['status'] == "ACT"]
    df = df[df['position'] == 'QB']
    df = df[(df['status'] == 'ACT') | (df['status'] == 'INA')]
    df = df[['status', 'team_abbr', 'position', 'first_name', 'last_name', 'gsis_id']]
    df.rename(columns={'gsis_id': 'passer_player_id'}, inplace=True)
    return df


def source_rookie_data(rookie_data):
    df = rookie_data[rookie_data['position'] == 'QB']
    df = df[~df['gsis_id'].isna()]
    df = df[df['round'] == 1]
    df = df[['gsis_id', 'season', 'pfr_player_name']]
    df.rename(columns={'gsis_id': 'passer_player_id'}, inplace=True)
    return df


def source_rookie_pass_data(game_data, rookie_data):
    df = game_data[game_data['play_type'] == 'pass']
    df = df[df['passer_player_id'].isin(rookie_data['passer_player_id'])]  
    return df


def source_rookie_run_data(game_data, rookie_data):
    df = game_data[game_data['play_type'] == 'run']
    df = df[df['rusher_player_id'].isin(rookie_data['passer_player_id'])]  
    return df


def epa_cumulative_data(game_data, pos_columns):
    df = game_data.groupby(['passer_player_id', 'passer_player_name']) \
                  .apply(lambda x: x.assign(qb_sum_epa = x['qb_epa'].sum())) \
                  .reset_index(drop=True) \
                  .loc[:, pos_columns] \
                  .dropna(subset=['passer_player_name']) \
                  .drop_duplicates() \
                  .dropna() 
    return df


def epa_season_data(game_data):
    df = game_data.groupby(['passer_player_id', 'season']) \
                  .apply(lambda x: x.assign(qb_sum_epa = x['qb_epa'].sum())) \
                  .reset_index(drop=True) \
                  .loc[:, ['season','passer_player_id','passer_player_name','posteam','qb_sum_epa']] \
                  .dropna(subset=['passer_player_name']) \
                  .drop_duplicates() \
                  .dropna()
    return df


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


def epa_pass_efficiency_data(game_data, side):
    df = game_data[game_data['pass'] == 1].groupby(['game_id', side]).agg(
                   pass_epa_game=('epa', 'sum'), 
                   n_pass=('pass', 'count'), 
                   pass_epa_dropback=('epa', 'mean'),
                   succ_pass_pct=('success', 'mean')
                   ).reset_index().dropna()
    return df


def epa_rush_efficiency_data(game_data, side):
    df = game_data[game_data['pass'] == 0].groupby(['game_id', side]).agg(
                   rush_epa_game=('epa', 'sum'),
                   n_rush=('rush', 'count'),
                   rush_epa_dropback=('epa', 'mean'),
                   succ_rush_pct=('success', 'mean')
                   ).reset_index().dropna()
    return df


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


def rookie_play_data(pass_df, run_df):
    rookie_plays = pd.concat([pass_df, run_df])
    rookie_plays = rookie_plays.groupby(['season','game_id', 'passer_player_id']).apply(
                                            lambda x: x['epa'].sum()).reset_index()
    rookie_plays.columns = ['season','game_id', 'passer_player_id', 'qb_epa']
    rookie_plays = rookie_plays.drop_duplicates()
    return rookie_plays


def rookie_epa_data(rookie_df, rookie_plays):
    rookie_epa = pd.merge(rookie_df, rookie_plays, on=['passer_player_id', 'season'], how='left')
    return rookie_epa


def rookie_mean_data(rookie_df):
    rookie_mean = rookie_df['qb_epa'].mean()
    return rookie_mean


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


def qb_starter_rankings_data(qb_starters, qb_rankings):
    df = pd.merge(qb_starters, qb_rankings, on=['passer_player_name','passer_player_id', 'posteam'])
    df = df.reset_index(drop=True)
    return df


def qb_rankings_adj_data(epa_data, starter_data, rookie_mean):
    starter_data.rename(columns={'posteam': 'team'}, inplace=True)
    qb_adj = epa_data.merge(starter_data, on='team', how='left')
    # qb_adj.at[7,"wt_avg"] = float(qb_adj.at[7,"wt_avg_team"]) * .85
    # qb_adj.at[2,"wt_avg"] = float(qb_adj.at[2,"wt_avg_team"]) + 5
    return qb_adj


def game_efficiency_data(game_data, epa_data):
    qb_plays = game_data.groupby('passer_player_id').size().reset_index(name='n_passes')
    qb_plays = qb_plays.dropna()
    qb_plays = qb_plays.drop_duplicates()
    epa_data['n_passes'] = epa_data['passer_player_id'].map(qb_plays.set_index('passer_player_id')['n_passes'])
    out = epa_data[epa_data['n_passes'] > 10].copy()
    out['epa_play'] = out['qb_sum_epa'] / out['n_passes']
    return out


def game_efficiency_by_season_data(game_data, epa_data):
    qb_plays = game_data.groupby(['passer_player_id', 'passer_player_name', 'season']).size().reset_index(name='n_passes')
    qb_plays = qb_plays.dropna(subset=['passer_player_name'])
    qb_plays = qb_plays.drop_duplicates()
    out = pd.merge(epa_data, qb_plays, on=['season', 'passer_player_id', 'passer_player_name'], how='left')
    out = out[out['n_passes'] > 10]
    out = out.dropna()
    out['epa_play'] = out['qb_sum_epa'] / out['n_passes']
    return out


def game_efficiency_ball_side_data(game_data, side):
    efficiency = game_data.groupby(['game_id', side]).agg(game_epa=('epa', 'sum'), 
                                                            avg_cpoe=('cpoe', 'mean'),
                                                            sum_cpoe=('cpoe', 'sum')).reset_index()
    efficiency = efficiency.dropna()
    return efficiency


def game_efficiency_pass_data(game_data, side):
    pass_eff = game_data[game_data['pass'] == 1].groupby(['game_id', side]).agg(
                pass_epa_game=('epa', 'sum'), 
                n_pass=('pass', 'count'), 
                pass_epa_dropback=('epa', 'mean'),
                succ_pass_pct=('success', 'mean')
                ).reset_index().dropna()
    return pass_eff


def game_efficiency_run_data(game_data, side):
    run_eff = game_data[game_data['pass'] == 0].groupby(['game_id', side]).agg(
                rush_epa_game=('epa', 'sum'),
                n_rush=('rush', 'count'),
                rush_epa_dropback=('epa', 'mean'),
                succ_rush_pct=('success', 'mean')
                ).reset_index().dropna()
    return run_eff


def game_efficiency_pass_run_data(pass_data, run_data, side):
    pos_epa = pd.merge(pass_data, run_data, on=['game_id', side], how='left')
    pos_epa = pos_epa.dropna()
    pos_epa['n_plays'] = pos_epa['n_pass'] + pos_epa['n_rush']
    pos_epa[['year', 'week', 'away_team', 'home_team']] = pos_epa['game_id'].str.split('_', expand=True)
    temp_dict = {"OAK":"LV", "SD":"LAC"}
    for key,value in temp_dict.items():
        pos_epa.loc[pos_epa["home_team"] == key, "home_team"] = value
        pos_epa.loc[pos_epa["away_team"] == key, "away_team"] = value
    return pos_epa


def game_efficiency_pts_game_data(game_data, side):
    game_data['eff_fg_pts'] = game_data['fg_prob'] * 3 * game_data['field_goal_attempt']
    pts_per_game = game_data.groupby(['game_id', side]).agg(
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
    pts_per_game = pts_per_game.loc[pts_per_game[side].notna()].drop_duplicates()
    return pts_per_game


def game_efficiency_combined_data(eff_data, points_data, side):
    eff_points = pd.merge(eff_data, points_data, on=['game_id', side, 'home_team', 'away_team'], how='left')
    if side == 'posteam':
        eff_points['poss_score'] = pd.NA
        for row in range(len(eff_points)):
            if eff_points['posteam'][row] == eff_points['home_team'][row]:
                eff_points['poss_score'][row] = eff_points['home_final'][row]
            else:
                eff_points['poss_score'][row] = eff_points['away_final'][row]
    if side == 'defteam':
        eff_points['score_allowed'] = pd.NA
        for row in range(len(eff_points)):
            if eff_points['defteam'][row] == eff_points['home_team'][row]:
                eff_points['score_allowed'][row] = eff_points['home_final'][row]
            else:
                eff_points['score_allowed'][row] = eff_points['away_final'][row]
    return eff_points


def game_efficiency_epa_data(game_data, epa_data, side):
    epa = game_data[game_data['cpoe'].notna()]
    epa = epa.groupby(['game_id', side]).agg(avg_cpoe=('cpoe', 'mean'), total_cpoe=('cpoe', 'sum')).reset_index()
    out = pd.merge(epa_data, epa, on=['game_id', side], how='left').drop_duplicates()
    return out


def game_efficiency_pass_epa_graph_data(epa_data):
    plt.hist(epa_data['pass_epa_game'], bins=range(-40, 41, 1))
    plt.xlim(-40, 40)
    plt.title('Pass EPA Per Game')
    plt.show()
    

def game_efficiency_run_epa_graph_data(epa_data):
    plt.hist(epa_data['rush_epa_game'], bins=range(-25, 26, 1))
    plt.xlim(-25, 25)
    plt.title('Run EPA Per Game')
    plt.show()


def game_efficiency_down_data(game_data, dwn, pass_run, side):
    down = game_data[game_data['down'] == dwn]
    down_eff = down[down['play_type'] == pass_run].groupby(['game_id', side]).agg(
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


def game_efficiency_all_down_data(df1, df2, df3, df4, df5, df6, side):
    all_down_eff = df1.merge(df2, on = ['game_id',side], how = 'left') 
    all_down_eff = all_down_eff.merge(df3, on = ['game_id',side], how = 'left') 
    all_down_eff = all_down_eff.merge(df4, on = ['game_id',side], how = 'left') 
    all_down_eff = all_down_eff.merge(df5, on = ['game_id',side], how = 'left') 
    all_down_eff = all_down_eff.merge(df6, on = ['game_id',side], how = 'left') 
    return all_down_eff


def game_efficiency_total_data(df, df2, side):
    out = pd.merge(df, df2, on=['game_id', side], how='left')
    out['pass_rate'] = out['n_pass'] / out['n_plays']
    out['run_rate'] = 1 - out['pass_rate'] 
    out['pass_rate_first'] = out['n_play_pass_1'] / out['n_play_pass_1'] + out['n_play_run_1']
    out['pass_rate_second'] = out['n_play_pass_2'] / out['n_play_pass_2'] + out['n_play_run_2']
    return out