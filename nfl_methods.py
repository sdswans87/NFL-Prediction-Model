# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 05:10:30 2024

@author: swan0
"""
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator 
from h2o.estimators import H2ORandomForestEstimator 
from h2o.estimators.glm import H2OGeneralizedLinearEstimator 
from h2o.estimators.deeplearning import H2ODeepLearningEstimator 
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator 
import matplotlib.pyplot as plt
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import seaborn as sns


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


def source_offense_data(game_data, nfl_range): 
    off_data = source_game_data(game_data, nfl_range)
    off_data['week'] = np.where((off_data['week'] == 1) & (off_data['season'] == 2023), 19, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 2) & (off_data['season'] == 2023), 20, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 3) & (off_data['season'] == 2023), 21, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 4) & (off_data['season'] == 2023), 22, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 5) & (off_data['season'] == 2023), 23, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 6) & (off_data['season'] == 2023), 24, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 7) & (off_data['season'] == 2023), 25, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 8) & (off_data['season'] == 2023), 26, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 9) & (off_data['season'] == 2023), 27, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 10) & (off_data['season'] == 2023), 28, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 11) & (off_data['season'] == 2023), 29, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 12) & (off_data['season'] == 2023), 30, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 13) & (off_data['season'] == 2023), 31, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 14) & (off_data['season'] == 2023), 32, off_data['week'])
    off_data['week'] = np.where((off_data['week'] == 15) & (off_data['season'] == 2023), 33, off_data['week'])
    off_data['season'] = 2023
    off_data = off_data[off_data['week'] > 10]
    off_data['week'] = off_data['week'] - 10
    off_data['week'].value_counts()
    return off_data


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


def ensemble_prep_data(eff_data, columns):
    df = eff_data.drop(columns=columns)
    smp_size = int(0.80 * df.shape[0])
    np.random.seed(5)
    train_test = np.random.choice(df.index, size=smp_size, replace=False)
    df_train = df.loc[train_test]
    df_test = df.drop(train_test)
    return [df_train, df_test, df]


def ensemble_model_data(eff_data, index):
    ensemble_h2o = h2o.H2OFrame(eff_data[index])
    return ensemble_h2o


def ensemble_model_train_gbm(df, y, x, nfolds, seed):
    m = H2OGradientBoostingEstimator(nfolds=nfolds, keep_cross_validation_predictions=True, seed=seed)
    out = m.train(x=x, y=y, training_frame=df)
    return out


def ensemble_model_train_rf(df, y, x, nfolds, seed):
    m = H2ORandomForestEstimator(nfolds=nfolds, keep_cross_validation_predictions=True, seed=seed)
    out = m.train(x=x, y=y, training_frame=df)
    return out


def ensemble_model_train_lr(df, y, x, nfolds, seed):
    m = H2OGeneralizedLinearEstimator(nfolds=nfolds, keep_cross_validation_predictions=True, seed=seed)
    out = m.train(x=x, y=y, training_frame=df)
    return out


def ensemble_model_train_nn(df, y, x, nfolds, seed):
    m = H2ODeepLearningEstimator(nfolds=nfolds, keep_cross_validation_predictions=True, seed=seed)
    out = m.train(x=x, y=y, training_frame=df)
    return out


def ensemble_model_stacked_estimator(mod, mod2, mod3, mod4, df, y, x):
    m = H2OStackedEnsembleEstimator(metalearner_algorithm="glm", base_models=[mod, mod2, mod3, mod4])
    out = m.train(x=x, y=y, training_frame=df)
    return out


def ensemble_model_performance(h2o_df):
    out = H2OGradientBoostingEstimator.model_performance(h2o_df)
    return out


def ensemble_final_efficiency_data(game_data, eff_columns):
    eff_data = ensemble_prep_data(game_data, eff_columns)
    out = ensemble_model_data(eff_data, 2)
    return out


def ensemble_predictors_data(h2o_stacked, eff_data):
    predictors = h2o_stacked.predict(eff_data)
    out = predictors.as_data_frame()
    return out


def ensemble_epa_data(eff_data, predictor, field):
    epa = pd.concat([eff_data, predictor], axis=1)
    epa.columns.values[3] = field
    return epa


def ensemble_kicking_efficiency_data(game_data, side):
    kicking_df = game_data[game_data['field_goal_result'].notna()]
    kicking_df['exp_pts'] = kicking_df['fg_prob'] * 3
    kicking_df['act_pts'] = np.where(kicking_df['field_goal_result'] == 'made', 3, 0)
    kicking_df['added_pts'] = kicking_df['act_pts'] - kicking_df['exp_pts']
    out = kicking_df.groupby(['season', side]).agg(
        total_added_pts=('added_pts', 'sum'),
        total_kicks=('field_goal_result', 'count'),
        kicks_per_game=('field_goal_result', lambda x: len(x) / 17),
        avg_kick_exp_pts=('exp_pts', 'mean'),
        avg_pts_per_kick=('act_pts', 'mean'),
        add_pts_per_kick=('added_pts', 'mean'),
        add_pts_per_game=('added_pts', lambda x: sum(x) / 17)
    ).reset_index()
    return out


def ensemble_starting_kicker_data(game_data):
    kickers = game_data[game_data['field_goal_attempt'] == 1][['posteam', 'kicker_player_name', 'kicker_player_id']]
    kickers = kickers.dropna(subset=['kicker_player_id']).drop_duplicates()
    return kickers


def ensemble_adjusted_points_data(epa_data, kicking_data):
    epa_data['kick_pts_add'] = epa_data['posteam'].map(kicking_data.set_index('posteam')['add_pts_per_game'])
    epa_data['adj_pts'] = epa_data['total_effective_pts'] + epa_data['kick_pts_add']
    return epa_data


def ensemble_schedule_off_adjustment(game_data):
    sched_adj = game_data[(~game_data['epa'].isna()) & (~game_data['posteam'].isna()) & (game_data['play_type'].isin(['pass', 'run']))]
    sched_adj = sched_adj.groupby(['game_id', 'week', 'posteam']).agg(off_epa=('epa', 'mean'), off_plays=('play_type', 'count')).reset_index()
    sched_adj['off_epa'] = sched_adj.groupby('posteam')['off_epa'].transform(lambda x: x.rolling(window=14, min_periods=1).mean().shift())
    sched_adj['off_plays'] = sched_adj.groupby('posteam')['off_plays'].transform(lambda x: x.rolling(window=14, min_periods=1).mean().shift())
    sched_adj = sched_adj.sort_values('week').reset_index(drop=True)
    return sched_adj


def ensemble_schedule_def_adjustment(game_data):
    sched_adj = game_data[(~game_data['epa'].isna()) & (~game_data['defteam'].isna()) & (game_data['play_type'].isin(['pass', 'run']))]
    sched_adj = sched_adj.groupby(['game_id', 'week', 'defteam']).agg(def_epa=('epa', 'mean'), def_plays=('play_type', 'count')).reset_index()
    sched_adj['def_epa'] = sched_adj.groupby('defteam')['def_epa'].transform(lambda x: x.rolling(window=14, min_periods=1).mean().shift())
    sched_adj['def_plays'] = sched_adj.groupby('defteam')['def_plays'].transform(lambda x: x.rolling(window=14, min_periods=1).mean().shift())
    sched_adj = sched_adj.sort_values('week').reset_index(drop=True)
    return sched_adj


def ensemble_epa_lg_avg_data(epa_data):
    epa_lg_avg = epa_data.groupby('week').agg(league_mean=('epa', 'mean')).reset_index()
    epa_lg_avg['league_mean'] = epa_lg_avg['league_mean'].rolling(window=14, min_periods=1).mean().shift()
    return epa_lg_avg


def ensemble_epa_off_avg_data(off_epa, league_epa):
    epa_off = pd.merge(off_epa, league_epa, on='week', how='left').tail(32)
    epa_off['adj_off_epa'] = epa_off['off_epa'] - epa_off['league_mean']
    epa_off['adj_off_pts'] = epa_off['adj_off_epa'] * epa_off['off_plays']
    epa_off = epa_off[['posteam', 'adj_off_epa', 'adj_off_pts']]
    epa_off.columns = ['team', 'adj_off_epa', 'adj_off_pts']
    return epa_off


def ensemble_epa_def_avg_data(def_epa, league_epa):
    epa_def = pd.merge(def_epa, league_epa, on='week', how='left').tail(32)
    epa_def['adj_def_epa'] = epa_def['def_epa'] - epa_def['league_mean']
    epa_def['adj_def_pts'] = epa_def['adj_def_epa'] * epa_def['def_plays']
    epa_def = epa_def[['defteam', 'adj_def_epa', 'adj_def_pts']]
    epa_def.columns = ['team', 'adj_def_epa', 'adj_def_pts']
    return epa_def


def ensemble_sched_diff_off_data(off_data, epa_data):
    adj_off = off_data[['posteam', 'game_id']].copy()
    adj_off[['year', 'week', 'away', 'home']] = adj_off['game_id'].str.split('_', expand=True)
    adj_off['opposition'] = np.where(adj_off['posteam'] == adj_off['away'], adj_off['home'], adj_off['away'])
    adj_off = adj_off.drop(columns=['year', 'week', 'away', 'home'])
    adj_off['adjust'] = adj_off['opposition'].map(epa_data.set_index('team')['adj_off_pts'])
    adj_off = adj_off.rename(columns={'posteam': 'team'})
    adj_off = adj_off.groupby('team').agg(off_vs_avg=('adjust', 'mean'))
    return adj_off


def ensemble_sched_diff_def_data(def_data, epa_data):
    adj_def = def_data[['defteam', 'game_id']].copy()
    adj_def[['year', 'week', 'away', 'home']] = adj_def['game_id'].str.split('_', expand=True)
    adj_def['opposition'] = np.where(adj_def['defteam'] == adj_def['away'], adj_def['home'], adj_def['away'])
    adj_def = adj_def.drop(columns=['year', 'week', 'away', 'home'])
    adj_def['adjust'] = adj_def['opposition'].map(epa_data.set_index('team')['adj_def_pts'])
    adj_def = adj_def.rename(columns={'defteam': 'team'})
    adj_def = adj_def.groupby('team').agg(def_vs_avg=('adjust', 'mean'))
    return adj_def


def ensemble_weighted_offense_data(final_off, final_def_adj):
    final_def_adj = final_def_adj.reset_index()
    weighted_offense = final_off.groupby('posteam').apply(lambda x: x.rolling(window=10, on='predicted_points', win_type='boxcar').mean().iloc[-1])
    weighted_offense = weighted_offense.reset_index()[['posteam', 'predicted_points']]
    weighted_offense['sched_adj'] = weighted_offense['posteam'].map(final_def_adj.set_index('team')['def_vs_avg'])
    weighted_offense['adj_off'] = weighted_offense['predicted_points'] - weighted_offense['sched_adj']
    return weighted_offense
    

def ensemble_weighted_defense_data(final_def, final_off_adj):
    final_off_adj = final_off_adj.reset_index()
    weighted_defense = final_def.groupby('defteam').apply(lambda x: x.rolling(window=10, on='predicted_points_conceded', win_type='boxcar').mean().iloc[-1])
    weighted_defense = weighted_defense.reset_index()[['defteam', 'predicted_points_conceded']]
    weighted_defense['sched_adj'] = weighted_defense['defteam'].map(final_off_adj.set_index('team')['off_vs_avg'])
    weighted_defense['adj_def'] = weighted_defense['predicted_points_conceded'] - weighted_defense['sched_adj']
    return weighted_defense


def ensemble_qb_update_data(qb_data, weighted_off):
    adj_qb_remaining = sum(range(1, 18))
    adj_qb_gone = sum(range(9, 18))
    adj_qb_ratio = (adj_qb_remaining - adj_qb_gone) / adj_qb_remaining
    qb_data['sched_adj'] = qb_data['team'].map(weighted_off.set_index('posteam')['sched_adj'])
    qb_data['offense_update'] = qb_data.apply(lambda row: float(row['wt_avg'] * adj_qb_ratio) 
                                                            - (float(row['wt_avg_team']) - (row['sched_adj'] * adj_qb_ratio)), axis=1)
    qb_no_change = ['BUF', 'SF', 'NE', 'CIN', 'DAL', 'DET', 'JAX', 'KC', 'LAC', 'MIN',
                    'SEA', 'NYJ','CAR', 'ARI', 'PHI', 'WAS', 'GB', 'ATL', 'NO', 'LV', 
                    'CHI', 'CLE', 'HOU', 'LA', 'MIA', 'TEN', 'NYG', 'PIT', 'TB', 'DEN']
    qb_data['offense_update'] = qb_data.apply(lambda row: 0 if row['team'] in qb_no_change else row['offense_update'], axis=1)
    return qb_data


def ensemble_power_rankings_data(qb_data, weighted_off, weighted_def):
    qb_update = qb_data.drop_duplicates(subset=['team'])
    power_rank = weighted_off[['posteam']].copy()
    power_rank.columns = ['team']
    power_rank['adj_off'] = power_rank['team'].map(weighted_off.set_index('posteam')['adj_off'])
    power_rank['adj_def'] = power_rank['team'].map(weighted_def.set_index('defteam')['adj_def'])
    power_rank['pts_vs_avg'] = power_rank['adj_off'] - power_rank['adj_def']
    power_rank['qb_adj'] = power_rank['team'].map(qb_update.set_index('team')['offense_update'])
    power_rank['final_ranking'] = power_rank['pts_vs_avg'] + power_rank['qb_adj']
    power_rank.to_csv('power_rank.csv', index=False)
    return power_rank


def ensemble_hfa_data(sched_data):
    nfl_hfa = sched_data[sched_data['season'] != 2020]
    nfl_hfa = nfl_hfa[nfl_hfa['season'] != 2023] # remove
    nfl_hfa = nfl_hfa[nfl_hfa['game_type'] == 'REG']
    nfl_hfa = nfl_hfa.drop(['away_moneyline', 'home_moneyline', 'away_spread_odds', 
                            'home_spread_odds', 'old_game_id', 'gsis', 'nfl_detail_id', 
                            'pfr', 'pff', 'espn', 'over_odds', 'under_odds'], axis=1)
    home_spread = nfl_hfa.groupby('season').agg(home_line=('spread_line', 'mean'), result_avg=('result', 'mean')).reset_index()
    line_spread = sns.lmplot(data=home_spread, y='home_line', x='season', ci=None)
    line_spread.set_axis_labels('Season', 'Home Line')
    line_result = sns.lmplot(data=home_spread, y='result_avg', x='season', ci=None)
    line_result.set_axis_labels('Season', 'Result Average')
    hfa_adj = home_spread['home_line'].tail(1).values[0]
    return hfa_adj


def ensemble_curr_schedule_data(nfl_schedules):
    sched_2023 = nfl_schedules[nfl_schedules['season'] == 2023]
    return sched_2023


def predict_odds_engine(df, wk, hfa, power_rank, qb_update_2023):
    qb_update_2023 = qb_update_2023.drop_duplicates(subset=['team'])
    df = df[df['week'] == wk]
    df = df[['game_id', 'gameday', 'weekday', 'home_team', 'away_team', 'away_rest', 'home_rest', 'spread_line', 'total_line']]
    df['home_ranking'] = df['home_team'].map(power_rank.set_index('team')['final_ranking'])
    df['away_ranking'] = df['away_team'].map(power_rank.set_index('team')['final_ranking'])
    df['starting_home_qb'] = df['home_team'].map(qb_update_2023.set_index('team')['passer_player_name'])
    df['starting_away_qb'] = df['away_team'].map(qb_update_2023.set_index('team')['passer_player_name'])
    df['unregressed'] = df['home_ranking'] - df['away_ranking'] + hfa
    df['regressed_number'] = 0.5 * df['unregressed'] + 0.5 * df['spread_line']
    df.to_csv(rf'week_{wk}.csv', index=False)
    return df