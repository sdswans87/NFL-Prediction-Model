# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:08:57 2023

@author: swan0
"""
import numpy as np
import pandas as pd
import time

class epa_data_class():
    def __init__(self, import_obj):

        # start runtime
        start_run = time.time()

        # create close game data past eight years
        self.close_game_data = self.clean_data_func(import_obj.nfl_data, range(2016,2024))

        # create active quarterbacks
        self.active_qbs = self.active_qb_func(import_obj.quarterback_data)

        # create starting quarterbacks
        self.starting_qbs = self.starting_qb_func(self.close_game_data, season=2023, week=15)

        # calculate cumulative epa 
        self.cumulative_epa = self.cumulative_epa_func(self.close_game_data, "posteam")
        self.cumulative_epa_allowed = self.cumulative_epa_func(self.close_game_data, "defteam")

        # calculate season epa past 8 seasons
        self.season_epa = self.season_epa_func(self.close_game_data)
        self.season_epa_allowed = self.season_epa_func(self.close_game_data)

        # calculate passing value epa
        self.passing_value_epa = self.passing_value_epa_func(import_obj.nfl_data, range(2021,2024))

        # create quarterback rankings
        self.qb_rankings = self.qb_rankings_func(self.passing_value_epa)

        # create starting qbs 
        self.starter_rankings = self.starter_rankings_func(self.starting_qbs, self.qb_rankings)

        # create qbs by team
        self.team_epa = self.team_epa_func(self.passing_value_epa[1])

        # rookie impact
        self.first_rd_qb = self.first_rd_qb_func(import_obj.rookie_data)  

        # rookie pass value
        self.rookie_pass = self.rookie_pass_func(import_obj.nfl_data, self.first_rd_qb)

        # rookie run value
        self.rookie_run = self.rookie_run_func(import_obj.nfl_data, self.first_rd_qb)

        # rookie play value
        self.rookie_play = self.rookie_play_func(self.rookie_pass, self.rookie_run)

        # rookie season epa
        self.rookie_epa = self.rookie_epa_func(self.first_rd_qb, self.rookie_play)

        # rookie average epa
        self.rookie_mean = self.rookie_mean_func(self.rookie_epa)
        
        # qb rankings adjusments (add next years rookie data here)
        self.qb_rankings_adj = self.qb_rankings_adj_func(self.team_epa, self.starter_rankings, self.rookie_mean)

        # end runtime
        end_run = time.time()
        end_run = (end_run - start_run)/60
        print("epa data runtime: " + str(end_run) + " minutes.")


    def clean_data_func(self, nfl_df, nfl_range): 
        
        # filter to selected seasons 
        clean_df = nfl_df.loc[(nfl_df['season'] >= min(nfl_range)) & 
                              (nfl_df['season'] <= max(nfl_range))]
        
        # fix aaron rodgers name
        clean_df["passer_player_name"] = clean_df["passer_player_name"].apply(
                                        lambda x: "A.Rodgers" if x == "Aa.Rodgers" else x)
        
        # drop columns 
        clean_df = clean_df.drop(columns=['nflverse_game_id', 'possession_team','offense_formation', 
                                          'offense_personnel', 'defenders_in_box','defense_personnel',
                                          'number_of_pass_rushers', 'players_on_play', 'offense_players',
                                          'defense_players','n_offense', 'n_defense'])
        
        # exclude playoffs, qb_kneel, qb_spike
        clean_df = clean_df[clean_df['season_type'] == 'REG']
        clean_df = clean_df[clean_df['qb_kneel'] == 0]
        clean_df = clean_df[clean_df['qb_spike'] == 0]
        
        # create score delta
        clean_df['current_score_differential'] = clean_df['posteam_score'] - clean_df['defteam_score']
        
        # create blowout column
        clean_df['blow_out'] = np.where((clean_df['qtr'] == 4) & (np.abs(clean_df['current_score_differential'])
                                                                  > 13.5), 1, 0)
        clean_df['blow_out'] = np.where((clean_df['qtr'] > 2) & (abs(clean_df['current_score_differential'])
                                                                 > 27.5), 1, clean_df['blow_out'])
        return clean_df
    

    def active_qb_func(self, nfl_data):
        qbs_2023 = nfl_data[nfl_data['status'] == "ACT"]
        qbs_2023 = qbs_2023[qbs_2023['position'] == 'QB']
        qbs_2023 = qbs_2023[(qbs_2023['status'] == 'ACT') | (qbs_2023['status'] == 'INA')]
        qbs_2023 = qbs_2023[['status', 'team_abbr', 'position', 'first_name', 'last_name', 'gsis_id']]
        qbs_2023.rename(columns={'gsis_id': 'passer_player_id'}, inplace=True)
        return qbs_2023


    def starting_qb_func(self, nfl_data, season, week):
        qbs_2023 = nfl_data[nfl_data['season'] == season]
        qbs_2023 = qbs_2023[qbs_2023['week'] == week]
        qbs_2023 = qbs_2023.groupby(['passer_player_id', 'passer_player_name', 'posteam']) \
                                     .apply(lambda x: x.assign(sum_epa = x['epa'].sum())) \
                                     .loc[:, ['passer_player_name', 'passer_player_id', 'posteam']] \
                                     .dropna(subset=['passer_player_name']) \
                                     .drop_duplicates() \
                                     .dropna() \
                                     .reset_index(drop=True) 
        qbs_2023 = qbs_2023.drop([21,6,13,16,19,31,23,26,27,29,34,39,44])
        qbs_2023 = qbs_2023.reset_index(drop=True)
        qbs_2023.iloc[0] = ["T.Heinicke", "00-0031800", "ATL"]
        qbs_2023.iloc[11] = ["R.Tannehill", "00-0029701", "TEN"]
        qbs_2023.iloc[23] = ["G.Smith", "00-0030565", "SEA"]
        return qbs_2023
    

    def cumulative_epa_func(self, nfl_df, pos_side):
        if pos_side == "posteam":
            columns = ['passer_player_id', 'passer_player_name', 'posteam', 'qb_sum_epa']
        if pos_side == "defteam":
            columns = ['passer_player_id', 'passer_player_name', 'posteam', 'defteam','qb_sum_epa']
        df = nfl_df.groupby(['passer_player_id', 'passer_player_name']) \
                   .apply(lambda x: x.assign(qb_sum_epa = x['qb_epa'].sum())) \
                   .reset_index(drop=True) \
                   .loc[:, columns] \
                   .dropna(subset=['passer_player_name']) \
                   .drop_duplicates() \
                   .dropna() 
        return df
    

    def season_epa_func(self, nfl_df):
        df = nfl_df.groupby(['passer_player_id', 'season']) \
                   .apply(lambda x: x.assign(qb_sum_epa = x['qb_epa'].sum())) \
                   .reset_index(drop=True) \
                   .loc[:, ['season','passer_player_id','passer_player_name','posteam','qb_sum_epa']] \
                   .dropna(subset=['passer_player_name']) \
                   .drop_duplicates() \
                   .dropna()
        return df
    

    def passing_value_epa_func(self, nfl_df, nfl_range):
        
        # clean qb data
        qb_df = self.clean_data_func(nfl_df, nfl_range)

        # create passers dataframe
        passers = pd.DataFrame(qb_df['passer_player_id'])
        passers.columns = ['IDs']
        passers = passers[~passers['IDs'].isna()]
        passers = passers.groupby('IDs').size().reset_index(name='passes_thrown')
        
        # filter less than 50 completions
        passers = passers[passers['passes_thrown'] > 50]
        passers = passers.drop(columns=['passes_thrown'])
        passers = passers.drop_duplicates()
    
        # set pass/run dataframes
        pass_df = qb_df[qb_df['play_type'] == 'pass']
        run_df = qb_df[(qb_df['play_type'] == 'run') & (qb_df['rusher_player_id'].isin(passers['IDs']))]
    
        # concat pass/run
        out = pd.concat([pass_df, run_df])
        out['passer_player_id'] = np.where(out['play_type'] == 'run', out['rusher_player_id'], 
                                           out['passer_player_id'])
        
        # create qb epa
        out2 = out.groupby(['game_id', 'passer_player_id', 'passer_player_name', 'posteam', 
                            'season', 'week']).agg(qb_sum_epa=('epa', 'sum')).reset_index()
        out2 = out2[~out2['passer_player_name'].isna()].drop_duplicates()

        # count qb games played
        qb_count = out2.groupby('passer_player_id').size().reset_index(name='game_count')
        out2['game_count'] = out2['passer_player_id'].map(qb_count.set_index('passer_player_id')['game_count'])

        # create team epa
        out3 = out.groupby(['game_id','posteam','season','week']).agg(team_qb_epa=('epa','sum')).reset_index()
        out3 = out3.drop_duplicates()
        return [out2, out3]
    

    def qb_rankings_func(self, quarterback_df): 

        # groupby passer player id
        qb_n = quarterback_df[0].groupby('passer_player_id').size().reset_index(name='games')

        # create qb list 
        qb_list = pd.DataFrame({'passer_player_id': quarterback_df[0]['passer_player_id'].unique()})
        qb_list.columns = ['passer_player_id']
        qb_list = qb_list.merge(qb_n, on='passer_player_id', how='left')
        qb_list.columns = ['passer_player_id', 'games']
        qb_list = qb_list[qb_list['games'] >= 4]

        # calculate qb rankings
        player_ids = qb_list["passer_player_id"].unique()
        out = pd.DataFrame()
        for temp_ids in player_ids:
            df2 = quarterback_df[0][quarterback_df[0]['passer_player_id'] == temp_ids]
            df2 = df2.tail(17)
            x = min(15, len(df2) - 2)
            df2['wt_avg'] = df2["qb_sum_epa"].rolling(x).mean()
            out = out.append(df2.tail(1))
        out = out[['posteam', 'passer_player_id', 'passer_player_name', 'wt_avg']]
        return out
    

    def starter_rankings_func(self, quarterback_df, rankings_df):
        df = pd.merge(quarterback_df, rankings_df, on=['passer_player_name','passer_player_id', 'posteam'])
        df = df.reset_index(drop=True)
        adds = pd.DataFrame({'passer_player_name': ['E.Stick'],
                             'passer_player_id': ["00-0035282"],
                             'posteam': ["LAC"],
                             "wt_avg": [float(-6.021435)]})
        df = pd.concat([df, adds])
        return df
    

    def team_epa_func(self, nfl_df):
        tms = nfl_df['posteam'].unique()
        out = pd.DataFrame()
        df = nfl_df[nfl_df['season'] == 2022]
        for tm in tms:
            df2 = df[df['posteam'] == tm]
            x = df2.shape[0] - 2
            df2['wt_avg_team'] = df2["team_qb_epa"].rolling(x).mean()
            out = pd.concat([out, df2.tail(1)])
        out = out[['posteam', 'wt_avg_team']]
        out.columns = ['team', 'wt_avg_team']
        return out
    

    def first_rd_qb_func(self, nfl_df):
        first_rd_qb = nfl_df[nfl_df['season'] > 2002]
        first_rd_qb = first_rd_qb[first_rd_qb['position'] == 'QB']
        first_rd_qb = first_rd_qb[~first_rd_qb['gsis_id'].isna()]
        first_rd_qb = first_rd_qb[first_rd_qb['round'] == 1]
        first_rd_qb = first_rd_qb[['gsis_id', 'season', 'pfr_player_name']]
        first_rd_qb.rename(columns={'gsis_id': 'passer_player_id'}, inplace=True)
        return first_rd_qb
    

    def rookie_pass_func(self, nfl_df, rookie_df):
        rookie_pass = nfl_df[nfl_df['play_type'] == 'pass']
        rookie_pass = rookie_pass[rookie_pass['passer_player_id'].isin(rookie_df['passer_player_id'])]  
        return rookie_pass
    

    def rookie_run_func(self, nfl_df, rookie_df):
        rookie_run = nfl_df[nfl_df['play_type'] == 'run']
        rookie_run = rookie_run[rookie_run['rusher_player_id'].isin(rookie_df['passer_player_id'])]  
        return rookie_run
    

    def rookie_play_func(self, pass_df, run_df):
        rookie_plays = pd.concat([pass_df, run_df])
        rookie_plays = rookie_plays.groupby(['season','game_id', 'passer_player_id']).apply(
                                              lambda x: x['epa'].sum()).reset_index()
        rookie_plays.columns = ['season','game_id', 'passer_player_id', 'qb_epa']
        rookie_plays = rookie_plays.drop_duplicates()
        return rookie_plays
    

    def rookie_epa_func(self, rookie_df, rookie_plays):
        rookie_epa = pd.merge(rookie_df, rookie_plays, on=['passer_player_id', 'season'], how='left')
        return rookie_epa
    

    def rookie_mean_func(self, rookie_df):
        rookie_mean = rookie_df['qb_epa'].mean()
        return rookie_mean
    

    def qb_rankings_adj_func(self, team_epa, starters, rookie_mean):
        starters.rename(columns={'posteam': 'team'}, inplace=True)
        qb_update_2023 = team_epa.merge(starters, on='team', how='left')
        qb_update_2023.at[7,"wt_avg"] = float(qb_update_2023.at[7,"wt_avg_team"]) * .85
        qb_update_2023.at[2,"wt_avg"] = float(qb_update_2023.at[2,"wt_avg_team"]) + 5
        return qb_update_2023