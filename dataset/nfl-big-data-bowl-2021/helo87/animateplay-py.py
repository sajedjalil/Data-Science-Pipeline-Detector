import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import numpy as np 
import pandas as pd

import field_py as field
import helper_py as h

class AnimatePlay:
    def __init__(self, gameId, playId, df_tracks, defense):
        # Home = offense, away = defense from here on out
        # TODO: rewrite to be def and off instead of home and away
        self.LINE_WIDTH = 2
        self.df_tracks = df_tracks
        self.df_plays = pd.read_csv('../input/nfl-big-data-bowl-2021/plays.csv')
        self.play_df = h.slice_frame(self.df_tracks, playId, gameId)
        self.play_meta_df = h.slice_frame(self.df_plays, playId, gameId)
        self.frames = self.play_df['frameId'].unique()

        self.df_games = pd.read_csv('../input/nfl-big-data-bowl-2021/games.csv')
        df_game = self.df_games[self.df_games['gameId'].isin([gameId])]
        possession_team = h.slice_frame(self.df_plays, playId, gameId)['possessionTeam'].values[0]
        defense = defense
        if defense == 'home':
            offense = 'away'
        else:
            offense = 'home'
        
        self.xy_columns = ['x','y']

        self.play = h.slice_frame(self.play_meta_df, playId, gameId)
        self.title = self.play['playDescription'].squeeze()
        self.fig, self.ax = field.create_football_field(plt_or_fig='fig', hashes=True);
        # set home to be offense and away to be defense so that colors will be according to possession
        self.df_offense, self.num_offense = self.get_team_info(self.play_df, offense)
        self.df_defense, self.num_defense = self.get_team_info(self.play_df, defense)
        # TODO: Make this function
        self.df_football, _ = self.get_team_info(self.play_df, 'football')
        self.player_names = self.df_offense['displayName'].unique().tolist() + \
                            self.df_defense['displayName'].unique().tolist() + ['Football']
        self.marker_offense = self.get_marker('mediumslateblue', 'midnightblue')
        self.marker_defense = self.get_marker('mediumseagreen', 'aquamarine')
        self.marker_football = self.get_marker('sandybrown', 'peachpuff')
        self.positions_offense = self.get_labels(self.num_offense, 'black')
        self.positions_defense = self.get_labels(self.num_defense, 'black')
        self.lines_defense = self.get_lines(self.num_defense, 'mediumseagreen')
        self.lines_offense = self.get_lines(self.num_offense, 'mediumslateblue')
        self.lines_football = self.get_lines(1, 'chocolate')
        self.anim = animation.FuncAnimation(self.fig, 
                                self.animate, 
                                init_func=self.anim_init,
                                frames=len(self.frames), 
                                interval=100, 
                                blit=True)

    # Create trajectories of objects in supplied xy coords
    def get_lines(self, N, color):
        lines = []
        for index in range(N):
            lobj = self.ax.plot([],[],lw=self.LINE_WIDTH,color=color)[0]
            lines.append(lobj)
        return lines

    # Create empty marker for animation with defined styles
    def get_marker(self, face_color, edge_color):
        marker, = self.ax.plot([], [], 
                  lw=self.LINE_WIDTH, 
                  markerfacecolor=face_color, 
                  markeredgecolor=edge_color, 
                  linestyle='None', 
                  marker='o', 
                  markersize=30, 
                  zorder=100, 
                  alpha=1.0,)
        return marker

    def get_team_info(self, df, team):
        df_team = df[df['team'] == team]
        if team != 'football':
            num_players = len(df_team['displayName'].unique())
        else:
            num_players = 1
        return df_team[['x', 'y', 'o','frameId','position','displayName', 'nflId']], num_players

    # Create empty positions for text labels with defined styles
    def get_labels(self, N, color):
        positions = []
        for index in range(N):
            pobj = self.ax.text([],[],'', fontsize=12, color=color, zorder=100)
            positions.append(pobj)
        return positions
    
    # initialization function for animation: plot the background of each frame
    def anim_init(self):
        for line in self.lines_offense+self.lines_defense+self.lines_football:
            line.set_data([], [])
        self.marker_offense.set_data([], [])
        self.marker_football.set_data([], [])
        self.marker_defense.set_data([], [])
        for position in self.positions_offense+self.positions_defense:
            position.set_text('')
        return tuple(self.lines_offense+self.lines_defense+self.lines_football+[self.marker_offense]+[self.marker_defense]+[self.marker_football]+self.positions_offense+self.positions_defense) 

    # animation function. This is called sequentially
    def animate(self, i):
        frameId = self.frames[i]
        df_offense_i = self.df_offense[self.df_offense['frameId'].isin(self.frames[:i+1])].sort_values('frameId')
        df_defense_i = self.df_defense[self.df_defense['frameId'].isin(self.frames[:i+1])].sort_values('frameId')
        df_football_i = self.df_football[self.df_football['frameId'].isin(self.frames[:i+1])].sort_values('frameId')
        df_football_i['position'] = ''
        df_players_i = pd.concat([df_offense_i, df_defense_i, df_football_i])
        
        player_lines = self.lines_offense + self.lines_defense + self.lines_football
        player_positions = self.positions_offense + self.positions_defense
        for (idx, player) in enumerate(self.player_names):
            df_ = df_players_i[df_players_i['displayName'].isin([player])].sort_values('frameId')
            xy_ = np.array(df_[self.xy_columns])
            player_lines[idx].set_data(xy_[:,0], xy_[:,1])
            if player is not 'Football':
                player_positions[idx].set_text(df_.position.unique()[0])
                player_positions[idx].set_x(xy_[-1,0]-1.25)
                player_positions[idx].set_y(xy_[-1,1]-0.60)
            
        xy_offense_i = np.array(df_offense_i[df_offense_i['frameId'].isin([self.frames[i]])][self.xy_columns])
        self.marker_offense.set_data(xy_offense_i[:,0], xy_offense_i[:,1])
        xy_defense_i = np.array(df_defense_i[df_defense_i['frameId'].isin([self.frames[i]])][self.xy_columns])
        self.marker_defense.set_data(xy_defense_i[:,0], xy_defense_i[:,1])
        xy_football_i = np.array(df_football_i[df_football_i['frameId'].isin([self.frames[i]])][self.xy_columns])
        self.marker_football.set_data(xy_football_i[:,0], xy_football_i[:,1])

        return tuple(self.lines_offense+self.lines_defense+self.lines_football+[self.marker_offense]+[self.marker_defense]+[self.marker_football]+self.positions_offense+self.positions_defense)