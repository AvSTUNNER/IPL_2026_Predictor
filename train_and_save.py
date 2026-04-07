import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

# 1. Load Data
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, 'IPL_matches.csv'))

# Basic cleaning
df = df.dropna(subset=['match_won_by', 'team1', 'team2', 'toss_winner', 'date', 'venue'])
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

df['team1'] = df['team1'].astype(str)
df['team2'] = df['team2'].astype(str)
df['match_won_by'] = df['match_won_by'].astype(str)

all_teams = sorted(list(set(df['team1'].unique()) | set(df['team2'].unique())))
df = df[df['match_won_by'].isin(all_teams)]

# 2. Advanced Feature Engineering (Historical Win Rates)
team_stats = {}
for team in all_teams:
    matches_played = len(df[(df['team1'] == team) | (df['team2'] == team)])
    wins = len(df[df['match_won_by'] == team])
    win_rate = wins / matches_played if matches_played > 0 else 0.5
    team_stats[team] = win_rate

df['team1_historical_win_rate'] = df['team1'].map(team_stats)
df['team2_historical_win_rate'] = df['team2'].map(team_stats)

# Toss win match win correlation
df['is_toss_winner'] = (df['team1'] == df['toss_winner']).astype(int)

# Target variable (1 if team1 wins, 0 if team2 wins)
df['target'] = (df['match_won_by'] == df['team1']).astype(int)

# Encoders
le_team = LabelEncoder()
le_team.fit(all_teams)
le_toss_decision = LabelEncoder()
le_toss_decision.fit(df['toss_decision'].unique())
le_venue = LabelEncoder()
le_venue.fit(df['venue'].unique())

df['team1_enc'] = le_team.transform(df['team1'])
df['team2_enc'] = le_team.transform(df['team2'])
df['toss_decision_enc'] = le_toss_decision.transform(df['toss_decision'])
df['venue_enc'] = le_venue.transform(df['venue'])

# Select features
features = ['team1_enc', 'team2_enc', 'venue_enc', 'toss_decision_enc', 'is_toss_winner', 'team1_historical_win_rate', 'team2_historical_win_rate']
X = df[features]
y = df['target']

# 3. Model Training
# Using GradientBoostingClassifier as it was the best in advanced_model.py
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X, y)

# 4. Save Assets
assets_dir = os.path.join(script_dir, 'model_assets')
os.makedirs(assets_dir, exist_ok=True)

with open(os.path.join(assets_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(assets_dir, 'le_team.pkl'), 'wb') as f:
    pickle.dump(le_team, f)

with open(os.path.join(assets_dir, 'le_toss_decision.pkl'), 'wb') as f:
    pickle.dump(le_toss_decision, f)

with open(os.path.join(assets_dir, 'le_venue.pkl'), 'wb') as f:
    pickle.dump(le_venue, f)

with open(os.path.join(assets_dir, 'team_stats.pkl'), 'wb') as f:
    pickle.dump(team_stats, f)

print(f"Model and assets saved in {assets_dir}")
