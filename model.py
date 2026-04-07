import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
import random

import os

# Load and clean data
base_path = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base_path, 'IPL_matches.csv'))
df = df.dropna(subset=['match_won_by', 'team1', 'team2', 'toss_winner'])
df['team1'] = df['team1'].astype(str)
df['team2'] = df['team2'].astype(str)
df['match_won_by'] = df['match_won_by'].astype(str)
df['toss_winner'] = df['toss_winner'].astype(str)

all_teams = sorted(list(set(df['team1'].unique()) | set(df['team2'].unique())))
df = df[df['match_won_by'].isin(all_teams)]

active_teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians', 
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru', 
    'Sunrisers Hyderabad'
]

# Encoders
le_team = LabelEncoder()
le_team.fit(all_teams)

le_toss_decision = LabelEncoder()
le_toss_decision.fit(df['toss_decision'].unique())

le_venue = LabelEncoder()
le_venue.fit(df['venue'].unique())

def prepare_features(match_df):
    X = pd.DataFrame()
    X['team1'] = le_team.transform(match_df['team1'])
    X['team2'] = le_team.transform(match_df['team2'])
    X['toss_winner'] = le_team.transform(match_df['toss_winner'])
    X['toss_decision'] = le_toss_decision.transform(match_df['toss_decision'])
    X['venue'] = le_venue.transform(match_df['venue'])
    return X

X = prepare_features(df)
y = le_team.transform(df['match_won_by'])

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("Model trained successfully.")

def predict_match(t1, t2, venue, toss_winner, toss_decision):
    teams = sorted([t1, t2])
    match_data = pd.DataFrame({
        'team1': [teams[0]],
        'team2': [teams[1]],
        'toss_winner': [toss_winner],
        'toss_decision': [toss_decision],
        'venue': [venue]
    })
    
    if venue not in le_venue.classes_:
        # Fallback to similar venue or first venue
        matches = [v for v in le_venue.classes_ if venue.split(',')[0] in v]
        venue = matches[0] if matches else le_venue.classes_[0]
        match_data['venue'] = [venue]

    X_match = prepare_features(match_data)
    
    # Force result to be one of the two playing teams
    probs = model.predict_proba(X_match)[0]
    t1_idx = le_team.transform([t1])[0]
    t2_idx = le_team.transform([t2])[0]
    
    if probs[t1_idx] >= probs[t2_idx]:
        return t1
    else:
        return t2

# Simulate IPL 2026
print("\nSimulating IPL 2026 Tournament...")
venues_map = {
    'Chennai Super Kings': 'MA Chidambaram Stadium, Chepauk, Chennai',
    'Delhi Capitals': 'Arun Jaitley Stadium, Delhi',
    'Gujarat Titans': 'Narendra Modi Stadium, Ahmedabad',
    'Kolkata Knight Riders': 'Eden Gardens, Kolkata',
    'Lucknow Super Giants': 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
    'Mumbai Indians': 'Wankhede Stadium, Mumbai',
    'Punjab Kings': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
    'Rajasthan Royals': 'Sawai Mansingh Stadium, Jaipur',
    'Royal Challengers Bengaluru': 'M Chinnaswamy Stadium, Bengaluru',
    'Sunrisers Hyderabad': 'Rajiv Gandhi International Stadium, Uppal, Hyderabad'
}

standings = {team: 0 for team in active_teams}
matchups = list(combinations(active_teams, 2))

for t1, t2 in matchups:
    for i in range(2):
        home_team = t1 if i == 0 else t2
        venue = venues_map[home_team]
        toss_winner = random.choice([t1, t2])
        toss_decision = random.choice(['bat', 'field'])
        
        winner = predict_match(t1, t2, venue, toss_winner, toss_decision)
        standings[winner] += 2

sorted_standings = sorted(standings.items(), key=lambda x: x[1], reverse=True)
print("\nLeague Stage Final Standings (Top 4):")
for i in range(min(4, len(sorted_standings))):
    print(f"{i+1}. {sorted_standings[i][0]} - {sorted_standings[i][1]} points")

top_4 = [s[0] for s in sorted_standings[:4]]
q1_winner = predict_match(top_4[0], top_4[1], venues_map[top_4[0]], top_4[0], 'field')
q1_loser = top_4[1] if q1_winner == top_4[0] else top_4[0]
el_winner = predict_match(top_4[2], top_4[3], venues_map[top_4[2]], top_4[2], 'field')
q2_winner = predict_match(q1_loser, el_winner, venues_map[q1_loser], q1_loser, 'field')
champion = predict_match(q1_winner, q2_winner, 'Narendra Modi Stadium, Ahmedabad', q1_winner, 'field')

print(f"\n*** IPL 2026 Predicted Champion: {champion} ***")
