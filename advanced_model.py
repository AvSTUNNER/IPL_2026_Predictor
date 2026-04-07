import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from itertools import combinations
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

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

# 3. Model Training & Vigorous Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'ROC-AUC': roc}

results_df = pd.DataFrame(results).T
print("\n--- Vigorous Model Evaluation ---")
print(results_df.round(3))

# Select Best Model (based on ROC-AUC)
best_model_name = results_df['ROC-AUC'].idxmax()
best_model = models[best_model_name]
print(f"\nSelected Best Model: {best_model_name}")

# 4. Visualizations
plots_dir = os.path.join(script_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# A. Model Comparison Plot
results_df[['Accuracy', 'ROC-AUC', 'F1-Score']].plot(kind='bar', figsize=(10, 6))
plt.title('Model Comparison Metrics')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'model_comparison.png'))
plt.close()

# B. Feature Importance Plot
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importances ({best_model_name})')
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    plt.close()
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importances ({best_model_name})')
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    plt.close()

# C. Confusion Matrix
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Team 2 Win', 'Team 1 Win'], yticklabels=['Team 2 Win', 'Team 1 Win'])
plt.title(f'Confusion Matrix ({best_model_name})')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
plt.close()

print(f"\nVisualizations saved in '{plots_dir}' directory.")

# 5. Tournament Simulation (2026)
active_teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians', 
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru', 
    'Sunrisers Hyderabad'
]

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

def predict_match_prob(t1, t2, venue, toss_winner, toss_decision, model):
    teams = sorted([t1, t2])
    actual_t1, actual_t2 = teams[0], teams[1]
    
    venue_enc_val = le_venue.transform([venue])[0] if venue in le_venue.classes_ else le_venue.classes_[0]
    toss_dec_enc_val = le_toss_decision.transform([toss_decision])[0]
    is_toss_win = 1 if toss_winner == actual_t1 else 0
    t1_win_rate = team_stats.get(actual_t1, 0.5)
    t2_win_rate = team_stats.get(actual_t2, 0.5)
    
    match_data = pd.DataFrame({
        'team1_enc': [le_team.transform([actual_t1])[0]],
        'team2_enc': [le_team.transform([actual_t2])[0]],
        'venue_enc': [venue_enc_val],
        'toss_decision_enc': [toss_dec_enc_val],
        'is_toss_winner': [is_toss_win],
        'team1_historical_win_rate': [t1_win_rate],
        'team2_historical_win_rate': [t2_win_rate]
    })
    
    prob = model.predict_proba(match_data)[0][1] 
    
    if prob >= 0.5:
        return actual_t1, prob, actual_t2, 1-prob
    else:
        return actual_t2, 1-prob, actual_t1, prob

print("\n--- Simulating IPL 2026 Tournament ---")
standings = {team: 0 for team in active_teams}
matchups = list(combinations(active_teams, 2))

for t1, t2 in matchups:
    for i in range(2):
        home_team = t1 if i == 0 else t2
        venue = venues_map[home_team]
        toss_winner = random.choice([t1, t2])
        toss_decision = random.choice(['bat', 'field'])
        
        winner, win_prob, loser, lose_prob = predict_match_prob(t1, t2, venue, toss_winner, toss_decision, best_model)
        standings[winner] += 2

sorted_standings = sorted(standings.items(), key=lambda x: x[1], reverse=True)
print("\nLeague Stage Final Standings (Top 4):")
for i in range(4):
    print(f"{i+1}. {sorted_standings[i][0]} - {sorted_standings[i][1]} points")

top_4 = [s[0] for s in sorted_standings[:4]]

print("\n--- IPL 2026 Playoffs ---")
q1_winner, q1_prob, q1_loser, _ = predict_match_prob(top_4[0], top_4[1], venues_map[top_4[0]], top_4[0], 'field', best_model)
print(f"Qualifier 1: {top_4[0]} vs {top_4[1]} -> {q1_winner} wins ({q1_prob:.1%} probability)")

el_winner, el_prob, _, _ = predict_match_prob(top_4[2], top_4[3], venues_map[top_4[2]], top_4[2], 'field', best_model)
print(f"Eliminator : {top_4[2]} vs {top_4[3]} -> {el_winner} wins ({el_prob:.1%} probability)")

q2_winner, q2_prob, _, _ = predict_match_prob(q1_loser, el_winner, venues_map[q1_loser], q1_loser, 'field', best_model)
print(f"Qualifier 2: {q1_loser} vs {el_winner} -> {q2_winner} wins ({q2_prob:.1%} probability)")

final_venue = 'Narendra Modi Stadium, Ahmedabad'
champion, champ_prob, runner_up, _ = predict_match_prob(q1_winner, q2_winner, final_venue, q1_winner, 'field', best_model)
print(f"FINAL      : {q1_winner} vs {q2_winner} at {final_venue}")
print(f"*** IPL 2026 CHAMPION: {champion} ({champ_prob:.1%} win probability) ***")