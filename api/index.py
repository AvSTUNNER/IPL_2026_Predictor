from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import numpy as np

app = Flask(__name__)

# Paths for assets
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, 'model_assets')

# Load assets lazily
def load_assets():
    assets = {}
    asset_files = {
        'model': 'model.pkl',
        'le_team': 'le_team.pkl',
        'le_toss_decision': 'le_toss_decision.pkl',
        'le_venue': 'le_venue.pkl',
        'team_stats': 'team_stats.pkl'
    }
    for key, filename in asset_files.items():
        path = os.path.join(ASSETS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Asset not found: {path}")
        with open(path, 'rb') as f:
            assets[key] = pickle.load(f)
    return assets

@app.route('/')
def home():
    try:
        # Check if assets exist
        status = {
            "api_status": "running",
            "assets_dir": ASSETS_DIR,
            "assets_found": os.path.exists(ASSETS_DIR),
            "files": os.listdir(ASSETS_DIR) if os.path.exists(ASSETS_DIR) else "dir not found"
        }
        # Try to load one asset to verify
        load_assets()
        status["assets_load"] = "success"
        return jsonify(status)
    except Exception as e:
        return jsonify({"api_status": "running", "error": str(e), "trace": "Check Vercel logs"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        assets = load_assets()
        data = request.json
        t1 = data['team1']
        t2 = data['team2']
        venue = data['venue']
        toss_winner = data['toss_winner']
        toss_decision = data['toss_decision']

        teams = sorted([t1, t2])
        actual_t1, actual_t2 = teams[0], teams[1]

        # Use assets
        le_team = assets['le_team']
        le_venue = assets['le_venue']
        le_toss_decision = assets['le_toss_decision']
        team_stats = assets['team_stats']
        model = assets['model']

        # Encode features
        if venue not in le_venue.classes_:
            # Fallback logic if venue not found
            fallback_venue = le_venue.classes_[0]
            venue_enc_val = le_venue.transform([fallback_venue])[0]
        else:
            venue_enc_val = le_venue.transform([venue])[0]

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
            winner = actual_t1
            win_prob = prob
        else:
            winner = actual_t2
            win_prob = 1 - prob

        return jsonify({
            'winner': winner,
            'probability': float(win_prob),
            'team1': actual_t1,
            'team2': actual_t2,
            'team1_probability': float(prob),
            'team2_probability': float(1-prob)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Metadata endpoint to help the frontend
@app.route('/metadata', methods=['GET'])
def metadata():
    try:
        assets = load_assets()
        return jsonify({
            'teams': list(assets['le_team'].classes_),
            'venues': list(assets['le_venue'].classes_),
            'toss_decisions': list(assets['le_toss_decision'].classes_)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
