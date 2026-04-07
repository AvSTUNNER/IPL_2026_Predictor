

# IPL 2026 Winner Prediction Model (Advanced)

This project uses historical IPL data (2008-2025) to predict the winner of the 2026 season. It features robust feature engineering, multiple machine learning models, vigorous evaluation, and an end-to-end tournament simulation.

## Project Structure
- `IPL.csv`: Raw ball-by-ball dataset (not included in repo due to size).
- `IPL_matches.csv`: Cleaned match-level dataset generated from raw data.
- `advanced_model.py`: Python script containing the ML pipeline, evaluations, and tournament simulation.
- `plots/`: Directory containing generated visualization charts.
- `requirements.txt`: Python dependencies.

## Advanced Methodology
1. **Feature Engineering**: 
   - Calculated **historical win rates** for every team before simulation to represent current "form".
   - Engineered toss success (`is_toss_winner`), encoded venues, and historical head-to-head parameters.
2. **Model Selection & Evaluation**: 
   - Trained **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.
   - Evaluated rigorously using **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.
   - Automatically selects the best model based on the highest ROC-AUC score (Random Forest).
3. **Visualizations**: 
   - Generates a **Model Comparison Chart** (`model_comparison.png`).
   - Generates **Feature Importance Chart** (`feature_importance.png`).
   - Generates **Confusion Matrix** for the best model (`confusion_matrix.png`).
4. **Tournament Simulation**: Simulated a full 10-team round-robin league stage and standard playoffs. Output includes precise **Win Probabilities** for high-stakes matches.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the advanced simulation: `python advanced_model.py`

## Predicted Winner (2026)
Based on current historical trends, venue-specific performance, and the selected Random Forest model, the prediction for the IPL 2026 champion is **Chennai Super Kings** with a 52.9% win probability in the finals against Gujarat Titans.
