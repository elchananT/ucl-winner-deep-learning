# Champions League winner model

This project is an honest **season-level probability prototype**, not a claim that a neural network can foretell football. It combines historical Champions League results with season/team statistics, prevents future-season leakage, and reports probability calibration.

## Data sources

Download Kaggle CSVs into `data/` using the links documented in the notebook. The history data must include season, winner and team fields; team-stat data needs season, team and numeric performance features. Kaggle dataset schemas differ, so the small mapping cell makes those columns explicit.

Run `pip install -r requirements.txt` then `jupyter lab champions_league_winner.ipynb`.
