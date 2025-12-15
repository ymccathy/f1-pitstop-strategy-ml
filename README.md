# To Pit or Not To Pit? F1 Pit-Stop Prediction with Machine Learning
Sylvia Guo & Cathy Chen

Research Question: Can supervised machine learning models use lap-by-lap Formula 1 race data to predict when a driver will pit during a race? 

# Overview

### Brief
This project explores whether machine learning models can predict when a Formula 1 driver will pit during a race using lap-by-lap race data. Pitstops are rare, high-impact strategic decisions influenced by evolving race conditions such as tire wear, pace changes, weather, and Safety Cars. We frame pitstop prediction as a sequential time-series classification problem and compare a static baseline (Logistic Regression) with sequence models (LSTMs implemented in PyTorch and Keras). We find that sequence-based models substantially outperform the static baseline, and predictive performance varies meaningfully across race contexts: pitstops are most predictable in rainy conditions, for certain teams, on specific tracks, and pit-window prediction is strong (F1 = 0.80 for a 5-lap window). Overall, lap-sequence information is crucial for modeling pit-stop behavior.

# Replication Instruction
### Detailed instructions of how to replicate the results in your poster
1. Environment setup: Install dependencies using pip install -r requirements.txt.
2. Dataset prep: The cleaned dataset f1_race_data_2021_2025_final.csv is provided and ready to use for training. Simply load and train models directly
   To add additional seasons or future races, modify the year parameter in data.py (e.g., data = collect_single_year(2026, 24) for the 2026 season) and run the script to collect new
   data. After collecting all desired years, run merge_all_years.py to combine all year files into a refreshed dataset (e.g., f1_race_data_2021_2026_final.csv). This allows extension of
   the dataset as new F1 seasons become available.
3. Model Training + Evaluation: Open train.ipynb. Each model section is fully runnable.
   A. Logistic Regression (Baseline): Runs on per-lap features only, Outputs ROC-AUC and PR-AUC, Expected results: ROC–AUC ≈ 0.88 & PR–AUC ≈ 0.26
   B. PyTorch LSTM Model: Data is grouped by driver-race and padded, Masks are generated for variable-length sequences, LSTMs output a pit probability per lap, Expected results:ROC–AUC ≈ 0.97 & PR–AUC ≈ 0.79
   C. Keras LSTM Model: Masking, LSTM(units=...), Dense(1, activation="sigmoid"), Expected results: ROC–AUC ≈ 0.97 & PR–AUC ≈ 0.78

4. Recreating Poster Metrics & Figures: Inside the notebook, run the evaluation cells. We have weather-based metrics, with expected output rate by F1 score: Rain > Dry > Mixed.
  we also have team & track metrics. sort by F1: Best teams: Aston Martin, Racing Bulls, Mercedes
Best tracks: Chinese GP, Russian GP, Italian GP
Worst tracks: Hungarian, British, US GP
   and our Pit-Window Evaluation which uses our window function, and Pit Window = next 5 laps. Expected: Precision = 0.86, Recall = 0.75, F1 = 0.80

# Future Directions
### Brief
There are several extensions to improve both realism and performance of our model. First, the model could be trained directly on pit-window targets (e.g., PitNextK) instead of applying window logic only during evaluation, allowing it to better align with real strategic decision horizons. Second, alternative sequence models such as GRUs or attention-based architectures may help identify which laps and signals most strongly influence pit decisions while improving interpretability; prior work on lap-level Formula 1 data suggests GRUs can outperform LSTMs due to their simpler gating and improved efficiency on noisy sequential datasets. Finally, incorporating competitor-aware features (such as gaps to nearby cars, recent competitor pit activity, and pit-timing undercut/overcut indicators) would better capture race-wide strategic interactions and move the model closer to real Formula 1 decision-support systems.


# Contributions
Sylvia Guo and Cathy Chen collaborated closely and split the work evenly throughout this project. 

Both brainstormed the project topic and research direction together, spending 8+ hours researching the prior work, refining the problem scope, selecting appropriate models, deciding relevant features, and determining evaluation metrics. This included framing pitstop prediction as a sequential learning problem and identifying suitable performance measures for rare events.

Cathy focused on data collection and preprocessing, implementing the FastF1 API pipeline to gather, clean, and merge over 120,000 laps of race data across multiple seasons. She also addressed technical challenges related to multi-year data extraction. This stage took approximately 8 hours.

Sylvia focused on baseline logistic regression model setup and evaluation, built PyTorch LSTM pipeline with masking + padded sequences, debugged data shapes, masking, training loops, and metrics. Also led track/team characteristic analysis and figure styling. This stage took about 10 hours. 

Both team members contributed extensively to training and refining the sequence models, for example adjusting training epochs and implementing LSTM models using Keras. It took them around 8 hours. They also worked together on model evaluation across race characteristics (weather, team, track) and on pit-window prediction, which took approximately 4 hours. Both substantially to the final poster assembly and interpretation of results as well as poster design. 





## Summary of `data.py`

**Purpose:** it extracts clean, ready-to-train F1 race data from the FastF1 API for pit stop prediction.

**What it does:**
1. **Pulls race lap data** from FastF1 API for specified seasons/rounds
2. **Extracts features:** Lap times, tire age, compound, position, weather (temp, humidity, wind), gaps to cars ahead/behind
3. **Creates label:** `PitThisLap` (1 if driver pits on that lap, 0 otherwise)
4. **Cleans data:**
   - Removes invalid laps (missing data, outlier lap times)
   - Converts lap times to seconds
   - Forward-fills weather data
   - Fills missing gaps with 0
5. **Sorts data** by Year → Round → Driver → LapNumber (critical for LSTM time series)
6. **Outputs CSV** with 22 clean columns, sorted and ready for ML training

**Output:** A single CSV file with all race laps, features, and pit stop labels so that we do not need preprocessing needed. We can just train on top of it. 

## Summary of `train.ipynb`

**Purpose**: we build and evaluate supervised models (logistic regression and LSTM) to predict whether a driver will pit on a given lap using the cleaned race-lap dataset.

**What we did**:

1. **Prepared model inputs**

   a. Loaded the cleaned CSV output from data.py

   b. Split data by driver into train, validation, and test sets

   c. Standardized numeric features using a scaler fit on the training set

   d. Formed race-long sequences (driver by race) for the LSTM model

2. **Trained Logistic Regression (baseline)**

   a. Uses only per-lap features (no sequence structure)

   b. Fast to train and easy to interpret

   c. strong performance:

      - ROC-AUC: ~0.92

      - PR-AUC: ~0.23

3. Built and trained LSTM sequence model

   a. Implemented custom PyTorch Dataset and collate_fn

   b. Padded sequences and created masks for real lap entries

   c. Used pack_padded_sequence to handle variable sequence lengths

   d. Output a probability for each lap in the sequence

   e. After debugging masking and shape mismatches, the LSTM trained correctly

4. Final performance:

   - ROC-AUC: ~0.58

   - PR-AUC: ~0.04

We evaluated both models and found that logistic regression clearly outperformed the LSTM model

For this dataset, most pit-stop signal is already captured by static lap-level features (tire age, lap-time drop, compound, gaps). The LSTM does not gain additional predictive power from full race sequences, likely due to the rarity of pit laps, noisy lap-to-lap patterns, and missing team strategy context.

**Second step ideas**:

- expand dataset with more races

- Try GRUs, temporal CNNs, or attention models
