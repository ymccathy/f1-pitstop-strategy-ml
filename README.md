# f1-pitstop-strategy-ml
Research Question: Can supervised ML models trained on historical F1 data predict whether a driver will make a pit stop within the next k laps, and can those predictions be interpreted with feature-importance methods?



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
