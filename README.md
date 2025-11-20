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