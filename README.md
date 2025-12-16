# To Pit or Not To Pit? F1 Pit-Stop Prediction with Machine Learning
Team: Sylvia, Cathy  
Research Question: Can supervised machine learning models use lap-by-lap Formula 1 race data to predict when a driver will pit during a race? 

# Overview
This project explores whether machine learning models can predict when a Formula 1 driver will pit during a race using lap-by-lap race data. Pitstops are rare, high-impact strategic decisions influenced by evolving race conditions such as tire wear, pace changes, weather, and Safety Cars. We gather 120,000+ laps data, frame pitstop prediction as a sequential time-series classification problem and compare a static baseline (Logistic Regression) with sequence models (LSTMs implemented in PyTorch and Keras). We find that sequence-based models substantially outperform the static baseline, and predictive performance varies meaningfully across race contexts: pitstops are most predictable in rainy conditions, for certain teams, on specific tracks, and pit-window prediction is strong (F1 = 0.80 for a 5-lap window). Overall, lap-sequence information is crucial for modeling pit-stop behavior.

# Replication Instruction
### Detailed instructions
1. Environment setup: Install dependencies using pip install -r requirements.txt.
2. Dataset prep: The cleaned dataset f1_race_data_2021_2025_final.csv is provided and ready to use for training. Simply load and train models directly
   To add additional seasons or future races, modify the year parameter in data.py (e.g., data = collect_single_year(2026, 24) for the 2026 season) and run the script to collect new
   data. After collecting all desired years, run merge_all_years.py to combine all year files into a refreshed dataset (e.g., f1_race_data_2021_2026_final.csv). This allows extension of
   the dataset as new F1 seasons become available.
3. Model Training + Evaluation: Open train.ipynb. Each model section is fully runnable.
   A. Logistic Regression (Baseline): Runs on per-lap features only, Outputs ROC-AUC and PR-AUC, Expected results: ROC–AUC ≈ 0.88 & PR–AUC ≈ 0.26
   B. PyTorch LSTM Model: Data is grouped by driver-race and padded, Masks are generated for variable-length sequences, LSTMs output a pit probability per lap, Expected results:ROC–AUC ≈ 0.97 & PR–AUC ≈ 0.79
   C. Keras LSTM Model: Masking, LSTM(units=...), Dense(1, activation="sigmoid"), Expected results: ROC–AUC ≈ 0.97 & PR–AUC ≈ 0.78

4. Recreating Poster Metrics & Figures: Inside the notebook, run the evaluation cells. We have weather-based metrics, with expected output rate by F1 score: Rain > Mixed > Dry.
  we also have team & track metrics. sort by F1: Best teams: Aston Martin, Racing Bulls, Mercedes
Best tracks: Chinese GP, Russian GP, Italian GP
Worst tracks: Hungarian, British, US GP
   and our Pit-Window Evaluation which uses our window function, and Pit Window = next 5 laps. Expected: Precision = 0.86, Recall = 0.75, F1 = 0.80

# Future Directions
There are several extensions to improve both realism and performance of our model. First, the model could be trained directly on pit-window targets (e.g., PitNextK) instead of applying window logic only during evaluation, allowing it to better align with real strategic decision horizons. Second, alternative sequence models such as GRUs or attention-based architectures may help identify which laps and signals most strongly influence pit decisions while improving interpretability; prior work on lap-level Formula 1 data suggests GRUs can outperform LSTMs due to their simpler gating and improved efficiency on noisy sequential datasets (Piccolomini, Evangelista, & Rondelli, 2023). Finally, incorporating competitor-aware features (such as recent competitor pit activity and pit-timing undercut/overcut indicators) would better capture race-wide strategic interactions and move the model closer to real Formula 1 decision-support systems.


# Contributions
Sylvia Guo and Cathy Chen collaborated closely and split the work evenly throughout this project. 

Both brainstormed the project topic and research direction together, spending 8+ hours researching the prior work, refining the problem scope, selecting appropriate models, deciding relevant features, and determining evaluation metrics. This included framing pitstop prediction as a sequential learning problem and identifying suitable performance measures for rare events.

Cathy focused on data collection and preprocessing, implementing a FastF1 API pipeline to gather, clean, and merge over 120,000 laps of race data across multiple seasons. She also addressed technical challenges related to multi-year data extraction, summarized the project’s findings, and explored future directions. This stage took approximately 8 hours.

Sylvia focused on baseline logistic regression model setup and evaluation, built PyTorch LSTM pipeline with masking + padded sequences, debugged data shapes, masking, training loops, and metrics. Also led track/team characteristic analysis and figure styling. This stage took about 10 hours. 

Both team members contributed extensively to training and refining the sequence models, for example adjusting training epochs and implementing LSTM models using Keras. It took them around 8 hours. They also worked together on model evaluation across race characteristics (weather, team, track) and on pit-window prediction, which took approximately 4 hours. Both substantially to the final poster assembly and interpretation of results as well as poster design. 

# Reference

Ansel, J., Yang, E., He, H., Gimelshein, N., Jain, A., Voznesensky, M., Bao, B., Bell, P., Berard, D., Burovski, E., Chauhan, G., Chourdia, A., Constable, W., Desmaison, A., DeVito, Z., Ellison, E., Feng, W., Gong, J., Gschwind, M., … Chintala, S. (2024). PyTorch 2: Faster machine learning through dynamic Python bytecode transformation and graph compilation. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS ’24), Volume 2. ACM. https://doi.org/10.1145/3620665.3640366
Bonomi, A., Turri, E., & Iacca, G. (2023). Evolutionary F1 race strategy [Manuscript]. University of Trento. https://iris.unitn.it/retrieve/3f8f46bc-fd61-47ff-a1df-0880379b82e8/f1race_optimization.pdf
Bunker, R. P., & Thabtah, F. (2019). A machine learning framework for sport result prediction. Applied Computing and Informatics, 15(1), 27–33. https://doi.org/10.1016/j.aci.2017.09.005
Chollet, F., et al. (2015). Keras. https://keras.io
FastF1. (n.d.). FastF1. https://theoehrly.github.io/Fast-F1/
Formula1 Dictionary. (2023). Race strategy. https://www.formula1-dictionary.net/strategy_race.html
Piccolomini, E. L., Evangelista, D., & Rondelli, M. (2023). The future of Formula 1 racing: Neural networks to predict tyre strategy [Manuscript]. https://amslaurea.unibo.it/27922/1/Tesi_Massimo_Rondelli.pdf
Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). Dive into deep learning. Cambridge University Press. https://d2l.ai
