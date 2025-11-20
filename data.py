"""
F1 Clean Race Data Extractor
Script to pull race lap data from FastF1 API with cleaning for ML training
Outputs clean, ready-to-train data
"""

import fastf1
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

cache_dir = 'fastf1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

def calculate_gaps(laps_df):
    """
    Calculate time gap to car ahead and behind based on lap times and positions
    """
    gaps_data = []
    
    for lap_num in laps_df['LapNumber'].unique():
        lap_data = laps_df[laps_df['LapNumber'] == lap_num].copy()
        lap_data = lap_data.sort_values('Position')
        
        for idx, row in lap_data.iterrows():
            gap_ahead = None
            gap_behind = None
            
            pos = row['Position']
            
            # Gap to car ahead
            if pd.notna(pos) and pos > 1:
                ahead = lap_data[lap_data['Position'] == pos - 1]
                if not ahead.empty and pd.notna(row['LapTime']) and pd.notna(ahead.iloc[0]['LapTime']):
                    gap_ahead = (row['LapTime'] - ahead.iloc[0]['LapTime']).total_seconds()
            
            # Gap to car behind
            if pd.notna(pos):
                behind = lap_data[lap_data['Position'] == pos + 1]
                if not behind.empty and pd.notna(row['LapTime']) and pd.notna(behind.iloc[0]['LapTime']):
                    gap_behind = (behind.iloc[0]['LapTime'] - row['LapTime']).total_seconds()
            
            gaps_data.append({
                'Driver': row['Driver'],
                'LapNumber': lap_num,
                'GapAhead': gap_ahead,
                'GapBehind': gap_behind
            })
    
    return pd.DataFrame(gaps_data)


def get_race_laps(year, round_num):
    """
    Pull clean race lap data for a single race
    
    parameters:
    year : int
        Season year (e.g., 2023)
    round_num : int
        Round number in the season (e.g., 1 for first race)
    
    returns:
    DataFrame with columns:
        - Year, Round, EventName, Driver, DriverNumber, Team
        - LapNumber, LapTimeSeconds, Position
        - TireAge, Compound, Stint
        - TrackTemp, AirTemp, Humidity, Pressure, WindDirection, WindSpeed, Rainfall
        - GapAhead, GapBehind
        - SafetyCar (0 or 1) 
        - PitThisLap (label: 1 if pit on this lap, 0 otherwise)
    """
    print(f"Fetching {year} Round {round_num}...")
    
    try:
        # Load race session
        session = fastf1.get_session(year, round_num, 'R')
        session.load()
        
        event_name = session.event['EventName']
        print(f"  → {event_name}")
        
        laps = session.laps
        
        # extratc basic lap data
        lap_data = pd.DataFrame({
            'Year': year,
            'Round': round_num,
            'EventName': event_name,
            'Driver': laps['Driver'],
            'DriverNumber': laps['DriverNumber'],
            'Team': laps['Team'],
            'LapNumber': laps['LapNumber'],
            'LapTime': laps['LapTime'],
            'Position': laps['Position'],
            'TireAge': laps['TyreLife'],
            'Compound': laps['Compound'],
            'Stint': laps['Stint'],
            'PitInTime': laps['PitInTime'],
            'PitOutTime': laps['PitOutTime']
        })
        
        # it convert LapTime to seconds (for training --> important in ML)
        lap_data['LapTimeSeconds'] = lap_data['LapTime'].dt.total_seconds()
        
        # it creates PitThisLap label (1 if driver pits on this lap, 0 otherwise)
        lap_data['PitThisLap'] = lap_data['PitInTime'].notna().astype(int)
        
        # get weather data per lap
        weather = laps.get_weather_data()
        weather_data = pd.DataFrame({
            'TrackTemp': weather['TrackTemp'],
            'AirTemp': weather['AirTemp'],
            'Humidity': weather['Humidity'],
            'Pressure': weather['Pressure'],
            'WindDirection': weather['WindDirection'],
            'WindSpeed': weather['WindSpeed'],
            'Rainfall': weather['Rainfall']
        })
        
        # merge lap data with weather
        lap_data = lap_data.reset_index(drop=True)
        weather_data = weather_data.reset_index(drop=True)
        merged_data = pd.concat([lap_data, weather_data], axis=1)
        
        # calculate gaps between cars
        gaps = calculate_gaps(merged_data)
        merged_data = pd.merge(
            merged_data,
            gaps,
            on=['Driver', 'LapNumber'],
            how='left'
        )
        
        # add SafetyCar flag (simplified ?
        merged_data['SafetyCar'] = 0
        
        # Sort by lap number and driver (CRITICAL for time series)
        merged_data = merged_data.sort_values(['LapNumber', 'Driver']).reset_index(drop=True)
        
        print(f"  ✓ {len(merged_data)} laps collected")
        
        return merged_data
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}\n")
        return None



def clean_race_data(df):
    """
    this function clean the race data for ML training:
    - Remove invalid laps
    - Handle outliers
    - Remove laps with critical missing values
    - Keep only valid race laps
    """

    print(f"\n  Cleaning data...")
    initial_rows = len(df)
    
    # Remove laps with missing critical data
    df_clean = df.copy()
    
    # Must have lap time
    df_clean = df_clean[df_clean['LapTimeSeconds'].notna()]
    
    # Must have tire data
    df_clean = df_clean[df_clean['TireAge'].notna()]
    df_clean = df_clean[df_clean['Compound'].notna()]
    
    # Must have position
    df_clean = df_clean[df_clean['Position'].notna()]
    
    # Remove outlier lap times (likely invalid laps)
    # F1 lap times typically 80-130 seconds, filter out obvious errors
    df_clean = df_clean[df_clean['LapTimeSeconds'] > 70]  # Remove pit laps and errors
    df_clean = df_clean[df_clean['LapTimeSeconds'] < 200]  # Remove extremely slow laps
    
    # Remove invalid tire ages (tires don't last 100 laps)
    df_clean = df_clean[df_clean['TireAge'] < 100]
    
    # Forward fill weather data within each race (weather updates ~once per minute)
    weather_cols = ['TrackTemp', 'AirTemp', 'Humidity', 'Pressure', 
                    'WindDirection', 'WindSpeed', 'Rainfall']
    for col in weather_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean.groupby(['Year', 'Round'])[col].ffill()
    
    # Fill remaining missing gaps with 0 (car in first/last position)
    df_clean['GapAhead'] = df_clean['GapAhead'].fillna(0)
    df_clean['GapBehind'] = df_clean['GapBehind'].fillna(0)
    
    removed = initial_rows - len(df_clean)
    print(f"  ✓ Removed {removed} invalid laps ({removed/initial_rows*100:.1f}%)")
    print(f"  ✓ {len(df_clean)} clean laps remaining")
    
    return df_clean


def collect_multiple_races(year, rounds, clean=True):
    """
    Collect data from multiple races
    
    Parameters:
    year : int
        Season year
    rounds : list of int
        List of round numbers to collect
    clean : bool
        If True, apply data cleaning for ML training
    
    Returns: DataFrame with all race laps combined
    """
    all_races = []
    
    print(f"\n{'='*60}")
    print(f"Collecting F1 Race Data: {year} Season")
    print(f"{'='*60}\n")
    
    for round_num in rounds:
        race_data = get_race_laps(year, round_num)
        if race_data is not None:
            if clean:
                race_data = clean_race_data(race_data)
            all_races.append(race_data)
    
    if not all_races:
        print("No data collected!")
        return None
    
    # Combine all races
    combined = pd.concat(all_races, ignore_index=True)
    
    # Sort by Year, Round, Driver, LapNumber (CRITICAL for LSTM)
    combined = combined.sort_values(
        ['Year', 'Round', 'Driver', 'LapNumber']
    ).reset_index(drop=True)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Collection Complete!")
    print(f"{'='*60}")
    print(f"Total laps:    {len(combined):,}")
    print(f"Races:         {combined['Round'].nunique()}")
    print(f"Drivers:       {combined['Driver'].nunique()}")
    print(f"Teams:         {combined['Team'].nunique()}")
    print(f"Pit stops:     {combined['PitThisLap'].sum()}")
    print(f"Pit stop rate: {combined['PitThisLap'].mean()*100:.2f}%")
    print(f"{'='*60}\n")
    
    return combined


def save_data(df, filename='f1_race_data.csv'):
    """
    Save clean data to CSV with summary info
    """
    if df is None:
        print("No data to save!")
        return
    
    # Select final columns for ML (drop raw PitInTime/PitOutTime)
    final_cols = [
        'Year', 'Round', 'EventName', 'Driver', 'DriverNumber', 'Team',
        'LapNumber', 'LapTimeSeconds', 'Position',
        'TireAge', 'Compound', 'Stint',
        'TrackTemp', 'AirTemp', 'Humidity', 'Pressure',
        'WindDirection', 'WindSpeed', 'Rainfall',
        'GapAhead', 'GapBehind', 'SafetyCar',
        'PitThisLap'
    ]
    
    df_final = df[final_cols].copy()
    
    # Save to CSV
    df_final.to_csv(filename, index=False)
    print(f"✓ Data saved to: {filename}")
    
    # Data quality report
    print(f"\n{'='*60}")
    print("Data Quality Report")
    print(f"{'='*60}")
    
    # Check for missing values
    missing = df_final.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values:")
        for col in missing[missing > 0].index:
            pct = missing[col] / len(df_final) * 100
            print(f"  - {col:20s}: {missing[col]:,} ({pct:.1f}%)")
    else:
        print("\n✓ No missing values!")
    
    # Check data ranges
    print("\nData ranges:")
    print(f"  LapTimeSeconds:  {df_final['LapTimeSeconds'].min():.1f} - {df_final['LapTimeSeconds'].max():.1f}")
    print(f"  TireAge:         {df_final['TireAge'].min():.0f} - {df_final['TireAge'].max():.0f}")
    print(f"  Position:        {df_final['Position'].min():.0f} - {df_final['Position'].max():.0f}")
    print(f"  TrackTemp:       {df_final['TrackTemp'].min():.1f}°C - {df_final['TrackTemp'].max():.1f}°C")
    
    # Class distribution
    print("\nTarget variable (PitThisLap):")
    print(f"  No pit (0):  {(df_final['PitThisLap']==0).sum():,} ({(df_final['PitThisLap']==0).mean()*100:.2f}%)")
    print(f"  Pit (1):     {(df_final['PitThisLap']==1).sum():,} ({(df_final['PitThisLap']==1).mean()*100:.2f}%)")
    
    # Tire compounds
    print("\nTire compounds:")
    for compound, count in df_final['Compound'].value_counts().items():
        print(f"  {compound}: {count:,}")
    
    print(f"{'='*60}\n")
    
    # show sample
    print("test sample of cleaned data:")
    print(df_final[['Driver', 'LapNumber', 'Position', 'TireAge', 'Compound', 
                     'LapTimeSeconds', 'GapAhead', 'PitThisLap']].head(5))
    
    return df_final



# USAGE
if __name__ == "__main__":
    # to get single race data 
    # race_data = get_race_laps(year=2024, round_num=1)
    # race_data = clean_race_data(race_data)
    # save_data(race_data, 'bahrain_2024_clean.csv')
    
    # we used the first two races data from 2024 for milestone 1
    data = collect_multiple_races(
        year=2024,
        rounds=[1, 2],
        clean=True  # Set to False if you want raw data
    )
    save_data(data, 'f1_race_data_2024_clean.csv')
    
    # Full season 2024
    # data = collect_multiple_races(
    #     year=2024,
    #     rounds=list(range(1, 25)),  # 2024 has 24 races
    #     clean=True
    # )
    # save_data(data, 'f1_race_data_2024_full_clean.csv')
    
    # Combined seasons 2022-2024 
    # all_data = []
    # for year in [2022, 2023, 2024]:
    #     season_data = collect_multiple_races(
    #         year=year, 
    #         rounds=list(range(1, 23)),
    #         clean=True
    #     )
    #     if season_data is not None:
    #         all_data.append(season_data)
    # 
    # combined = pd.concat(all_data, ignore_index=True)
    # save_data(combined, 'f1_race_data_2022_2024_clean.csv')