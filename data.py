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


def get_race_laps(year, round_num, retry=3):
    """
    Pull clean race lap data for a single race with retry logic
    
    parameters:
    year : int
        Season year (e.g., 2023)
    round_num : int
        Round number in the season (e.g., 1 for first race)
    retry : int
        Number of retry attempts if download fails
    
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
    print(f"Fetching {year} Round {round_num}...", end=' ')
    
    for attempt in range(retry):
        try:
            # Load race session
            session = fastf1.get_session(year, round_num, 'R')
            session.load()
            
            event_name = session.event['EventName']
            print(f"→ {event_name}")
            
            laps = session.laps
            
            # extract basic lap data
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
            
            # convert LapTime to seconds
            lap_data['LapTimeSeconds'] = lap_data['LapTime'].dt.total_seconds()
            
            # create PitThisLap label
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
            
            # add SafetyCar flag
            merged_data['SafetyCar'] = 0
            
            # Sort by lap number and driver
            merged_data = merged_data.sort_values(['LapNumber', 'Driver']).reset_index(drop=True)
            
            print(f"  ✓ {len(merged_data)} laps collected")
            
            return merged_data
            
        except Exception as e:
            if attempt < retry - 1:
                print(f"  ⚠ Attempt {attempt + 1} failed, retrying...")
                import time
                time.sleep(5)  # Wait 5 seconds before retry
            else:
                print(f"  ✗ Error after {retry} attempts: {str(e)}")
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


def collect_single_year(year, num_races):
    """
    Collect all races for a single year
    
    Parameters:
    -----------
    year : int
        Season year (e.g., 2023)
    num_races : int
        Number of races in that season
    
    Returns:
    --------
    DataFrame with all races from that year
    """
    print(f"\n{'='*70}")
    print(f"  Collecting {year} Season - {num_races} Races")
    print(f"{'='*70}\n")
    
    year_data = []
    failed_races = []
    
    for round_num in range(1, num_races + 1):
        race_data = get_race_laps(year, round_num, retry=3)
        
        if race_data is not None:
            race_data = clean_race_data(race_data)
            year_data.append(race_data)
        else:
            failed_races.append(round_num)
    
    if not year_data:
        print(f"\n✗ No data collected for {year}!")
        return None
    
    # Combine all races
    combined = pd.concat(year_data, ignore_index=True)
    combined = combined.sort_values(['Round', 'Driver', 'LapNumber']).reset_index(drop=True)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  {year} Collection Summary")
    print(f"{'='*70}")
    print(f"  Races collected:  {len(year_data)} / {num_races}")
    if failed_races:
        print(f"  Failed races:     {failed_races}")
    print(f"  Total laps:       {len(combined):,}")
    print(f"  Drivers:          {combined['Driver'].nunique()}")
    print(f"  Pit stops:        {combined['PitThisLap'].sum()}")
    print(f"  Pit stop rate:    {combined['PitThisLap'].mean()*100:.2f}%")
    print(f"{'='*70}\n")
    
    return combined


def save_year_data(df, year):
    """Save single year data to CSV"""
    if df is None:
        return
    
    final_cols = [
        'Year', 'Round', 'EventName', 'Driver', 'DriverNumber', 'Team',
        'LapNumber', 'LapTimeSeconds', 'Position',
        'TireAge', 'Compound', 'Stint',
        'TrackTemp', 'AirTemp', 'Humidity', 'Pressure',
        'WindDirection', 'WindSpeed', 'Rainfall',
        'GapAhead', 'GapBehind', 'SafetyCar',
        'PitThisLap'
    ]
    
    filename = f'f1_data_{year}.csv'
    df[final_cols].to_csv(filename, index=False)
    print(f"✓ Saved to: {filename}\n")


# ============================================================================
# Collect one year at the time 
# ============================================================================

if __name__ == "__main__":
    
    # Change this to collect different years
    # Uncomment the year you want to collect:
    
    #data = collect_single_year(2021, 22)
    #save_year_data(data, 2021)
    
    #data = collect_single_year(2022, 22)
    #save_year_data(data, 2022)
    
    #data = collect_single_year(2023, 23)
    #save_year_data(data, 2023)
    
    #data = collect_single_year(2024, 24)
    #save_year_data(data, 2024)
    
    data = collect_single_year(2025, 24)
    save_year_data(data, 2025)