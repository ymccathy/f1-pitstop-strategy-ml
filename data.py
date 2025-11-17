"""
F1 Clean Race Data Extractor
Script to pull race lap data from FastF1 API with needed features
No preprocessing - just raw, clean data
"""

import fastf1
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Setup cache
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
    
    Parameters:
    -----------
    year : int
        Season year (e.g., 2023)
    round_num : int
        Round number in the season (e.g., 1 for first race)
    
    Returns:
    --------
    DataFrame with columns:
        - Year, Round, EventName, Driver, DriverNumber, Team
        - LapNumber, LapTime, Position
        - TireAge, Compound, Stint
        - TrackTemp, AirTemp, Humidity, Pressure, WindDirection, WindSpeed, Rainfall
        - GapAhead, GapBehind
        - SafetyCar (0 or 1)
        - PitInTime, PitOutTime (for creating labels later)
    """
    print(f"Fetching {year} Round {round_num}...")
    
    try:
        # Load race session
        session = fastf1.get_session(year, round_num, 'R')
        session.load()
        
        event_name = session.event['EventName']
        print(f"  → {event_name}")
        
        # Get all laps
        laps = session.laps
        
        # Extract basic lap data
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
        
        # Get weather data per lap
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
        
        # Merge lap data with weather
        lap_data = lap_data.reset_index(drop=True)
        weather_data = weather_data.reset_index(drop=True)
        merged_data = pd.concat([lap_data, weather_data], axis=1)
        
        # Calculate gaps between cars
        gaps = calculate_gaps(merged_data)
        merged_data = pd.merge(
            merged_data,
            gaps,
            on=['Driver', 'LapNumber'],
            how='left'
        )
        
        # Add SafetyCar flag (simplified - can be enhanced)
        merged_data['SafetyCar'] = 0
        
        # Sort by lap number and driver
        merged_data = merged_data.sort_values(['LapNumber', 'Driver']).reset_index(drop=True)
        
        print(f"  ✓ {len(merged_data)} laps collected\n")
        
        return merged_data
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}\n")
        return None


def collect_multiple_races(year, rounds):
    """
    Collect data from multiple races
    
    Parameters:
    -----------
    year : int
        Season year
    rounds : list of int
        List of round numbers to collect
    
    Returns:
    --------
    DataFrame with all race laps combined
    """
    all_races = []
    
    print(f"\n{'='*60}")
    print(f"Collecting F1 Race Data: {year} Season")
    print(f"{'='*60}\n")
    
    for round_num in rounds:
        race_data = get_race_laps(year, round_num)
        if race_data is not None:
            all_races.append(race_data)
    
    if not all_races:
        print("No data collected!")
        return None
    
    # Combine all races
    combined = pd.concat(all_races, ignore_index=True)
    
    # Summary
    print(f"{'='*60}")
    print(f"Collection Complete!")
    print(f"{'='*60}")
    print(f"Total laps:    {len(combined):,}")
    print(f"Races:         {combined['Round'].nunique()}")
    print(f"Drivers:       {combined['Driver'].nunique()}")
    print(f"Teams:         {combined['Team'].nunique()}")
    print(f"{'='*60}\n")
    
    return combined


def save_data(df, filename='f1_race_data.csv'):
    """
    Save data to CSV with summary info
    """
    if df is None:
        print("No data to save!")
        return
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"✓ Data saved to: {filename}")
    
    # Print column info
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  - {col:20s} ({non_null:,} non-null values)")
    
    # Show sample
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df



# USAGE 

if __name__ == "__main__":
    
    # race_data = get_race_laps(year=2023, round_num=1)
    # save_data(race_data, 'bahrain_2023.csv')
    
    # First two races from 2024 
    data = collect_multiple_races(
        year=2024,
        rounds=[1, 2]  # First 3 races of 2023
    )
    save_data(data, 'f1_race_data_2024.csv')
    
    # Full season
    # data = collect_multiple_races(
    #     year=2024,
    #     rounds=list(range(1, 23))  # All 22 races
    # )
    # save_data(data, 'f1_race_data_2024_full.csv')
    
    # Combined season 2022-24 
    # all_data = []
    # for year in [2022, 2023, 2024]:
    #     season_data = collect_multiple_races(year, list(range(1, 23)))
    #     if season_data is not None:
    #         all_data.append(season_data)
    # combined = pd.concat(all_data, ignore_index=True)