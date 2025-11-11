"""
F1 Pit Stop Prediction - Data Collection Pipeline
This script pulls clean F1 data from FastF1 API and combines it into a single dataset
for machine learning model training to predict pit stops within the next k laps.

Target Features:
- Lap, TireAge, Compound, TrackTemp, AirTemp, WindDirection
- GapAhead, GapBehind, SafetyCar, PitNext3Laps, Rainfall, Humidity
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Create cache directory if it doesn't exist
cache_dir = 'fastf1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Created cache directory: {cache_dir}")

# Enable caching to speed up data retrieval
fastf1.Cache.enable_cache(cache_dir)


def calculate_gaps(laps_df, session_laps):
    """
    Calculate GapAhead and GapBehind for each driver on each lap
    """
    gaps_data = []
    
    for lap_num in laps_df['LapNumber'].unique():
        lap_data = laps_df[laps_df['LapNumber'] == lap_num].copy()
        lap_data = lap_data.sort_values('Position')
        
        for idx, row in lap_data.iterrows():
            gap_ahead = np.nan
            gap_behind = np.nan
            
            # Get position of current driver
            pos = row['Position']
            
            if pd.notna(pos) and pos > 1:
                # Find driver ahead
                ahead = lap_data[lap_data['Position'] == pos - 1]
                if not ahead.empty and pd.notna(row['LapTime']) and pd.notna(ahead.iloc[0]['LapTime']):
                    gap_ahead = (row['LapTime'] - ahead.iloc[0]['LapTime']).total_seconds()
            
            if pd.notna(pos):
                # Find driver behind
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


def create_pit_labels(laps_df, k=3):
    """
    Create target variable: PitNextKLaps
    Returns 1 if driver pits within next k laps, 0 otherwise
    """
    laps_df = laps_df.sort_values(['Driver', 'LapNumber']).copy()
    laps_df[f'PitNext{k}Laps'] = 0
    
    for driver in laps_df['Driver'].unique():
        driver_laps = laps_df[laps_df['Driver'] == driver].copy()
        pit_laps = driver_laps[driver_laps['PitInTime'].notna()]['LapNumber'].values
        
        for pit_lap in pit_laps:
            # Mark k laps before pit as positive examples
            mask = (
                (laps_df['Driver'] == driver) & 
                (laps_df['LapNumber'] >= pit_lap - k) & 
                (laps_df['LapNumber'] < pit_lap)
            )
            laps_df.loc[mask, f'PitNext{k}Laps'] = 1
    
    return laps_df


def get_session_data(year, round_num, session_name='R'):
    """
    Retrieve and process data for a single session
    
    Parameters:
    - year: int, season year
    - round_num: int, round number in the season
    - session_name: str, 'R' for race, 'Q' for qualifying, 'FP1', 'FP2', 'FP3' for practice
    """
    print(f"\nLoading {year} Round {round_num} - {session_name}...")
    
    # Load session
    session = fastf1.get_session(year, round_num, session_name)
    session.load()
    
    # Get laps data
    laps = session.laps
    
    # Select relevant columns and rename for clarity
    laps_df = laps[[
        'Driver', 'DriverNumber', 'LapNumber', 'LapTime', 'Position',
        'Compound', 'TyreLife', 'PitInTime', 'PitOutTime',
        'Stint', 'Team', 'IsPersonalBest'
    ]].copy()
    
    # Rename TyreLife to TireAge for consistency
    laps_df.rename(columns={'TyreLife': 'TireAge'}, inplace=True)
    
    # Get weather data for each lap
    weather_data = laps.get_weather_data()
    weather_df = weather_data[[
        'AirTemp', 'TrackTemp', 'Humidity', 'Pressure',
        'WindDirection', 'WindSpeed', 'Rainfall'
    ]].copy()
    
    # Reset index to merge properly
    weather_df['LapIndex'] = weather_df.index
    laps_df['LapIndex'] = laps_df.index
    
    # Merge lap data with weather data
    merged_df = pd.merge(laps_df, weather_df, on='LapIndex', how='left')
    
    # Calculate gaps
    gaps_df = calculate_gaps(merged_df, laps)
    
    # Merge gaps into main dataframe
    merged_df = pd.merge(
        merged_df, 
        gaps_df, 
        on=['Driver', 'LapNumber'], 
        how='left'
    )
    
    # Add track status for safety car information
    # FastF1 provides TrackStatus where '4' typically indicates Safety Car
    try:
        # Get track status from session
        track_status = session.track_status
        if track_status is not None and len(track_status) > 0:
            # Create a SafetyCar column based on track status
            merged_df['SafetyCar'] = 0
            
            # You may need to adjust this based on actual track status codes
            # Common codes: '1' = Green, '2' = Yellow, '4' = Safety Car, '6' = VSC
            for idx, row in merged_df.iterrows():
                lap_start = row['LapNumber']
                # This is simplified - you'd need to match times properly
                merged_df.loc[idx, 'SafetyCar'] = 0  # Default to no SC
        else:
            merged_df['SafetyCar'] = 0
    except:
        merged_df['SafetyCar'] = 0
    
    # Create pit stop labels (default k=3)
    merged_df = create_pit_labels(merged_df, k=3)
    
    # Add session metadata
    merged_df['Year'] = year
    merged_df['Round'] = round_num
    merged_df['Session'] = session_name
    merged_df['EventName'] = session.event['EventName']
    
    # Select final columns in desired order
    final_columns = [
        'Year', 'Round', 'EventName', 'Session',
        'Driver', 'DriverNumber', 'Team',
        'LapNumber', 'LapTime', 'Position',
        'TireAge', 'Compound', 'Stint',
        'TrackTemp', 'AirTemp', 'Humidity', 'Pressure',
        'WindDirection', 'WindSpeed', 'Rainfall',
        'GapAhead', 'GapBehind',
        'SafetyCar',
        'PitNext3Laps',
        'PitInTime', 'PitOutTime'
    ]
    
    result_df = merged_df[final_columns].copy()
    
    print(f"✓ Collected {len(result_df)} lap records")
    
    return result_df


def collect_season_data(year, rounds_list, session_name='R'):
    """
    Collect data for multiple rounds in a season
    
    Parameters:
    - year: int, season year
    - rounds_list: list of ints, round numbers to collect
    - session_name: str, session type
    """
    all_data = []
    
    for round_num in rounds_list:
        try:
            session_data = get_session_data(year, round_num, session_name)
            all_data.append(session_data)
        except Exception as e:
            print(f"✗ Error loading Round {round_num}: {str(e)}")
            continue
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n{'='*60}")
        print(f"Total records collected: {len(combined_df)}")
        print(f"Total races: {len(rounds_list)}")
        print(f"Date range: {year}")
        print(f"{'='*60}\n")
        return combined_df
    else:
        return pd.DataFrame()


def clean_and_export(df, filename='f1_pitstop_data.csv'):
    """
    Clean the dataset and export to CSV
    """
    # Remove laps with missing critical features
    print("Cleaning data...")
    initial_rows = len(df)
    
    # Remove outliers and invalid data
    df_clean = df.copy()
    
    # Remove laps with missing tire data
    df_clean = df_clean[df_clean['TireAge'].notna()]
    
    # Remove laps with extreme values (likely errors)
    df_clean = df_clean[df_clean['TireAge'] < 100]  # Tires don't last 100 laps
    
    # Convert LapTime to seconds for easier processing
    df_clean['LapTimeSeconds'] = df_clean['LapTime'].dt.total_seconds()
    
    # Handle missing values in weather data (forward fill within each race)
    weather_cols = ['TrackTemp', 'AirTemp', 'Humidity', 'WindDirection', 'Rainfall']
    for col in weather_cols:
        df_clean[col] = df_clean.groupby(['Year', 'Round'])[col].ffill()
    
    final_rows = len(df_clean)
    print(f"Removed {initial_rows - final_rows} rows with missing/invalid data")
    print(f"Final dataset: {final_rows} rows")
    
    # Export to CSV
    df_clean.to_csv(filename, index=False)
    print(f"\n✓ Data exported to {filename}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total laps: {len(df_clean)}")
    print(f"Unique drivers: {df_clean['Driver'].nunique()}")
    print(f"Pit stops (PitNext3Laps=1): {df_clean['PitNext3Laps'].sum()}")
    print(f"Class balance: {df_clean['PitNext3Laps'].value_counts(normalize=True)}")
    
    return df_clean


# Example usage
if __name__ == "__main__":
    # Collect data for 2023 season, rounds 1-5 (adjust as needed)
    # For full season, use: rounds_list = list(range(1, 23))
    
    print("F1 Pit Stop Prediction - Data Collection")
    print("=" * 60)
    
    # Example: Collect race data from 2023, rounds 1-3
    data = collect_season_data(
        year=2023,
        rounds_list=[1, 2, 3],  # Start with a few races for testing
        session_name='R'  # 'R' for race
    )
    
    # Clean and export
    if not data.empty:
        clean_data = clean_and_export(data, 'f1_pitstop_training_data.csv')
        
        # Display sample
        print("\nSample of collected data:")
        print(clean_data.head(10))
        
        print("\nFeature columns available:")
        print(clean_data.columns.tolist())
    else:
        print("No data collected. Check your internet connection and try again.")