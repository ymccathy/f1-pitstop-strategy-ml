import pandas as pd
import os

def merge_f1_years(year_files):
    
    all_data = []
    
    print(f"\n{'='*70}")
    print("  Merging F1 Data Files")
    print(f"{'='*70}\n")
    
    for filename in year_files:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            year = df['Year'].iloc[0]
            
            print(f"  ✓ {filename:25s} - {len(df):7,} laps, {df['Round'].nunique():2d} races")
            all_data.append(df)
        else:
            print(f"  ✗ {filename:25s} - FILE NOT FOUND")
    
    if not all_data:
        print("\n✗ No data files found!")
        return None


    combined = pd.concat(all_data, ignore_index=True)
    # Sort by Year, Round, Driver, LapNumber (CRITICAL for LSTM)
    combined = combined.sort_values(['Year', 'Round', 'Driver', 'LapNumber']).reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print("  FINAL MERGED DATASET")
    print(f"{'='*70}")
    print(f"  Years:            {sorted(combined['Year'].unique())}")
    print(f"  Total laps:       {len(combined):,}")
    print(f"  Total races:      {combined.groupby(['Year', 'Round']).ngroups}")
    print(f"  Unique drivers:   {combined['Driver'].nunique()}")
    print(f"  Unique teams:     {combined['Team'].nunique()}")
    print(f"  Total pit stops:  {combined['PitThisLap'].sum():,}")
    print(f"  Pit stop rate:    {combined['PitThisLap'].mean()*100:.2f}%")
    
    print(f"\n  Breakdown by year:")
    for year in sorted(combined['Year'].unique()):
        year_data = combined[combined['Year'] == year]
        print(f"    {year}: {len(year_data):7,} laps, "
              f"{year_data['Round'].nunique():2d} races, "
              f"{year_data['PitThisLap'].sum():4d} pit stops")
    
    print(f"\n  Tire compounds:")
    for compound, count in combined['Compound'].value_counts().items():
        print(f"    {compound:15s}: {count:7,} laps")
    
    print(f"{'='*70}\n")
    
    return combined

def save_final_dataset(df, filename='f1_race_data_2021_2025_final.csv'):
    """Save the final merged dataset"""
    if df is None:
        return
    
    df.to_csv(filename, index=False)
    print(f"✓ Final dataset saved to: {filename}")
    print(f"  File size: {os.path.getsize(filename) / 1024 / 1024:.1f} MB\n")
    
    print("Sample data (first 10 rows):")
    print(df[['Year', 'Round', 'EventName', 'Driver', 'LapNumber', 'Position', 
              'TireAge', 'Compound', 'LapTimeSeconds', 'PitThisLap']].head(10))
    print("\n✓ Dataset ready!")


if __name__ == "__main__":
    
    year_files = [
        'f1_data_2021.csv',
        'f1_data_2022.csv',
        'f1_data_2023.csv',
        'f1_data_2024.csv',
        'f1_data_2025.csv'
    ]
    final_data = merge_f1_years(year_files)
  
    # Save final dataset
    if final_data is not None:
        save_final_dataset(final_data, 'f1_race_data_2021_2025_final.csv')