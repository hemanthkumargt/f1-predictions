import fastf1
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder


fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"


def get_circuit_metadata(year):
    schedule = fastf1.get_event_schedule(year)
    races_df = schedule[schedule['EventFormat'].str.lower() == 'conventional']
    circuits_data = []
    for _, row in races_df.iterrows():
        circuits_data.append({
            'RaceName': row['EventName'],
            'CircuitName': row.get('Location', 'Unknown'),
            'CircuitLength_km': row.get('CircuitLength', np.nan),
            'NumberOfLaps': row.get('NumLaps', np.nan),
            'Date': row.get('Date', pd.NaT)
        })
    return pd.DataFrame(circuits_data)


def extract_driver_circuit_stats(year, race_name):
    try:
        session = fastf1.get_session(year, race_name, 'R')
        session.load()
    except Exception as e:
        print(f"Failed to load session for {race_name}: {e}")
        return pd.DataFrame()

    laps = session.laps
    results = getattr(session, 'results', None)
    if results is None or results.empty:
        print(f"No results available for {race_name}")
        return pd.DataFrame()

    weather = getattr(session, 'weather', None)
    if weather is None:
        print(f"No weather data for {race_name}, skipping weather features")

    weather_features = {
        'AirTemp': getattr(weather, 'AirTemp', np.nan) if weather else np.nan,
        'TrackTemp': getattr(weather, 'TrackTemp', np.nan) if weather else np.nan,
        'Humidity': getattr(weather, 'Humidity', np.nan) if weather else np.nan,
        'WindSpeed': getattr(weather, 'WindSpeed', np.nan) if weather else np.nan
    }

    driver_stats = []
    for _, row in results.iterrows():
        driver = row.get('Driver', None)
        if driver is None:
            print(f"Missing driver info for a row in {race_name}, skipping")
            continue
        driver_laps = laps.pick_driver(driver)
        if driver_laps.empty:
            continue
        avg_lap_time = driver_laps['LapTime'].mean().total_seconds()
        fastest_lap = driver_laps['LapTime'].min().total_seconds()
        num_pit_stops = driver_laps['PitOutTime'].notna().sum()

        driver_stats.append({
            'Year': year,
            'RaceName': race_name,
            'Driver': driver,
            'Team': row.get('Team', 'Unknown'),
            'GridPosition': row.get('GridPosition', row.get('Grid', np.nan)),
            'FinalPosition': row.get('Position', np.nan),
            'Points': row.get('Points', 0),
            'AverageLapTime': avg_lap_time,
            'FastestLapTime': fastest_lap,
            'PitStops': num_pit_stops,
            'AirTemp': weather_features['AirTemp'],
            'TrackTemp': weather_features['TrackTemp'],
            'Humidity': weather_features['Humidity'],
            'WindSpeed': weather_features['WindSpeed']
        })

    return pd.DataFrame(driver_stats)


def prepare_dataset(year):
    circuit_meta_df = get_circuit_metadata(year)
    all_driver_stats = []
    race_names = circuit_meta_df['RaceName'].tolist()
    success_count = 0
    fail_count = 0

    for race in race_names:
        print(f"Extracting stats for {race}...")
        try:
            stats_df = extract_driver_circuit_stats(year, race)
            if stats_df.empty:
                print(f"No data extracted for {race}")
                fail_count += 1
                continue
            circuit_info = circuit_meta_df[circuit_meta_df['RaceName'] == race].iloc[0].to_dict()
            for key in ['CircuitName', 'CircuitLength_km', 'NumberOfLaps', 'Date']:
                stats_df[key] = circuit_info.get(key, np.nan)
            all_driver_stats.append(stats_df)
            success_count += 1
        except Exception as e:
            print(f"Failed to extract for {race}: {e}")
            fail_count += 1

    print(f"Races successfully processed: {success_count}")
    print(f"Races failed or skipped: {fail_count}")

    if not all_driver_stats:
        raise ValueError("No race data was successfully extracted. Check logs for details.")

    combined_df = pd.concat(all_driver_stats, ignore_index=True)
    combined_df.dropna(inplace=True)  # Drop rows with missing critical data
    return combined_df


def encode_features(df):
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    le_circuit = LabelEncoder()

    df['Driver_enc'] = le_driver.fit_transform(df['Driver'])
    df['Team_enc'] = le_team.fit_transform(df['Team'])
    df['Circuit_enc'] = le_circuit.fit_transform(df['CircuitName'])
    return df, le_driver, le_team, le_circuit


def train_lightgbm(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(params, train_data, valid_sets=[val_data], early_stopping_rounds=20, verbose_eval=True)
    return model


def main():
    year = 2025
    df = prepare_dataset(year)

    df, le_driver, le_team, le_circuit = encode_features(df)

    feature_cols = [
        'GridPosition', 'AverageLapTime', 'FastestLapTime', 'PitStops',
        'AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed',
        'Driver_enc', 'Team_enc', 'Circuit_enc', 'CircuitLength_km', 'NumberOfLaps'
    ]

    X = df[feature_cols]
    y = df['FinalPosition']

    df = df.sort_values('Date')
    split_idx = int(len(df) * 0.7)

    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    model = train_lightgbm(X_train, y_train, X_val, y_val)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE: {mae:.3f}")

    upcoming_race = df.iloc[split_idx:].copy()
    upcoming_race['PredictedPosition'] = y_pred

    podium_pred = upcoming_race.sort_values('PredictedPosition').head(3)[['Driver', 'Team', 'PredictedPosition']]
    print("\nPredicted Podium for next race:")
    print(podium_pred)


if __name__ == '__main__':
    main()
