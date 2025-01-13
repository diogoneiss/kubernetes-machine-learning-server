import os
import pathlib
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from fpgrowth_py import fpgrowth
from mlxtend.frequent_patterns import fpgrowth as fpgrowth_one_hot
from mlxtend.preprocessing import TransactionEncoder


load_dotenv()

# Load environment variables
MIN_SUPPORT = float(os.getenv("MIN_SUPPORT", 0.05))

BASE_DIR = os.getenv("BASE_DIR", "./datasets")
DATASETS_DIR = os.getenv("DATASETS_DIR", "./datasets")

PICKLES_FOLDER = os.getenv("PICKLES_FOLDER", "./pickles")

REGEX_FILENAME = os.getenv("REGEX_FILENAME", "2023_spotify_ds*.csv")

DATASET_LIST_FILE = f"{BASE_DIR}/datasets_list.txt"

PICKLE_ARTISTS_FILE = "artistsMapping.pickle"
PICKLE_TRACK_ID_TO_TRACK_INFO = "trackIdsToInfo.pickle"
PICKLE_DUPLICATED_TRACKS = "trackNameToRepeatedUris.pickle"

BASE_INDEX = 1

DROP_COLUMNS = ["duration_ms"]
FP_GROWTH_DROP_COLUMNS = ["track_uri", "album_name", "artist_uri"]


sample_ratio = 1
best_n_tracks = 5

total_songs = None

def validate_and_map_artists_names_to_ids(df: pl.DataFrame) -> dict:

    grouped = (
        df
        .group_by("artist_name")
        .agg([
            pl.col("artist_uri").n_unique().alias("unique_artist_uri_count"),
            pl.col("artist_uri").unique().alias("artist_uris")
        ])
    )

    duplicates = grouped.filter(pl.col("unique_artist_uri_count") > 1)

    if duplicates.shape[0] > 0:
        msg = f"Found {duplicates.shape[0]} duplicate artists"
        print(msg)
        print(duplicates)
        raise ValueError(msg)

    grouped = (
        df.group_by("artist_name")
        .agg(pl.col("artist_uri").unique().alias("artist_uris"))
    )

    # Convert grouped DataFrame to a Python dict
    artist_mapping = {}
    for row in grouped.iter_rows():
        name, uris = row
        artist_mapping[name] = uris[0]

    save_pickle(PICKLE_ARTISTS_FILE, artist_mapping)

    return artist_mapping


def extract_repeated_track_names(df):
    grouped_tracks = (
        df
        .group_by("track_name")
        .agg([
            pl.col("track_uri").n_unique().alias("unique_track_uri_count"),
            pl.col("track_uri").unique().alias("track_uri")
        ])
    )
    duplicate_songs = grouped_tracks.filter(pl.col("unique_track_uri_count") > 1)
    duplicate_songs_dicts = duplicate_songs.to_dicts()

    track_names_to_repeated_uris = {}

    for row in duplicate_songs_dicts:
        track_name = row["track_name"]
        track_uris = row["track_uri"]
        track_names_to_repeated_uris[track_name] = track_uris

    if duplicate_songs.shape[0] > 0:
        msg = f"Found {duplicate_songs.shape[0]} duplicate songs, saving to {PICKLE_DUPLICATED_TRACKS}"
        print(msg)

        save_pickle(PICKLE_DUPLICATED_TRACKS, track_names_to_repeated_uris)


def map_song_ids_to_song_info(df: pl.DataFrame) -> dict:

    df_mapping = (
        df.group_by("track_uri")
        .agg([
            pl.first("track_name").alias("track_name"),
            pl.first("artist_name").alias("artist_name"),
            pl.first("album_name").alias("album_name")
        ])
        .select([
            pl.col("track_uri"),
            pl.struct(["track_name", "artist_name", "album_name"]).alias("info")
        ])
    )

    rows = df_mapping.to_dicts()

    track_mapping = {row["track_uri"]: row["info"] for row in rows}

    save_pickle(PICKLE_TRACK_ID_TO_TRACK_INFO, track_mapping)

    return track_mapping


def save_pickle(pickle_path: str, data: dict):

    pathlib.Path(PICKLES_FOLDER).mkdir(parents=True, exist_ok=True)

    # Pickle the mapping
    full_path = pathlib.Path(PICKLES_FOLDER) / pickle_path
    with open(full_path, "wb") as f:
        pickle.dump(data, f)

def clean_df(df: pl.DataFrame) -> pl.DataFrame:
    df = df.drop(DROP_COLUMNS)
    return df

def read_tracks(file_path: str) -> pl.DataFrame:
    df = pl.read_csv(file_path)


    if 0 < sample_ratio < 1:
        print(f"Sample ratio: {sample_ratio}")
        sample_size = max(1, int(df.height * sample_ratio))
        df = df.head(sample_size)

    print(f"Rows: {df.height}, Columns: {df.width}")

    # prind unique pid
    print(f"Unique pids: {df['pid'].n_unique()}")
    print(f"Unique songs: {df['track_uri'].n_unique()}")
    return df

def get_most_frequent_tracks(df: pl.DataFrame) -> list:
    most_frequent_tracks = df.group_by("track_name").agg(
        pl.col("track_uri").count().alias("count")
    ).sort("count").reverse()

    most_frequent_dict = most_frequent_tracks.to_dicts()

    return most_frequent_dict

def save_most_frequent_tracks_dict(sorted_most_frequent_dict: list):
    best_percentage = 0.1
    desired_track_count = int(len(sorted_most_frequent_dict) * best_percentage)
    best_tracks = sorted_most_frequent_dict[:desired_track_count]


    track_list = [track["track_name"] for track in best_tracks]

    return track_list

def group_tracks_by_playlist_and_generate_homogeneous_data(df: pl.DataFrame) -> dict:
    grouped = df.group_by("pid").agg(
        pl.col("track_name").alias("track_names")
    )

    rows = grouped.to_dicts()

    #dictionary: { pid: [list_of_track_names], ... }
    playlists_dict = {
        row["pid"]: row["track_names"] for row in rows
    }

    return playlists_dict

def group_tracks_by_playlist_and_generate_heterogeneous_data(df: pl.DataFrame) -> dict:
    df = df.drop(FP_GROWTH_DROP_COLUMNS)

    pids = df["pid"].unique().to_list()
    playlists_dict = {}
    for pid in pids:
        df_pid = df.filter(pl.col("pid") == pid).drop("pid")

        pid_dict = df_pid.to_dict()
        pid_dicts = df_pid.to_dicts()

        playlists_dict[pid] = df_pid

    return playlists_dict

def calculate_and_save_fp_growth(playlist_dict: dict) -> list[dict]:
    transactions = list(playlist_dict.values())
    min_support = 0.05
    min_confidence = 0.02 * 2

    data = fpgrowth(
        transactions,
        minSupRatio=min_support,
        minConf=min_confidence
    )
    if data is None:
        msg = "Parameter error: No frequent item set found. Check values for min support and min confidence."
        print(msg)
        raise ValueError(msg)

    frequent_itemsets, rules = data

    if len(rules) == 0:
        msg = "No rules found. Please check the parameters."
        print(msg)
        raise ValueError(msg)

    songs_to_song_sets = {}
    for song, other_songs, confidence in rules:
        if song not in songs_to_song_sets:
            songs_to_song_sets[song] = dict()
        current_song_data = songs_to_song_sets[song]
        for other in other_songs:
            if other not in current_song_data:
                current_song_data[other] = confidence
            else:
                current_song_data[other] = max(current_song_data[other], confidence)

    songs_without_recommendations = total_songs - len(songs_to_song_sets.keys())
    print("Songs without recommendations:", songs_without_recommendations)

    return songs_to_song_sets

def calculate_and_save_fp_growth_fast(playlist_dict: dict, min_support = 0.07) -> list[dict]:

    start_time = pd.Timestamp.now()
    transactions = list(playlist_dict.values())

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)


    data = fpgrowth_one_hot(df, min_support=min_support, use_colnames=True)

    if data is None:
        msg = "Parameter error: No frequent item set found. Check values for min support and min confidence."
        print(msg)
        raise ValueError(msg)

    # filter frozensets with more than one element
    data_after = data[data['itemsets'].map(len) > 1]

    songs_to_song_sets = {}

    for row in data.itertuples():
        itemset = set(row.itemsets)
        confidence = row.support
        for song in itemset:
            other_songs = itemset - {song}
            if song not in songs_to_song_sets:
                songs_to_song_sets[song] = dict()
            current_song_data = songs_to_song_sets[song]
            for other in other_songs:
                if other not in current_song_data:
                    current_song_data[other] = confidence
                else:
                    current_song_data[other] = max(current_song_data[other], confidence)

    # pick keys without songs
    no_songs = []
    for key in songs_to_song_sets.keys():
        if not songs_to_song_sets[key]:
            no_songs.append(key)

    songs_without_recommendations = total_songs - len(songs_to_song_sets.keys())
    print("Songs without recommendations:", songs_without_recommendations)
    end_time = pd.Timestamp.now()
    duration = end_time - start_time
    print(f"Time elapsed: {duration}")


    info = f"min_support: {min_support} \tmissing songs: {songs_without_recommendations} \ttime: {duration}"
    runtime_info = (songs_without_recommendations, duration)
    return songs_to_song_sets, info, runtime_info

def recommend_tracks_for_track(songs_to_song_sets: dict, seed_track):
    if seed_track not in songs_to_song_sets:
        print(f"Track {seed_track} not found in the song recomendation list.")
        return set()

    recommendations = songs_to_song_sets[seed_track]
    # sort dict by value
    sorted_by_confidence = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    recommended = sorted_by_confidence[:best_n_tracks]
    song_names = [song[0] for song in recommended]
    return song_names


def get_dataset_list():
    if not os.path.exists(DATASET_LIST_FILE):
        print("Current dataset file not found. Initializing...")
        return write_dataset_and_reset_index()

    return read_dataset()

def read_dataset():
    with open(DATASET_LIST_FILE, "r") as f:
        datasets = f.read().splitlines()

    print(f"Datasets read from {DATASET_LIST_FILE}")
    return datasets

def write_dataset_and_reset_index():
    base_dir_path = pathlib.Path(DATASETS_DIR)
    base_dir_path.mkdir(parents=True, exist_ok=True)

    dataset_paths = sorted(base_dir_path.glob(REGEX_FILENAME))

    datasets = [str(p.as_posix()) for p in dataset_paths]

    if not datasets:
        raise FileNotFoundError("No datasets found with pattern. Please check your environment setup.")

    list_file_path = pathlib.Path(DATASET_LIST_FILE)
    with list_file_path.open("w", encoding="utf-8") as f:
        for dataset in datasets:
            f.write(f"{dataset}\n")

    print(f"Datasets written to {list_file_path}")

    return datasets

DATASET_HISTORY_FILE = "dataset_history.csv"

def read_history_csv() -> list:
    """
    Reads the dataset history from a CSV file with columns: time, dataset_index, dataset_file
    Returns a list of rows (including the header if it exists).
    If file does not exist, it returns an empty list.
    """
    if not os.path.exists(DATASET_HISTORY_FILE):
        print(f"{DATASET_HISTORY_FILE} not found, returning empty history.")
        return []

    with open(DATASET_HISTORY_FILE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    return lines

def get_next_run_index() -> int:
    """
    Reads the history CSV and returns the most recently used dataset index.
    If no history exists, return BASE_INDEX (or 1).
    """
    history_lines = read_history_csv()

    # If the file has only a header or is empty, return the base index
    if len(history_lines) <= 1:
        print("No previous run found in history, defaulting to base index.")
        return BASE_INDEX

    # Last line of the CSV (excluding the header) will have 3 columns:
    # e.g., "2025-01-10 10:30:00, 3, /path/to/dataset3.csv"
    last_line = history_lines[-1]
    parts = last_line.split(",")

    # The dataset_index is expected in the second column (index 1)
    try:
        last_index = int(parts[1].strip())
        new_index = last_index + 1

        if new_index > len(datasets):
            new_index = BASE_INDEX

        return new_index
    except (ValueError, IndexError):
        print("History file had a malformed line. Defaulting to base index.")
        return BASE_INDEX

def append_dataset_history(dataset_index: int, dataset_file: str):
    """
    Appends a new line to the dataset history CSV:
      time, dataset_index, dataset_file
    If file doesn't exist, it creates one with a header.
    """
    file_existed = os.path.exists(DATASET_HISTORY_FILE)

    with open(DATASET_HISTORY_FILE, "a", encoding="utf-8") as f:
        if not file_existed:
            f.write("time,dataset_index,dataset_file\n")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{now_str},{dataset_index},{dataset_file}\n"
        f.write(line)

    print(f"Appended dataset {dataset_index} ({dataset_file}) to history.")

if __name__ == "__main__":
    datasets = get_dataset_list()

    new_index = get_next_run_index()

    selected_dataset = datasets[new_index - 1]

    print(f"Selected dataset: {selected_dataset}")

    df_tracks_crude = read_tracks(selected_dataset)
    df_tracks = clean_df(df_tracks_crude)

    total_songs = df_tracks["track_uri"].n_unique()

    validate_and_map_artists_names_to_ids(df_tracks)
    extract_repeated_track_names(df_tracks)

    map_song_ids_to_song_info(df_tracks)

    most_frequent_tracks = get_most_frequent_tracks(df_tracks)

    #TODO save to pickle and read in api server
    save_most_frequent_tracks_dict(most_frequent_tracks)

    # Pass the selected dataset to the grouping function
    pidToTracksDict = group_tracks_by_playlist_and_generate_homogeneous_data(df_tracks)
    experiment_supports = False

    if experiment_supports:

        # Create an empty DataFrame with the required columns
        results_df = pd.DataFrame(columns=["min_support", "songs_without_recommendations", "duration"])

        min_supports = np.arange(0.03, 0.2, 0.0025).tolist()

        for min_support in min_supports:
            min_support = round(min_support, 3)
            print(f"Calculating for min_support: {min_support}")
            rules, info, runtime_info = calculate_and_save_fp_growth_fast(pidToTracksDict, min_support)
            songs_without_recommendations, duration = runtime_info

            current_result = pd.DataFrame([{
                "min_support": min_support,
                "songs_without_recommendations": songs_without_recommendations,
                "duration": duration.total_seconds()
            }])

            # Concatenate the current result with the results DataFrame
            results_df = pd.concat([results_df, current_result], ignore_index=True)

            results_df.to_csv("fp_growth_experiment_results.csv", index=False)


    songs_to_song_sets, _, _ = calculate_and_save_fp_growth_fast(pidToTracksDict, MIN_SUPPORT)
    #calculate_and_save_fp_growth(pidToTracksDict)


    seed = 'Gold Digger'
    suggestions = recommend_tracks_for_track(songs_to_song_sets, seed)
    print(f"Recommendations when someone listens to '{seed}':", suggestions)

    append_dataset_history(new_index, selected_dataset)

    print("Run complete. Exiting.")
    sys.exit(0)
