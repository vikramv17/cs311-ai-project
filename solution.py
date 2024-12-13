import subprocess
import requests
import json
import time
import argparse
import pandas as pd
from typing import Sequence
import numpy as np
import random
import warnings
from sklearn import metrics
from RandomForest import RandomForest
from decision import DecisionBranch, DecisionLeaf, DecisionNode

# Ignore specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")

def get_access_token():
    """Get the access token from the saved JSON file
    
    Returns:
        str: Access token for Spotify API
    """
    try:
        with open('spotify_token.json', 'r') as token_file:
            tokens = json.load(token_file)
            return tokens.get('access_token')
    except FileNotFoundError:
        print("Token file not found. Ensure you have completed the authorization.")
        return None

def get_all_saved_tracks(access_token):
    """Get all saved tracks for the user
    
    Args:
        access_token (str): Access token for Spotify API

    Returns:
        list: List of all saved tracks
    """
    url = "https://api.spotify.com/v1/me/top/tracks"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    all_tracks = []
    offset = 0
    limit = 50  # Maximum limit allowed by Spotify API

    while True:
        params = {
            'type': 'tracks',
            'time_range': 'long_term',
            'limit': limit,
            'offset': offset
        }
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        items = data.get('items', [])
        if not items:
            break
        
        # Collect track information
        for i, track in enumerate(items):
            all_tracks.append({
                'track_id': track['id'],
                'track_name': track['name'],
                'artist': track['artists'][0]['name'],
                'explicit': track['explicit'],
                'popularity': track['popularity'],
                'album_name': track['album']['name']
            })
        
        # Increment offset for next batch
        offset += limit
        print(f"Fetched {len(all_tracks)} tracks so far...")

        # Avoid hitting rate limits
        time.sleep(0.1)

    return all_tracks

def get_audio_features(access_token, track_ids):
    """Get audio features for multiple tracks
    
    Args:
        access_token (str): Access token for Spotify API
        track_ids (list): List of track IDs

    Returns:
        list: List of audio features for the tracks
    """
    url = "https://api.spotify.com/v1/audio-features"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    audio_features = []

    # Fetch in batches of 100 (Spotify API limit)
    for i in range(0, len(track_ids), 100):
        batch_ids = track_ids[i:i + 100]
        params = {
            'ids': ','.join(batch_ids)
        }
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        items = data.get('audio_features', [])
        
        # Collect track information
        for i, track in enumerate(items):
            if track is None:
                continue
            audio_features.append({
                'track_id': track['id'],
                'duration_ms': track['duration_ms'], 
                'acousticness': track['acousticness'], 
                'danceability': track['danceability'], 
                'energy': track['energy'], 
                'instrumentalness': track['instrumentalness'], 
                'key': track['key'], 
                'liveness': track['liveness'], 
                'loudness': track['loudness'], 
                'mode': track['mode'], 
                'speechiness': track['speechiness'], 
                'tempo': track['tempo'], 
                'valence': track['valence']
            })
        
        # Avoid hitting rate limits
        time.sleep(0.1)
    
    return audio_features

def fetch_tracks(access_token):
    """Fetch all saved tracks and their audio features
    
    Args:
        access_token (str): Access token for Spotify API

    Returns:
        pd.DataFrame: DataFrame of all saved tracks with audio features
    """
    if access_token:
        print("Fetching all saved tracks...")
        all_tracks = get_all_saved_tracks(access_token)
        
        if all_tracks:
            # Convert track data to a DataFrame
            df_tracks = pd.DataFrame(all_tracks)
            
            # Extract track IDs
            track_ids = df_tracks['track_id'].tolist()
            
            # Fetch audio features for all tracks
            print("Fetching audio features for tracks...")
            audio_features = get_audio_features(access_token, track_ids)
            df_audio_features = pd.DataFrame(audio_features)

            # Merge track data with audio features
            df = pd.merge(df_tracks, df_audio_features, left_on='track_id', right_on='track_id', how='inner')
            df['rank'] = range(1, len(df_audio_features) + 1)
            df['explicit'] = df['explicit'].astype(int)
            
            # Reorder columns for readability
            columns_order = ['rank', 'track_id', 'track_name', 'artist', 'album_name', 
                             'duration_ms', 'explicit', 'popularity', 'acousticness', 
                             'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 
                             'loudness', 'mode', 'speechiness', 'tempo', 'valence']
            df = df[columns_order]

            # Save the DataFrame to a JSON file
            # df.to_json('tracks.json', orient='records', lines=False)
            # Display the DataFrame
            print("All Tracks with Audio Features:")
            print(df)
        else:
            print("No tracks found or failed to fetch tracks.")
    else:
        print("Access token is missing. Unable to proceed.")

    return df

# No longer used in the final implementation due to API changes eliminating access to Audio Features
def search_song(search_term, access_token):
    """Search for a song and retrieve its audio features
    
    Args:
        search_term (str): Search term for the song
        access_token (str): Access token for Spotify API

    Returns:
        pd.DataFrame: DataFrame of the inputted song with audio features
    """
    track = {}

    url = "https://api.spotify.com/v1/search"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    params = {
        'q': search_term,
        'type': 'track',
        'limit': 1,
    }
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")

    data = response.json()
    item = data.get('tracks', {}).get('items', [{}])[0]
    name = item.get('name')
    id = item.get('id')

    track['id'] = id
    track['track_name'] = name
    track['artist'] = item.get('artists', [{}])[0].get('name')
    track['album_name'] = item.get('album', {}).get('name')
    track['explicit'] = item.get('explicit')
    track['popularity'] = item.get('popularity')

    # Fetch audio features for the inputted song
    url = f"https://api.spotify.com/v1/audio-features/{id}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")

    data = response.json()
    track['duration_ms'] = data.get('duration_ms')
    track['acousticness'] = data.get('acousticness')
    track['danceability'] = data.get('danceability')
    track['energy'] = data.get('energy')
    track['instrumentalness'] = data.get('instrumentalness')
    track['key'] = data.get('key')
    track['liveness'] = data.get('liveness')
    track['loudness'] = data.get('loudness')
    track['mode'] = data.get('mode')
    track['speechiness'] = data.get('speechiness')
    track['tempo'] = data.get('tempo')
    track['valence'] = data.get('valence')
    
    df_new_song = pd.DataFrame([track])

    print("Inputted Track with Audio Features:")
    print(df_new_song)

    return df_new_song

def search_song_v2(search_term, access_token):
    """Search for a song and retrieve its id
    
    Args:
        search_term (str): Search term for the song
        access_token (str): Access token for Spotify API

    Returns:
        str: ID of the inputted song
    """
    url = "https://api.spotify.com/v1/search"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    params = {
        'q': search_term,
        'type': 'track',
        'limit': 2,
    }
    response = requests.get(url, headers=headers, params=params)

    data = response.json()
    item = data.get('tracks', {}).get('items', [{}])[0]
    name = item.get('name')
    id = item.get('id')

    print(f"Inputted Track: {name}")

    return id

def song_similarity(song_data, new_song):
    """Return the most similar song to the inputted song
    
    Args:
        song_data (pd.DataFrame): DataFrame of all songs
        new_song (pd.DataFrame): DataFrame of the input
    
    Returns:
        tuple: Most similar song and artist
    """
    tracks = song_data.copy()

    # Find attributes of new song given by user
    features = ["acousticness", "danceability", "energy", "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]

    all_songs_feature_matrix = tracks[features].values
    new_song_feature_matrix = new_song[features].values.flatten()

    dot_product = np.dot(all_songs_feature_matrix, new_song_feature_matrix)
    magnitudes = np.linalg.norm(all_songs_feature_matrix, axis=1) * np.linalg.norm(new_song_feature_matrix)

    cosine_similarity = dot_product / magnitudes

    tracks["cosine_similarity"] = cosine_similarity
    sorted_similarity = tracks.sort_values(by="cosine_similarity", ascending=False)
    most_similar_song = sorted_similarity.iloc[0]

    return most_similar_song["track_name"], most_similar_song["artist"]

def most_similar_songs(training_tracks, test_tracks):
    """Find the most similar songs to the test songs
    
    Args:
        training_tracks (pd.DataFrame): DataFrame of training tracks
        test_tracks (pd.DataFrame): DataFrame of test
    """
    song_similarity_results = []

    for index, test_song in test_tracks.iterrows():
        most_similar_song, most_similar_song_artist = song_similarity(training_tracks, test_song)
        song_similarity_results.append({
            "song": test_song["track_name"],
            "artist": test_song["artist"],
            "most_similar_song": most_similar_song,
            "most_similar_song_artist": most_similar_song_artist
        })

    song_similarity_df = pd.DataFrame(song_similarity_results)
    
    print("Song Similarity Results:")
    print(song_similarity_df)

def information_gain(X: pd.DataFrame, y: pd.Series, attr: str) -> float:
    """Return the expected reduction in entropy from splitting X,y by attr
    
    Args:
        X (pd.DataFrame): Table of examples (as DataFrame)
        y (pd.Series): array-like example labels (target values)
        attr (str): Attribute to split on

    Returns:
        float: Expected reduction in entropy
    """
    # Calculate entropy before the split
    entropy = -sum([p * np.log2(p) for p in y.value_counts(normalize=True)])
    
    # Calculate entropy after the split
    remainder = 0.0
    for val, subset in X.groupby(attr, observed=False):
        subset_y = y.loc[subset.index]
        subset_entropy = -sum([p * np.log2(p) for p in subset_y.value_counts(normalize=True)])
        remainder += len(subset) / len(X) * subset_entropy
    
    return entropy - remainder

def learn_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    attrs: Sequence[str],
    y_parent: pd.Series,
) -> DecisionNode:
    """Recursively learn the decision tree

    Args:
        X (pd.DataFrame): Table of examples (as DataFrame)
        y (pd.Series): array-like example labels (target values)
        attrs (Sequence[str]): Possible attributes to split examples
        y_parent (pd.Series): array-like example labels for parents (parent target values)

    Returns:
        DecisionNode: Learned decision tree node
    """
    if X.empty:
        # Return plurality of parent examples
        return DecisionLeaf(y_parent.mode()[0])
    elif y.nunique() == 1:
        # Return classification
        return DecisionLeaf(y.iloc[0])
    elif len(attrs) == 0:
        # Return plurality of examples
        return DecisionLeaf(y.mode()[0])
    else:
        # Select attribute with highest information gain
        a = max(attrs, key=lambda a: information_gain(X, y, a))

        # Create branch node and recursively learn subtrees
        tree = DecisionBranch(a, {})
        for v, subset in X.groupby(a, observed=False):
            tree.branches[v] = learn_decision_tree(
                subset, y.loc[subset.index], [attr for attr in attrs if attr != a], y
            )

        return tree

def predict(tree: DecisionNode, X: pd.DataFrame):
    """Return array-like predctions for examples, X and Decision Tree, tree
    
    Args:
        tree (DecisionNode): Decision Tree
        X (pd.DataFrame): Table of examples (as DataFrame)

    Returns:
        pd.Series: Predicted labels for examples
    """

    # You can change the implementation of this function, but do not modify the signature

    # Invoke prediction method on every row in dataframe. `lambda` creates an anonymous function
    # with the specified arguments (in this case a row). The axis argument specifies that the function
    # should be applied to all rows.
    return X.apply(lambda row: tree.predict(row), axis=1)

def assign_labels_by_rank(df: pd.DataFrame, rank_column: str = "rank"):
    """Assigns labels to a dataframe based on the rank of the row
    
    Args:
        df (pd.DataFrame): DataFrame to assign labels to
        rank_column (str): Column to use for ranking

    Returns:
        pd.Series: Series of labels based on the rank
    """
    labels = pd.qcut(df[rank_column], q=5, labels=[0, 1, 2, 3, 4])
    return labels

def generate_bins_from_quartiles(df, columns):
    """Automatically generate 4 bins for numeric columns using quartiles
    
    Args:
        df (pd.DataFrame): DataFrame to generate bins for
        columns (list): List of columns to generate bins for

    Returns:
        dict: Dictionary of column names and their bin edges
    """
    bins = {}
    for column in columns:
        if column not in ["mode", "explicit"]:
            bins[column] = pd.qcut(df[column], q=4, duplicates='drop', retbins=True)[1]
    return bins

def bucketize_columns(data: pd.DataFrame, bins: dict):
    """Bucketize the columns in the dataframe based on the bins provided
    
    Args:
        data (pd.DataFrame): DataFrame to bucketize
        bins (dict): Dictionary of column names and their bin edges

    Returns:
        pd.DataFrame: Bucketized DataFrame
    """
    for column, bin_edges in bins.items():
        bin_labels = range(len(bin_edges) - 1)  # Generate labels for bins
        data[column] = pd.cut(data[column], bins=bin_edges, labels=bin_labels, include_lowest=True)
    
    # Manually binarize 'mode' and 'explicit' columns
    data['mode'] = pd.cut(data['mode'], bins=[-0.1, 0.5, 1.1], labels=[0, 1], include_lowest=True)
    data['explicit'] = pd.cut(data['explicit'], bins=[-0.1, 0.5, 1.1], labels=[0, 1], include_lowest=True)
    
    return data

def generate_training_and_test_data(df, tests=5, test_song=None):
    """Generate training and test data from the DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame to generate data from
        tests (int): Number of test songs to select
        test_song (pd.DataFrame): DataFrame of the inputted song

    Returns:
        tuple: Training and test data
    """
    # Assign labels based on their ranks
    training_labels = assign_labels_by_rank(df, rank_column="rank")
    # Create bins based on quartiles for int and float dtype columns
    numeric_columns = ["duration_ms", "explicit", "popularity", "acousticness", "danceability", "energy", "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]
    bins = generate_bins_from_quartiles(tracks, numeric_columns)

    # Generate training and test data
    training_tracks = df.drop(columns=["rank", "track_id", "album_name"])
    test_tracks = df.sample(n=tests, random_state=random.randint(1, 10000))
    test_tracks = pd.concat([test_tracks, test_song], ignore_index=False)
    test_labels = training_labels.loc[test_tracks.index]
    training_tracks = training_tracks.drop(test_tracks.index).reset_index(drop=True)
    training_labels = training_labels.drop(test_tracks.index).reset_index(drop=True)

    # Bucketize columns
    training_data = bucketize_columns(training_tracks, bins)
    test_data = bucketize_columns(test_tracks, bins)

    return training_tracks, test_tracks, training_data, training_labels, test_data, test_labels

def display_results(test_data, test_labels, pred_labels, pred_mean):
    """Display the results of the predictions
    
    Args:
        test_data (pd.DataFrame): DataFrame of test data
        test_labels (pd.Series): Series of true labels
        pred_labels (pd.Series): Series of predicted labels
        pred_mean (pd.Series): Series of mean predictions

    Returns:
        pd.Series: Series of absolute differences between true and predicted labels
    """
    # Create a DataFrame to store the song name, true label, and predicted label
    results = pd.DataFrame({
        "track_name": test_data["track_name"],
        "artist": test_data["artist"],
        "true_label": test_labels,
        "predicted_label": pred_labels,
        "mean_prediction": pred_mean
    })

    # Convert true_label and predicted_label to numeric dtype
    results["true_label"] = results["true_label"].astype(int)
    results["predicted_label"] = results["predicted_label"].astype(int)

    # Calculate the enjoyment probability for each song
    results["enjoyment_probability"] = results["mean_prediction"].apply(lambda x: f"{(4 - x) / 4:.2f}")

    # Print the results
    print("Results:\n", results)

    # Compute the absolute differences between true labels and predicted labels
    results["difference"] = (results["true_label"] - results["predicted_label"]).abs()

    # Count the occurrences of each difference
    difference_counts = results["difference"].value_counts().sort_index()

    # Reindex difference_counts to include all possible difference values and fill missing values with zeros
    difference_counts = difference_counts.reindex(range(5), fill_value=0)

    # Create a table to display the results
    difference_table = pd.DataFrame({
        "Difference": difference_counts.index,
        "Count": difference_counts.values
    })

    # Print the difference table
    print("Difference Table:\n", difference_table)

    return difference_counts

def compute_metrics(y_true, y_pred):
    """Compute metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.

    Returns:
        tuple: confusion matrix, accuracy, recall, precision, f1, proximity_score
    """
    # Mean Absolute Error
    mae = metrics.mean_absolute_error(y_true, y_pred)

    confusion = metrics.confusion_matrix(y_true, y_pred, labels = [0, 1, 2, 3, 4])
    accuracy = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    f1 = metrics.f1_score(y_true, y_pred, average='macro')

    # Binarize the results: correct if within 1 of the true label
    y_true_binary = np.array(y_true)
    y_pred_binary = np.array(y_pred)
    correct_within_1 = np.abs(y_true_binary - y_pred_binary) <= 1
    y_true_binary = np.ones_like(y_true_binary)
    y_pred_binary = correct_within_1.astype(int)

    binary_accuracy = metrics.accuracy_score(y_true_binary, y_pred_binary)
    binary_recall = metrics.recall_score(y_true_binary, y_pred_binary)
    binary_precision = metrics.precision_score(y_true_binary, y_pred_binary)
    binary_f1 = metrics.f1_score(y_true_binary, y_pred_binary)

    proximity_score = 1 - (mae / 4) # Proximity Score (normalized MAE)

    return confusion, accuracy, recall, precision, f1, binary_accuracy, binary_recall, binary_precision, binary_f1, proximity_score

def sum_metrics(y_true, y_pred, confusion, accuracy, recall, precision, f1, binary_accuracy, binary_recall, binary_precision, binary_f1, proximity_score, difference_counts_accum, difference_counts):
    """Sum the metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.
        confusion: Confusion matrix
        accuracy: Accuracy score
        recall: Recall score
        precision: Precision score
        f1: F1 score
        binary_accuracy: Binary accuracy score
        binary_recall: Binary recall score
        binary_precision: Binary precision score
        binary_f1: Binary F1 score
        proximity_score: Proximity score
        difference_counts_accum: Series of sum of difference counts
        difference_counts: Series of absolute differences between true and predicted labels

    Returns:
        tuple: confusion matrix, accuracy, recall, precision, f1, binary_accuracy, binary_recall, binary_precision, binary_f1, proximity_score, difference_counts
    """
    trial_confusion, trial_accuracy, trial_recall, trial_precision, trial_f1, trial_binary_accuracy, trial_binary_recall, trial_binary_precision, trial_binary_f1, trial_proximity_score = compute_metrics(y_true, y_pred)
    confusion += trial_confusion
    accuracy += trial_accuracy
    recall += trial_recall
    precision += trial_precision
    f1 += trial_f1
    binary_accuracy += trial_binary_accuracy
    binary_recall += trial_binary_recall
    binary_precision += trial_binary_precision
    binary_f1 += trial_binary_f1
    proximity_score += trial_proximity_score
    difference_counts_accum += difference_counts.values

    return confusion, accuracy, recall, precision, f1, binary_accuracy, binary_recall, binary_precision, binary_f1, proximity_score, difference_counts_accum

def display_metrics(metrics, difference_counts):
    """Calculate and display metrics to evaluate predictions
    
    Args:
        test_labels (pd.Series): Series of true labels
        pred_labels (pd.Series): Series of predicted
    """
    difference_table = pd.DataFrame({
        "Difference": [0, 1, 2, 3, 4],
        "Count": difference_counts
    })

    # Print the difference table
    print("Difference Table:\n", difference_table)

    for met, val in metrics.items():
        # Format the metric name
        formatted_met = met.replace('_', ' ').title()
        print(
            formatted_met,
            ": ",
            ("\n" if isinstance(val, np.ndarray) else ""),
            val,
            sep="",
        )

if __name__ == "__main__":
    # Step 1: Parse inputted song
    parser = argparse.ArgumentParser(description="CS311 AI Project")
    parser.add_argument(
        "-s",
        "--song",
        help="Search term for the inputted song",
    )
    parser.add_argument(
        "-n",
        "--number",
        help="Number of test songs to select",
    )
    parser.add_argument(
        "-t",
        "--tests",
        default=1,
        help="Number of tests to run",
    )
    args = parser.parse_args()
    inputted_song = args.song
    tests = int(args.tests)

    # Step 1: Start Flask server for Spotify authentication
    flask_process = subprocess.Popen(['python', 'spotify-login.py'])

    # Step 2: Prompt the user to authorize the app
    print("Please go to http://localhost:3000/login to authorize the app.")
    print("After authorizing, press Enter to continue...")
    input() # Wait for user input after authorization
    
    # Step 3: Fetch the access token
    access_token = get_access_token()
    if access_token:
        print("Access token obtained successfully.")
    else:
        print("Failed to obtain access token.")

    # Step 4: Terminate the Flask server once tracks are returned
    flask_process.terminate()

    # Step 5: Fetch all saved tracks and audio features, then combine into a DataFrame
    # tracks = fetch_tracks(access_token)
    tracks = pd.read_json("data/tracks.json") # to avoid fetching tracks every time

    # Step 6: Retrieve audio features of inputted song and remove it from training data
    # new_song = search_song(inputted_song, access_token)
    test_song_id = search_song_v2(inputted_song, access_token)
    test_song = tracks.loc[tracks["track_id"] == test_song_id]
    if test_song.empty and inputted_song:
        print("Inputted song not found in training data.")
    number = 0 if not test_song.empty and not args.number else int(args.number) if args.number else 5
    
    confusion = np.zeros((5, 5))
    accuracy, recall, precision, f1, binary_accuracy, binary_recall, binary_precision, binary_f1, proximity_score = 0, 0, 0, 0, 0, 0, 0, 0, 0
    difference_counts_accum = [0, 0, 0, 0, 0]

    for _ in range(tests):
        # Step 7: Generate training and test data
        training_tracks, test_tracks, training_data, training_labels, test_data, test_labels = generate_training_and_test_data(tracks, number, test_song)

        # Step 8: Determine most similar songs
        most_similar_songs(training_tracks, test_tracks)

        # Step 9: Predict likelihood of enjoyment
        # Train the random forest
        rf = RandomForest(n_trees=100, max_features="sqrt")
        rf.fit(training_data.drop(columns=["track_name", "artist"]), training_labels)
        # Make predictions on the test set
        tree_predictions, forest_prediction, mean_prediction = rf.forest_predict(test_data.drop(columns=["track_name", "artist"]))

        # Step 10: Display results and calculate metrics
        difference_counts = display_results(test_data, test_labels, forest_prediction, mean_prediction)
        confusion, accuracy, recall, precision, f1, binary_accuracy, binary_recall, binary_precision, binary_f1, proximity_score, difference_counts_accum = sum_metrics(test_labels, forest_prediction, confusion, accuracy, recall, precision, f1, binary_accuracy, binary_recall, binary_precision, binary_f1, proximity_score, difference_counts_accum, difference_counts)

    # Step 11: Display metrics
    average_metrics = {
        "confusion": confusion / tests,
        "accuracy": accuracy / tests,
        "recall": recall / tests,
        "precision": precision / tests,
        "f1": f1 / tests,
        "binary_accuracy": binary_accuracy / tests,
        "binary_recall": binary_recall / tests,
        "binary_precision": binary_precision / tests,
        "binary_f1": binary_f1 / tests,
        "proximity_score": proximity_score / tests,
    }   
    display_metrics(average_metrics, difference_counts_accum / tests)
