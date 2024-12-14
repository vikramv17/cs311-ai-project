import argparse
import pandas as pd
import numpy as np
import random
from sklearn import metrics
import SpotifyAPI as sp
import SongSimilarity as ss
import DecisionTree as dt
import RandomForest as rf

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
    labels = dt.assign_labels_by_rank(df, rank_column="rank")
    # Create bins based on quartiles for int and float dtype columns
    numeric_columns = ["duration_ms", "explicit", "popularity", "acousticness", "danceability", "energy", "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]
    bins = dt.generate_bins_from_quartiles(df, numeric_columns)

    # Generate training and test data
    training_tracks = df.drop(columns=["rank", "track_id", "album_name"])
    test_tracks = df.sample(n=tests, random_state=random.randint(1, 10000))
    test_tracks = pd.concat([test_tracks, test_song], ignore_index=False)
    test_labels = labels.loc[test_tracks.index]
    training_tracks = training_tracks.drop(test_tracks.index).reset_index(drop=True)
    training_labels = labels.drop(test_tracks.index).reset_index(drop=True)

    # Create a copy of the DataFrame to store the tree tracks
    tree_tracks = dt.bucketize_columns(df.drop(columns=["rank", "track_name", "artist", "track_id", "album_name"]), bins)

    # Bucketize columns
    training_data = dt.bucketize_columns(training_tracks, bins)
    test_data = dt.bucketize_columns(test_tracks, bins)

    return training_tracks, test_tracks, training_data, training_labels, test_data, test_labels, tree_tracks, labels

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
    print("Enjoyment Likelihood Results:\n", results)

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
    print("Difference Table:\n", difference_table.to_string(index=False))

    return difference_counts

def compute_metrics(y_true, y_pred):
    """Compute metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.

    Returns:
        list: confusion matrix, accuracy, recall, precision, f1, binary_accuracy, binary_recall, binary_precision, binary_f1, proximity_score
    """
    # Mean Absolute Error
    mae = metrics.mean_absolute_error(y_true, y_pred)

    # Compute confusion matrix, accuracy, recall, precision, and F1 score
    confusion = metrics.confusion_matrix(y_true, y_pred, labels = [0, 1, 2, 3, 4])
    accuracy = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    f1 = metrics.f1_score(y_true, y_pred, average='macro')

    # Binarize true and predicted labels: 0 (dislike) if label is 2, 3, or 4; 1 (like) if label is 0 or 1
    y_true_binary = np.where(np.isin(y_true, [0, 1]), 1, 0)
    y_pred_binary = np.where(np.isin(y_pred, [0, 1]), 1, 0)

    # Compute binary classification metrics
    binary_accuracy = metrics.accuracy_score(y_true_binary, y_pred_binary)
    binary_recall = metrics.recall_score(y_true_binary, y_pred_binary)
    binary_precision = metrics.precision_score(y_true_binary, y_pred_binary)
    binary_f1 = metrics.f1_score(y_true_binary, y_pred_binary)

    proximity_score = 1 - (mae / 4) # Proximity Score (normalized MAE)

    return [confusion, accuracy, recall, precision, f1, binary_accuracy, binary_recall, binary_precision, binary_f1, proximity_score]

def sum_metrics(y_true, y_pred, sum_metrics, difference_counts):
    """Sum the metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.
        sum_metrics: Dictionary of summed metrics
        difference_counts: Series of absolute differences between true and predicted labels

    Returns:
        dict: Dictionary of summed metrics
    """
    computed_metrics = compute_metrics(y_true, y_pred)

    # Sum the metrics
    for i, key in enumerate(sum_metrics.keys()):
        if key != "difference_counts":
            sum_metrics[key] += computed_metrics[i]
        else:
            sum_metrics[key] += difference_counts.values

    return sum_metrics

def display_metrics(metrics, tests):
    """Display metrics to evaluate predictions
    
    Args:
        metrics (dict): Dictionary of metrics to display
    """
    difference_table = pd.DataFrame({
        "Difference": [0, 1, 2, 3, 4],
        "Count": metrics["difference_counts"] / tests
    })

    # Print the difference table
    print("Difference Table:\n", difference_table.to_string(index=False))

    # Display the metrics
    for met, val in metrics.items():
        if met != "difference_counts":
            # Format the metric name
            formatted_met = met.replace('_', ' ').title()
            print(
                formatted_met,
                ": ",
                ("\n" if isinstance(val, np.ndarray) else ""),
                val / tests,
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

    # Step 2: Start Flask server for Spotify authentication and store access token
    access_token = sp.flask_server()

    # Step 3: Fetch all saved tracks and audio features, then combine into a DataFrame
    # tracks = sp.fetch_tracks(access_token)
    tracks = pd.read_json("data/tracks.json") # to avoid fetching tracks every time

    # Step 4: Retrieve audio features of inputted song and remove it from training data
    # new_song = search_song(inputted_song, access_token)
    test_song_id = sp.search_song_v2(inputted_song, access_token)
    test_song = tracks.loc[tracks["track_id"] == test_song_id]
    if test_song.empty and inputted_song:
        print("Inputted song not found in training data.")
    number = 0 if not test_song.empty and not args.number else int(args.number) if args.number else 5
    
    # Initialize average metrics
    average_metrics = {
        "confusion": np.zeros((5, 5)),
        "accuracy": 0,
        "recall": 0,
        "precision": 0,
        "f1": 0,
        "binary_accuracy": 0,
        "binary_recall": 0,
        "binary_precision": 0,
        "binary_f1": 0,
        "proximity_score": 0,
        "difference_counts": [0, 0, 0, 0, 0]
    } 

    for _ in range(tests):
        # Step 5: Generate training and test data
        training_tracks, test_tracks, training_data, training_labels, test_data, test_labels, tree_tracks, tree_labels = generate_training_and_test_data(tracks, number, test_song)

        # Step 6: Determine most similar songs
        ss.most_similar_songs(training_tracks, test_tracks)
        
        # Step 7: Predict likelihood of enjoyment
        # Train the random forest
        forest = rf.RandomForest(n_trees=100, max_features="sqrt")
        forest.fit(training_data.drop(columns=["track_name", "artist"]), training_labels)
        # Make predictions on the test set
        tree_predictions, forest_prediction, mean_prediction = forest.forest_predict(test_data.drop(columns=["track_name", "artist"]))

        # Step 8: Display results and calculate metrics
        difference_counts = display_results(test_data, test_labels, forest_prediction, mean_prediction)
        average_metrics = sum_metrics(test_labels, forest_prediction, average_metrics, difference_counts)

    # Step 9: Display feature importance ranking
    tree = dt.learn_decision_tree(tree_tracks, tree_labels, tree_tracks.columns, tree_labels)
    tree.feature_importance()

    # Step 10: Display metrics 
    display_metrics(average_metrics, tests)
