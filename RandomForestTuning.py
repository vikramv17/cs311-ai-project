import json
import pandas as pd
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn import metrics
from RandomForest import RandomForest 
import DecisionTree as dt 

#Pre-processing stuff taken from D-Tree file

# Load the data from a JSON file into a pandas DataFrame
with open("data/tracks.json", "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Assign labels based on their ranks
training_labels = dt.assign_labels_by_rank(df, rank_column="rank")
print("Data with Labels:\n", df)

# Create bins based on quartiles for int and float dtype columns
numeric_columns = ["duration_ms", "explicit", "popularity", "acousticness", "danceability", "energy", "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]
bins = dt.generate_bins_from_quartiles(df, numeric_columns)

# Bucketize columns
df = dt.bucketize_columns(df, bins)

# Generate training and test data
training_data, test_data, test_labels = dt.generate_training_and_test_data(df, training_labels)

#Hyperparameter tuning

param_grid = {
    "n_trees": [50, 100, 150],
    "max_depth": [3,5,7]
}

#Get the combinations
grid = ParameterGrid(param_grid)

#5-fold cross validation
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=15)

results = []
for params in grid:
    print(f"Running with parameters: {params}")
    fold_accuracy = []
    fold_proximity_score = []

    #Perform CV & training
    for train_idx, val_idx in cv.split(df, training_labels):
        train_data, val_data = df.iloc[train_idx], df.iloc[val_idx]
        train_labels, val_labels = training_labels.iloc[train_idx], training_labels.iloc[val_idx]

        rf = RandomForest(
            n_trees=params["n_trees"],
            max_depth=params["max_depth"]
        )
        features = training_data.drop(columns=["track_name", "artist"]).columns

        rf.fit(train_data[features], train_labels)

        #Quantify Random Forest performance
        _, forest_prediction, _ = rf.forest_predict(val_data[features])
        accuracy = metrics.accuracy_score(val_labels, forest_prediction)
        mae = metrics.mean_absolute_error(val_labels, forest_prediction)
        label_range = max(val_labels) - min(val_labels)
        proximity_score = 1 - (mae / label_range)

        fold_accuracy.append(accuracy)
        fold_proximity_score.append(proximity_score)

    #CV results
    mean_accuracy = sum(fold_accuracy) / n_splits
    mean_proximity_score = sum(fold_proximity_score) / n_splits
    results.append({"params": params, "mean_accuracy": mean_accuracy, "mean_proximity_score": mean_proximity_score})

#Results sorted by accuracy
sorted_results = sorted(results, key=lambda x: x["mean_accuracy"], reverse=True)
print("\nFinal Results (sorted by accuracy):\n")
for res in sorted_results:
    print("Parameters:", res["params"])
    print("Mean Accuracy:", res["mean_accuracy"])
    print("Mean Proximity Score:", res["mean_proximity_score"])