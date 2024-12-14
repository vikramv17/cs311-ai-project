import math
import numpy as np
import json
import pandas as pd
from typing import List, Union
import DecisionTree as dt

class RandomForest:
    #use sklearn's defaults
    def __init__(self, n_trees: int = 100, max_features: Union[str, int] = "sqrt"):
        """Create Random Forest

        Args:
            n_trees (int): Number of trees in forest
            max_features (str): Number of features to consider when looking for best split
        """
        #hyperparameters
        self.n_trees = n_trees
        self.max_features = max_features

        self.trees = []

    def bootstrapping(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Get sample (w/ replacement) of whole dataset"""
        random_indices = np.random.choice(len(X), size = len(X), replace=True)
        return X.iloc[random_indices], y.iloc[random_indices]
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train random forest on the dataset."""
        n_samples, n_features = X.shape

        if self.max_features == "sqrt":
            self.max_features = int(math.sqrt(n_features))

        elif self.max_features == "log2":
            self.max_features = int(np.log2(n_features))

        for _ in range(self.n_trees):
            sampled_X, sampled_y = self.bootstrapping(X,y)
            
    
            tree = dt.learn_decision_tree(
                X=sampled_X,
                y=sampled_y,
                attrs=np.random.choice(X.columns, self.max_features, replace=False),
                y_parent=sampled_y
            )

            self.trees.append(tree)
        
    def forest_predict(self, X: pd.DataFrame) -> tuple:
        #predictions from all trees (columns are index of tree, row values are the predicted labels for given sample)
        tree_predictions = pd.DataFrame({i: dt.predict(tree, X) for i, tree in enumerate(self.trees)})

        # Majority label
        majority_label = tree_predictions.mode(axis=1)[0]

        # Mean prediction
        mean_prediction = tree_predictions.mean(axis=1)

        return tree_predictions, majority_label, mean_prediction

if __name__ == "__main__":
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
    
    # Train the random forest
    rf = RandomForest(n_trees=100, max_features="sqrt")
    rf.fit(training_data.drop(columns=["track_name", "artist"]), training_labels)

    # Make predictions on the test set
    tree_predictions, forest_prediction, mean_prediction = rf.forest_predict(test_data.drop(columns=["track_name", "artist"]))

    # Display results
    dt.display_results(test_data, test_labels, forest_prediction)

    # Calculate and display metrics
    dt.display_metrics(test_labels, forest_prediction)