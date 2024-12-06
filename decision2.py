"""
CS311 Programming Assignment 5: Decision Trees
"""

import argparse, os, random, sys
from typing import Any, Dict, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import json
import warnings

# Ignore entropy calculation warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar multiply")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log2")

# Type alias for nodes in decision tree
DecisionNode = Union["DecisionBranch", "DecisionLeaf"]

class DecisionBranch:
    """Branching node in decision tree"""

    def __init__(self, attr: str, branches: Dict[Any, DecisionNode]):
        """Create branching node in decision tree

        Args:
            attr (str): Splitting attribute
            branches (Dict[Any, DecisionNode]): Children nodes for each possible value of `attr`
        """
        self.attr = attr
        self.branches = branches

    def predict(self, x: pd.Series):
        """Return predicted labeled for array-like example x"""
        return self.branches[x[self.attr]].predict(x)

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Test Feature", self.attr)
        for val, subtree in self.branches.items():
            print(" " * 4 * indent, self.attr, "=", val, "->", end=" ")
            subtree.display(indent + 1)

class DecisionLeaf:
    """Leaf node in decision tree"""

    def __init__(self, label):
        """Create leaf node in decision tree

        Args:
            label: Label for this node
        """
        self.label = label

    def predict(self, x):
        """Return predicted labeled for array-like example x"""
        return self.label

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Label=", self.label)

def information_gain(X: pd.DataFrame, y: pd.Series, attr: str) -> float:
    """Return the expected reduction in entropy from splitting X,y by attr"""
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

def fit(X: pd.DataFrame, y: pd.Series) -> DecisionNode:
    """Return train decision tree on examples, X, with labels, y"""
    # You can change the implementation of this function, but do not modify the signature
    return learn_decision_tree(X, y, X.columns, y)

def predict(tree: DecisionNode, X: pd.DataFrame):
    """Return array-like predctions for examples, X and Decision Tree, tree"""

    # You can change the implementation of this function, but do not modify the signature

    # Invoke prediction method on every row in dataframe. `lambda` creates an anonymous function
    # with the specified arguments (in this case a row). The axis argument specifies that the function
    # should be applied to all rows.
    return X.apply(lambda row: tree.predict(row), axis=1)

def compute_metrics(y_true, y_pred):
    """Compute metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.

    Returns:
        dict: Dictionary of metrics in including confusion matrix, accuracy, recall, precision and F1
    """
    # Mean Absolute Error
    mae = metrics.mean_absolute_error(y_true, y_pred)
    
    # Proximity Score (normalized MAE)
    label_range = max(y_true) - min(y_true)
    proximity_score = 1 - (mae / label_range)

    return {
        "confusion": metrics.confusion_matrix(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred, average='macro'),
        "precision": metrics.precision_score(y_true, y_pred, average='macro'),
        "f1": metrics.f1_score(y_true, y_pred, average='macro'),
        "proximity_score": proximity_score,
    }

def assign_labels_by_rank(df: pd.DataFrame, rank_column: str = "rank"):
    '''Assigns labels to a dataframe based on the rank of the row'''
    labels = pd.qcut(df[rank_column], q=5, labels=[0, 1, 2, 3, 4])
    return labels

def generate_bins_from_quartiles(df, columns):
    """
    Automatically generate 4 bins for numeric columns using quartiles.
    """
    bins = {}
    for column in columns:
        if column not in ["mode", "explicit"]:
            bins[column] = pd.qcut(df[column], q=4, duplicates='drop', retbins=True)[1]
    return bins

def bucketize_columns(data: pd.DataFrame, bins: dict):
    '''Bucketize the columns in the dataframe based on the bins'''
    for column, bin_edges in bins.items():
        bin_labels = range(len(bin_edges) - 1)  # Generate labels for bins
        data[column] = pd.cut(data[column], bins=bin_edges, labels=bin_labels, include_lowest=True)
    
    # Manually binarize 'mode' and 'explicit' columns
    data['mode'] = pd.cut(data['mode'], bins=[-0.1, 0.5, 1.1], labels=[0, 1], include_lowest=True)
    data['explicit'] = pd.cut(data['explicit'], bins=[-0.1, 0.5, 1.1], labels=[0, 1], include_lowest=True)
    
    return data

if __name__ == "__main__":
    # Load the data from a JSON file into a pandas DataFrame
    with open("tracks.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Assign labels based on their ranks
    training_labels = assign_labels_by_rank(df, rank_column="rank")
    print("Data with Labels:\n", df)
    
    # Create bins based on quartiles for int and float dtype columns
    numeric_columns = ["duration_ms", "explicit", "popularity", "acousticness", "danceability", "energy", "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]
    bins = generate_bins_from_quartiles(df, numeric_columns)
    
    # Bucketize columns
    df = bucketize_columns(df, bins)

    training_data = df.drop(columns=["rank", "track_id", "album_name"])

    # Randomly select 5 songs for testing
    test_data = training_data.sample(n=100, random_state=42)
    test_labels = training_labels.loc[test_data.index]

    # Drop the selected songs from the training data
    training_data = training_data.drop(test_data.index)

    # Reset the index of the training data
    training_data = training_data.reset_index(drop=True)
    tree = fit(training_data.drop(columns=["track_name", "artist"]), training_labels)
    # tree.display()

    # Load test data and predict labels with previously learned tree
    pred_labels = predict(tree, test_data.drop(columns=["track_name", "artist"]))

    # Create a DataFrame to store the song name, true label, and predicted label
    results = pd.DataFrame({
        "track_name": test_data["track_name"],
        "artist": test_data["artist"],
        "true_label": test_labels,
        "predicted_label": pred_labels
    })

    # Convert true_label and predicted_label to numeric dtype
    results["true_label"] = results["true_label"].astype(int)
    results["predicted_label"] = results["predicted_label"].astype(int)

    # Print the results
    print("Results:\n", results)

    # Compute the absolute differences between true labels and predicted labels
    results["difference"] = (results["true_label"] - results["predicted_label"]).abs()

    # Count the occurrences of each difference
    difference_counts = results["difference"].value_counts().sort_index()

    # Create a table to display the results
    difference_table = pd.DataFrame({
        "Difference": difference_counts.index,
        "Count": difference_counts.values
    })

    # Print the difference table
    print("Difference Table:\n", difference_table)

    # Compute and print accuracy metrics
    predict_metrics = compute_metrics(test_labels, pred_labels)
    for met, val in predict_metrics.items():
        # Format the metric name
        formatted_met = met.replace('_', ' ').title()
        print(
            formatted_met,
            ": ",
            ("\n" if isinstance(val, np.ndarray) else ""),
            val,
            sep="",
        )