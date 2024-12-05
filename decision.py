import numpy as np
import pandas as pd
import json

class TreeLeaf:
    '''Object that represents a leaf in the decision tree'''
    def __init__(self, prediction, probability):
        self.prediction = prediction
        self.probability = probability
        
    def predict(self, _):
        return self.prediction, self.probability
    
class TreeNode:
    def __init__(self, feature, branches, majority_label, majority_probability):
        self.feature = feature
        self.branches = branches
        self.majority_label = majority_label
        self.majority_probability = majority_probability
        
    def predict(self, x):
        feature_val = x.iloc[self.feature]
        if feature_val in self.branches:
            return self.branches[feature_val].predict(x)
        return self.majority_label, self.majority_probability
    
class NonBinaryDecisionTree:
    def __init__(self):
        self.tree = None
        
    def create_tree(self, data: pd.DataFrame, depth=0, max_depth=20):
        """
        Recursively create the decision tree with depth-based stopping.
        """
        label_col = data.columns[-1]
        labels = data[label_col]
        
        # Stopping condition: Maximum depth reached
        if depth >= max_depth:
            majority_label = labels.mode().iloc[0]
            probability = (labels == majority_label).mean()
            return TreeLeaf(prediction=majority_label, probability=probability)
        
        # Stopping condition: All labels are the same
        if labels.nunique() == 1:
            return TreeLeaf(prediction=labels.iloc[0], probability=1.0)
        
        # Stopping condition: No features remain
        if data.shape[1] <= 1:
            majority_label = labels.mode().iloc[0]
            probability = (labels == majority_label).mean()
            return TreeLeaf(prediction=majority_label, probability=probability)
        
        # Find the best feature to split on
        features = data.columns[:-1]
        best_feature, _, best_sub = self.find_best_split(data[features], labels)
        
        # Stopping condition: No valid splits found
        if best_feature is None:
            majority_label = labels.mode().iloc[0]
            probability = (labels == majority_label).mean()
            return TreeLeaf(prediction=majority_label, probability=probability)
        
        # Create branches for each unique value of the best feature
        branches = {}
        feature_name = features[best_feature]
        majority_label = labels.mode().iloc[0]
        majority_probability = (labels == majority_label).mean()
        for val, sub in best_sub.items():
            if sub.empty:  # Handle empty subsets
                branches[val] = TreeLeaf(prediction=majority_label, probability=majority_probability)
            else:
                sub_labels = labels.loc[sub.index]
                sub = sub.assign(**{label_col: sub_labels})
                branches[val] = self.create_tree(sub, depth + 1, max_depth)
        return TreeNode(feature=best_feature, branches=branches, majority_label=majority_label, majority_probability=majority_probability)
    
    def find_best_split(self, data: pd.DataFrame, labels: pd.Series):
        best_feature = None
        best_entropy = float('inf')
        best_subsets = None
        for feature, feature_name in enumerate(data.columns):
            subsets = self.split(data, feature_name)
            entropy = self.partition_entropy([labels.loc[i.index] for i in subsets.values()])
            # Skip features that create no improvement
            if entropy >= best_entropy:
                continue
            best_entropy = entropy
            best_feature = feature
            best_subsets = subsets
        return best_feature, best_entropy, best_subsets
    
    def split(self, data: pd.DataFrame, feature_name: str):
        return {value: data[data[feature_name] == value] for value in data[feature_name].unique()}
        
    def partition_entropy(self, subsets):
        total_count = sum(len(i) for i in subsets)
        entropy = sum(len(i) / total_count * self.entropy(i.value_counts(normalize=True)) for i in subsets if len(i) > 0)
        return entropy
    
    def entropy(self, chance):
        return -np.sum(chance * np.log2(chance + 1e-9))
    
    def train(self, data: pd.DataFrame, max_depth=20):
        self.tree = self.create_tree(data, depth=0, max_depth=max_depth)
        
    def predict_one(self, x):
        return self.tree.predict(x)
    
    def predict(self, data: pd.DataFrame):
        predictions = [self.predict_one(row) for _, row in data.iterrows()]
        return np.array([pred for pred, _ in predictions]), np.array([prob for _, prob in predictions])
    
def assign_labels_by_rank(df: pd.DataFrame, rank_column: str = "rank"):
    '''Assigns labels to a dataframe based on the rank of the row'''
    df["labels"] = pd.qcut(df[rank_column], q=5, labels=[0, 1, 2, 3, 4])
    return df

def generate_bins_from_quartiles(df, columns):
    """
    Automatically generate 4 bins for numeric columns using quartiles.
    """
    bins = {}
    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            min_val = df[column].min()
            max_val = df[column].max()
            q1 = df[column].quantile(0.25)
            q2 = df[column].quantile(0.50)
            q3 = df[column].quantile(0.75)
            
            # Define 4 bins
            bin_edges = [min_val, q1, q2, q3, max_val]
            bins[column] = sorted(set(bin_edges))  # Ensure bin edges are unique and sorted
    return bins

def bucketize_columns(data: pd.DataFrame, bins: dict):
    '''Bucketize the columns in the dataframe based on the bins'''
    for column, bin_edges in bins.items():
        bin_labels = range(len(bin_edges) - 1)  # Generate 4 labels for 4 bins: 0, 1, 2, 3
        data[column] = pd.cut(data[column], bins=bin_edges, labels=bin_labels, include_lowest=True)
    return data

if __name__ == "__main__":
    # Load the data from a JSON file into a pandas DataFrame
    with open("tracks.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Assign labels based on their ranks
    df = assign_labels_by_rank(df, rank_column="rank")
    print("Data with Labels:\n", df)
    
    # Create bins based on quartiles for int and float dtype columns
    numeric_columns = ["rank", "duration_ms", "popularity", "acousticness", "danceability", "energy", "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]
    bins = generate_bins_from_quartiles(df, numeric_columns)
    
    # Bucketize columns
    df = bucketize_columns(df, bins)
    
    # Exclude the track_name for training
    training_data = df.drop(columns=["track_name"])
    
    # Train the decision tree with a maximum depth
    dt = NonBinaryDecisionTree()
    dt.train(training_data, max_depth=10)
    
    # Predict the labels for the training data
    predictions, probabilities = dt.predict(training_data.drop(columns=["labels"]))
    df["predictions"] = predictions
    df["prediction_probabilities"] = probabilities
    print("Data with Predictions and Probabilities:\n", df[["track_name", "labels", "predictions", "prediction_probabilities"]])
    
    # Print rows where prediction_probabilities isn't 1
    uncertain_predictions = df[df["prediction_probabilities"] != 1]
    if not uncertain_predictions.empty:
        print("Rows with Uncertain Predictions:\n", uncertain_predictions)
    
    none_predictions = df[df["predictions"] == "None"]
    if not none_predictions.empty:
        print("Rows with 'None' Predictions:\n", none_predictions)
    print("None of the rows contain 'None' for predictions.")