import math
import numpy as np
import pandas as pd
from typing import List, Sequence, Union
from DecisionTree import learn_decision_tree, predict, DecisionNode

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
    
            tree = learn_decision_tree(
                X=sampled_X,
                y=sampled_y,
                attrs=np.random.choice(X.columns, self.max_features, replace=False),
                y_parent=sampled_y
            )

            self.trees.append(tree)
        
    def forest_predict(self, X: pd.DataFrame):
        #predictions from all trees (columns are index of tree, row values are the predicted labels for given sample)
        tree_predictions = pd.DataFrame({i: predict(tree, X) for i, tree in enumerate(self.trees)})

        #majority label
        return tree_predictions.mode(axis=1)[0] 

            

        


