"""
CS311 Programming Assignment 5: Decision Trees

Full Name: Hedavam Solano

Description of performance on Hepatitis data set:

TODO: Answer the questions included in the assignment

Description of Adult dataset discretization and selected features:

TODO: Answer the questions included in the assignment

Potential effects of using your model for marketing a cash-back credit card:

TODO: Answer the questions included in the assignment
"""

import argparse, os, random, sys
from typing import Any, Dict, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

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
        """Return predicted label for array-like example x"""
        #Implement prediction based on value of self.attr in x till we reach leaf node
        attr_value = x[self.attr]
        subtree = self.branches[attr_value]
        return subtree.predict(x)


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
        """Return predicted label for array-like example x"""
        return self.label

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Label=", self.label)




def information_gain(X: pd.DataFrame, y: pd.Series, attr: str) -> float:
    """Return the expected reduction in entropy from splitting X,y by attr"""

    #before split
    label1_prop = y.value_counts(normalize=True).get(1,0) #(p/p+n) - treat label 1 as p
    initial_entropy = entropy_helper(label1_prop)

    #split...?
    splits = X.groupby(attr, observed = False)

    #after split
    remainder_entropy = 0.0
    for split_attr, split_example_indices in splits.groups.items():
        #find labels for new split
        split_y = y.loc[split_example_indices]

        split_label1_prop = split_y.value_counts(normalize=True).get(1,0) #p_k/p_k + n_k
        split_entropy = entropy_helper(split_label1_prop)

        weight = len(split_y) / len(y) #p_k + n_k / p + n

        remainder_entropy += weight * split_entropy

    info_gain = initial_entropy - remainder_entropy
        
    return info_gain

def entropy_helper(q: float) -> float:
    #if the split gives us no gain cuz all examples have labels of same class
    if q == 0 or q == 1:
        return 0
    
    entropy = -(q * np.log(q) + (1-q) * np.log(1-q))
    return entropy

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

    #Base Cases - create leaf nodes

    #return majority class of parent examples
    if X.empty:
        return DecisionLeaf(label = y_parent.mode()[0]) #in case there's many modes (same num labels for diff splits, pick first)
    
    #if all examples have the same classification then return the classification
    elif y.nunique() == 1:
        return DecisionLeaf(label = y.iloc[0])
    
    #if attributes is empty then return PLURALITY-VALUE(examples)
    elif len(attrs) == 0: 
       return DecisionLeaf(label = y.mode()[0])
    
    #Recursive Step - create branches based on attributes
    else:
        #find index of attribute with largest information gain & get rid of it for future splitting
        attr_information_gains = [information_gain(X, y, attr) for attr in attrs]

        best_attr_index = np.argmax(attr_information_gains)
        best_attr = attrs[best_attr_index]

        remaining_attributes = [attr for attr in attrs if attr != attrs[best_attr_index]]

        #create a new decision tree w/ attribute A as root
        tree = DecisionBranch(best_attr,{})

        #split and create subtrees
        for split_attr, split_example_indices in X.groupby(best_attr, observed = False).groups.items():
            #grab split's examples and labels
            split_X = X.loc[split_example_indices]
            split_y = y.loc[split_example_indices]

            subtree = learn_decision_tree(split_X, split_y, remaining_attributes, y)

            #add a branch to tree with label (A = v) and subtree subtree
            tree.branches[split_attr] = subtree

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


def load_adult(feature_file: str, label_file: str):

    # Load the feature file
    examples = pd.read_table(
        feature_file,
        dtype={
            "age": int,
            "workclass": "category",
            "education": "category",
            "marital-status": "category",
            "occupation": "category",
            "relationship": "category",
            "race": "category",
            "sex": "category",
            "capital-gain": int,
            "capital-loss": int,
            "hours-per-week": int,
            "native-country": "category",
        },
    )
    labels = pd.read_table(label_file).squeeze().rename("label")


    # TODO: LAST! SIMPLE! Select columns and choose a discretization for any continuous columns. Our decision tree algorithm
    # only supports discretized features and so any continuous columns (those not already "category") will need
    # to be discretized.

    # For example the following discretizes "hours-per-week" into "part-time" [0,40) hours and
    # "full-time" 40+ hours. Then returns a data table with just "education" and "hours-per-week" features.

    examples["hours-per-week"] = pd.cut(
        examples["hours-per-week"],
        bins=[0, 40, sys.maxsize],
        right=False,
        labels=["part-time", "full-time"],
    )

    return examples[["education", "hours-per-week"]], labels


# You should not need to modify anything below here


def load_examples(
    feature_file: str, label_file: str, **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load example features and labels. Additional arguments are passed to
    the pandas.read_table function.

    Args:
        feature_file (str): Delimited file of categorical features
        label_file (str): Single column binary labels. Column name will be renamed to "label".

    Returns:
        Tuple[pd.DataFrame,pd.Series]: Tuple of features and labels
    """
    return (
        pd.read_table(feature_file, dtype="category", **kwargs),
        pd.read_table(label_file, **kwargs).squeeze().rename("label"),
    )


def compute_metrics(y_true, y_pred):
    """Compute metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.

    Returns:
        dict: Dictionary of metrics in including confusion matrix, accuracy, recall, precision and F1
    """
    return {
        "confusion": metrics.confusion_matrix(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test decision tree learner")
    parser.add_argument(
        "-p",
        "--prefix",
        default="small1",
        help="Prefix for dataset files. Expects <prefix>.[train|test]_[data|label].txt files (except for adult). Allowed values: small1, tennis, hepatitis, adult.",
    )
    parser.add_argument(
        "-k",
        "--k_splits",
        default=10,
        type=int,
        help="Number of splits for stratified k-fold testing",
    )

    args = parser.parse_args()

    if args.prefix != "adult":
        # Derive input files names for test sets
        train_data_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.train_data.txt"
        )
        train_labels_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.train_label.txt"
        )
        test_data_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.test_data.txt"
        )
        test_labels_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.test_label.txt"
        )

        # Load training data and learn decision tree
        train_data, train_labels = load_examples(train_data_file, train_labels_file)
        tree = fit(train_data, train_labels)
        tree.display()

        # Load test data and predict labels with previously learned tree
        test_data, test_labels = load_examples(test_data_file, test_labels_file)
        pred_labels = predict(tree, test_data)

        # Compute and print accuracy metrics
        predict_metrics = compute_metrics(test_labels, pred_labels)
        for met, val in predict_metrics.items():
            print(
                met.capitalize(),
                ": ",
                ("\n" if isinstance(val, np.ndarray) else ""),
                val,
                sep="",
            )
    else:
        # We use a slightly different procedure with "adult". Instead of using a fixed split, we split
        # the data k-ways (preserving the ratio of output classes) and test each split with a Decision
        # Tree trained on the other k-1 splits.
        data_file = os.path.join(os.path.dirname(__file__), "data", "adult.data.txt")
        labels_file = os.path.join(os.path.dirname(__file__), "data", "adult.label.txt")
        data, labels = load_adult(data_file, labels_file)

        scores = []

        kfold = StratifiedKFold(n_splits=args.k_splits)
        for train_index, test_index in kfold.split(data, labels):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            tree = fit(X_train, y_train)
            y_pred = predict(tree, X_test)
            scores.append(metrics.accuracy_score(y_test, y_pred))

            tree.display()

        print(
            f"Mean (std) Accuracy (for k={kfold.n_splits} splits): {np.mean(scores)} ({np.std(scores)})"
        )
