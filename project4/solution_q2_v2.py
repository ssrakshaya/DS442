"""
solution_q2.py
DS/CMPSC 442 - Artificial Intelligence - Project 4, Question 2
Naive Bayes Classification for Diabetes Prediction

This script implements a Bayesian Network-based classifier for diabetes prediction
using glucose and blood pressure measurements. The implementation includes:
- Manual stratified train-test split
- Conditional Probability Table (CPT) computation with Laplace smoothing
- Inference by enumeration for posterior probability calculation
- Prediction and accuracy evaluation

Author: Akshaya
"""

import pandas as pd
import numpy as np
from collections import defaultdict


def load_dataset(filepath='Naive-Bayes-Classification-Data.csv'):
    """
    Load the diabetes dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the dataset
    """
    data = pd.read_csv(filepath)
    # Rename columns to match expected format
    data.columns = ['X1', 'X2', 'Y']
    return data


def stratified_split(data, test_ratio=0.3, random_seed=42):
    """
    Perform stratified split to maintain class proportions in train and test sets.
    
    Args:
        data: DataFrame to split
        test_ratio: Proportion of data to use for testing (default 0.3)
        random_seed: Random seed for reproducibility (default 42)
        
    Returns:
        train_data, test_data: Two DataFrames containing training and test data
    """
    np.random.seed(random_seed)
    
    # Separate data by class
    class_0 = data[data['Y'] == 0].copy()
    class_1 = data[data['Y'] == 1].copy()
    
    # Shuffle each class
    class_0 = class_0.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    class_1 = class_1.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Calculate split indices
    test_size_0 = int(len(class_0) * test_ratio)
    test_size_1 = int(len(class_1) * test_ratio)
    
    # Split each class
    test_0 = class_0[:test_size_0]
    train_0 = class_0[test_size_0:]
    
    test_1 = class_1[:test_size_1]
    train_1 = class_1[test_size_1:]
    
    # Combine train and test sets
    train_data = pd.concat([train_0, train_1], ignore_index=True)
    test_data = pd.concat([test_0, test_1], ignore_index=True)
    
    # Shuffle the combined sets
    train_data = train_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return train_data, test_data


def compute_prior_Y(train):
    """
    Compute prior probabilities P(Y) for each class.
    
    Args:
        train: Training DataFrame
        
    Returns:
        Dictionary with P(Y=0) and P(Y=1)
    """
    total = len(train)
    count_y1 = len(train[train['Y'] == 1])
    count_y0 = len(train[train['Y'] == 0])
    
    priors = {
        0: count_y0 / total,
        1: count_y1 / total
    }
    
    return priors


def compute_cpt(train, feature_column, target_column='Y'):
    """
    Compute Conditional Probability Table P(Feature | Y) with Laplace smoothing.
    
    Args:
        train: Training DataFrame
        feature_column: Name of the feature column ('X1' or 'X2')
        target_column: Name of the target column (default 'Y')
        
    Returns:
        Nested dictionary: cpt[y][feature_value] = probability
    """
    cpt = defaultdict(lambda: defaultdict(float))
    
    # Get all unique feature values in training data
    unique_features = train[feature_column].unique()
    num_unique_features = len(unique_features)
    
    # Compute CPT for each class
    for y in [0, 1]:
        data_y = train[train[target_column] == y]
        total_count_y = len(data_y)
        
        # Count occurrences of each feature value given Y=y
        feature_counts = data_y[feature_column].value_counts().to_dict()
        
        # Apply Laplace smoothing: P(X=x|Y=y) = (count(X=x, Y=y) + 1) / (count(Y=y) + |unique X values|)
        for feature_value in unique_features:
            count = feature_counts.get(feature_value, 0)
            cpt[y][feature_value] = (count + 1) / (total_count_y + num_unique_features)
    
    return cpt


def compute_posterior(x1, x2, priors, cpt_x1, cpt_x2):
    """
    Compute posterior probabilities P(Y | X1=x1, X2=x2) using inference by enumeration.
    
    Args:
        x1: Glucose level value
        x2: Blood pressure level value
        priors: Dictionary containing P(Y)
        cpt_x1: CPT for P(X1 | Y)
        cpt_x2: CPT for P(X2 | Y)
        
    Returns:
        Dictionary with P(Y=0 | x1, x2) and P(Y=1 | x1, x2)
    """
    posteriors = {}
    unnormalized = {}
    
    # Compute unnormalized probabilities for each class
    for y in [0, 1]:
        # P(Y=y) * P(X1=x1 | Y=y) * P(X2=x2 | Y=y)
        prior_y = priors[y]
        
        # Get conditional probabilities (use small value if not in training data)
        p_x1_given_y = cpt_x1[y].get(x1, 1e-10)
        p_x2_given_y = cpt_x2[y].get(x2, 1e-10)
        
        unnormalized[y] = prior_y * p_x1_given_y * p_x2_given_y
    
    # Normalize to get posterior probabilities
    total = unnormalized[0] + unnormalized[1]
    
    if total > 0:
        posteriors[0] = unnormalized[0] / total
        posteriors[1] = unnormalized[1] / total
    else:
        # If both are zero, use uniform distribution
        posteriors[0] = 0.5
        posteriors[1] = 0.5
    
    return posteriors


def predict(test_data, priors, cpt_x1, cpt_x2):
    """
    Generate predictions for test data and compute accuracy.
    
    Args:
        test_data: Test DataFrame
        priors: Prior probabilities P(Y)
        cpt_x1: CPT for P(X1 | Y)
        cpt_x2: CPT for P(X2 | Y)
        
    Returns:
        predictions: List of predicted labels
        lookup_table: List of tuples (x1, x2, P(Y=1|x1,x2), P(Y=0|x1,x2))
        accuracy: Classification accuracy
    """
    predictions = []
    lookup_table = []
    
    for idx, row in test_data.iterrows():
        x1 = row['X1']
        x2 = row['X2']
        
        # Compute posterior probabilities
        posteriors = compute_posterior(x1, x2, priors, cpt_x1, cpt_x2)
        
        # Store in lookup table
        lookup_table.append((x1, x2, posteriors[1], posteriors[0]))
        
        # Make prediction: predict Y=1 if P(Y=1|x1,x2) > P(Y=0|x1,x2)
        if posteriors[1] > posteriors[0]:
            predictions.append(1)
        else:
            predictions.append(0)
    
    # Compute accuracy
    correct = sum(pred == true for pred, true in zip(predictions, test_data['Y']))
    accuracy = correct / len(predictions)
    
    return predictions, lookup_table, accuracy


def main():
    """
    Main function to orchestrate the diabetes prediction workflow.
    """
    # Load dataset
    data = load_dataset('Naive-Bayes-Classification-Data.csv')
    
    # Perform stratified split
    train_data, test_data = stratified_split(data, test_ratio=0.3, random_seed=42)
    
    # ========== 2.1.1: Compute P(Y) ==========
    print("2.1.1")
    priors = compute_prior_Y(train_data)
    print(f"P(Y=0) = {priors[0]:.6f}")
    print(f"P(Y=1) = {priors[1]:.6f}")
    print()
    
    # ========== 2.1.2: Compute P(X1 | Y) ==========
    print("2.1.2")
    cpt_x1 = compute_cpt(train_data, 'X1')
    print("P(X1 | Y) - Conditional Probability Table for Glucose:")
    print("Format: P(X1=value | Y=class)")
    print()
    # Display sample of CPT (first 10 unique X1 values for each class)
    unique_x1 = sorted(train_data['X1'].unique())[:10]
    for x1_val in unique_x1:
        print(f"X1={x1_val}: P(X1={x1_val}|Y=0)={cpt_x1[0][x1_val]:.6f}, P(X1={x1_val}|Y=1)={cpt_x1[1][x1_val]:.6f}")
    print("... (CPT computed for all unique glucose values)")
    print()
    
    # ========== 2.1.3: Compute P(X2 | Y) ==========
    print("2.1.3")
    cpt_x2 = compute_cpt(train_data, 'X2')
    print("P(X2 | Y) - Conditional Probability Table for Blood Pressure:")
    print("Format: P(X2=value | Y=class)")
    print()
    # Display sample of CPT (first 10 unique X2 values for each class)
    unique_x2 = sorted(train_data['X2'].unique())[:10]
    for x2_val in unique_x2:
        print(f"X2={x2_val}: P(X2={x2_val}|Y=0)={cpt_x2[0][x2_val]:.6f}, P(X2={x2_val}|Y=1)={cpt_x2[1][x2_val]:.6f}")
    print("... (CPT computed for all unique blood pressure values)")
    print()
    
    # ========== 2.2.1: Inference by Enumeration ==========
    print("2.2.1")
    print("Inference by Enumeration - Computing P(Y | X1, X2) for test data")
    print("Sample calculations for first 5 test examples:")
    print()
    for idx in range(min(5, len(test_data))):
        row = test_data.iloc[idx]
        x1, x2 = row['X1'], row['X2']
        posteriors = compute_posterior(x1, x2, priors, cpt_x1, cpt_x2)
        print(f"Test example {idx+1}: X1={x1}, X2={x2}")
        print(f"  P(Y=1 | X1={x1}, X2={x2}) = {posteriors[1]:.6f}")
        print(f"  P(Y=0 | X1={x1}, X2={x2}) = {posteriors[0]:.6f}")
    print("... (computed for all test examples)")
    print()
    
    # ========== 2.2.2: Lookup Table ==========
    print("2.2.2")
    print("Lookup Table for P(Y | X1, X2) on Test Data")
    print("Format: X1, X2, P(Y=1|X1,X2), P(Y=0|X1,X2)")
    print()
    predictions, lookup_table, accuracy = predict(test_data, priors, cpt_x1, cpt_x2)
    
    # Display first 10 entries of lookup table
    print("First 10 entries:")
    for i in range(min(10, len(lookup_table))):
        x1, x2, p_y1, p_y0 = lookup_table[i]
        print(f"X1={x1}, X2={x2}: P(Y=1)={p_y1:.6f}, P(Y=0)={p_y0:.6f}")
    print(f"... (total {len(lookup_table)} test examples)")
    print()
    
    # ========== 2.3: Predictions and Accuracy ==========
    print("2.3")
    print("Predictions and Model Evaluation")
    print()
    print("Sample predictions (first 10 test examples):")
    for i in range(min(10, len(predictions))):
        x1, x2 = test_data.iloc[i]['X1'], test_data.iloc[i]['X2']
        true_label = test_data.iloc[i]['Y']
        pred_label = predictions[i]
        p_y1, p_y0 = lookup_table[i][2], lookup_table[i][3]
        print(f"X1={x1}, X2={x2}: Predicted Y={pred_label}, True Y={true_label}, P(Y=1)={p_y1:.4f}")
    print("...")
    print()
    print(f"Total test examples: {len(test_data)}")
    print(f"Correct predictions: {sum(pred == true for pred, true in zip(predictions, test_data['Y']))}")
    print(f"Model Accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)")
    print()


if __name__ == "__main__":
    main()