"""
DS/CMPSC 442 Project-4, Question-2
Implementing a Prediction Model for Diabetes Diagnosis

Author: Original implementation
Date: Fall 2025

This implementation uses only Python 3.9 built-in functions and the csv module.
No external ML libraries are used.
"""

import csv
import random
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)


def load_data(filename):
    """
    Load the dataset from CSV file.
    
    Args:
        filename (str): Path to the CSV file
        
    Returns:
        list: List of tuples (X1, X2, Y) where all values are integers
    """
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row
        for row in reader:
            x1 = int(float(row[0]))  # Glucose level (X1)
            x2 = int(float(row[1]))  # Blood pressure level (X2)
            y = int(row[2])           # Diabetes label (Y: 0 or 1)
            data.append((x1, x2, y))
    return data


def stratified_split(data, train_ratio=0.7):
    """
    Perform stratified train-test split to preserve class ratios.
    This ensures proportions of Y=0 and Y=1 are maintained in both subsets.
    
    Args:
        data (list): List of data points (X1, X2, Y)
        train_ratio (float): Ratio of training data (default 0.7 for 70%)
        
    Returns:
        tuple: (train_data, test_data)
    """
    # Separate data by class
    class_0 = [d for d in data if d[2] == 0]
    class_1 = [d for d in data if d[2] == 1]
    
    # Shuffle each class independently
    random.shuffle(class_0)
    random.shuffle(class_1)
    
    # Calculate split points
    train_size_0 = int(len(class_0) * train_ratio)
    train_size_1 = int(len(class_1) * train_ratio)
    
    # Split each class
    train_data = class_0[:train_size_0] + class_1[:train_size_1]
    test_data = class_0[train_size_0:] + class_1[train_size_1:]
    
    # Shuffle the combined datasets
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    return train_data, test_data


def compute_prior(train_data):
    """
    Compute P(Y) - prior probabilities of each class.
    
    This calculates:
    - P(Y=0): probability of no diabetes
    - P(Y=1): probability of diabetes
    
    Args:
        train_data (list): Training dataset
        
    Returns:
        dict: {0: P(Y=0), 1: P(Y=1)}
    """
    counts = {0: 0, 1: 0}
    for _, _, y in train_data:
        counts[y] += 1
    
    total = len(train_data)
    prior = {y: counts[y] / total for y in counts}
    
    return prior


def compute_conditional_prob(train_data, feature_index):
    """
    Compute P(Xi | Y) for a given feature using Laplace smoothing.
    
    This calculates the conditional probability of each feature value
    given the class label Y.
    
    Args:
        train_data (list): Training dataset
        feature_index (int): 0 for X1 (glucose), 1 for X2 (blood pressure)
        
    Returns:
        dict: Nested dictionary {y: {feature_value: probability}}
    """
    # Count occurrences of (feature_value, y)
    counts = defaultdict(lambda: defaultdict(int))
    class_counts = {0: 0, 1: 0}
    feature_values = set()
    
    for row in train_data:
        feature_value = row[feature_index]
        y = row[2]
        counts[y][feature_value] += 1
        class_counts[y] += 1
        feature_values.add(feature_value)
    
    # Apply Laplace smoothing (add-one smoothing) to handle zero probabilities
    # Formula: P(X=x|Y=y) = (count(X=x, Y=y) + 1) / (count(Y=y) + |X|)
    # where |X| is the number of unique values of X
    num_unique_values = len(feature_values)
    cond_prob = defaultdict(dict)
    
    for y in [0, 1]:
        for feature_value in feature_values:
            numerator = counts[y][feature_value] + 1
            denominator = class_counts[y] + num_unique_values
            cond_prob[y][feature_value] = numerator / denominator
    
    # Store smoothing factor for unseen values during testing
    cond_prob['_smoothing_'] = {
        0: 1 / (class_counts[0] + num_unique_values),
        1: 1 / (class_counts[1] + num_unique_values)
    }
    
    return cond_prob


def get_probability(cond_prob, y, feature_value):
    """
    Get conditional probability with fallback for unseen values.
    
    If a feature value wasn't seen during training, use the
    Laplace smoothing probability.
    
    Args:
        cond_prob (dict): Conditional probability table
        y (int): Class label (0 or 1)
        feature_value: Feature value to look up
        
    Returns:
        float: Probability P(feature_value | Y=y)
    """
    if feature_value in cond_prob[y]:
        return cond_prob[y][feature_value]
    else:
        # Use smoothed probability for unseen values
        return cond_prob['_smoothing_'][y]


def inference(x1, x2, prior, cond_x1, cond_x2):
    """
    Compute P(Y | X1=x1, X2=x2) using inference by enumeration.
    
    Based on the conditional independence assumption:
    P(X1, X2, Y) = P(Y) * P(X1 | Y) * P(X2 | Y)
    
    We compute:
    P(Y | X1, X2) = P(Y) * P(X1 | Y) * P(X2 | Y) / P(X1, X2)
    
    Where normalization ensures: P(Y=0|X1,X2) + P(Y=1|X1,X2) = 1
    
    Args:
        x1: Glucose level value
        x2: Blood pressure level value
        prior (dict): P(Y)
        cond_x1 (dict): P(X1 | Y)
        cond_x2 (dict): P(X2 | Y)
        
    Returns:
        dict: {0: P(Y=0|X1,X2), 1: P(Y=1|X1,X2)}
    """
    # Compute unnormalized joint probabilities for each value of Y
    unnormalized = {}
    for y in [0, 1]:
        prob_y = prior[y]
        prob_x1_given_y = get_probability(cond_x1, y, x1)
        prob_x2_given_y = get_probability(cond_x2, y, x2)
        # P(Y, X1, X2) = P(Y) * P(X1|Y) * P(X2|Y)
        unnormalized[y] = prob_y * prob_x1_given_y * prob_x2_given_y
    
    # Normalize to get P(Y | X1, X2)
    total = unnormalized[0] + unnormalized[1]
    normalized = {y: unnormalized[y] / total for y in [0, 1]}
    
    return normalized


def predict(posterior):
    """
    Make prediction based on posterior probabilities.
    
    Predict Y=1 if P(Y=1|X1,X2) > P(Y=0|X1,X2), otherwise predict Y=0.
    
    Args:
        posterior (dict): {0: P(Y=0|X), 1: P(Y=1|X)}
        
    Returns:
        int: Predicted class (0 or 1)
    """
    return 1 if posterior[1] > posterior[0] else 0


def calculate_accuracy(predictions, true_labels):
    """
    Calculate classification accuracy.
    
    Accuracy = Number of correct predictions / Total number of predictions
    
    Args:
        predictions (list): Predicted labels
        true_labels (list): True labels
        
    Returns:
        float: Accuracy (between 0 and 1)
    """
    correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    return correct / len(predictions)


def main():
    """
    Main function to execute all parts of Question 2.
    """
    # Load the dataset
    print("Loading Naive-Bayes-Classification-Data.csv...")
    data = load_data('Naive-Bayes-Classification-Data.csv')
    print(f"Total samples loaded: {len(data)}")
    
    # Perform stratified split: 70% training, 30% testing
    train_data, test_data = stratified_split(data, train_ratio=0.7)
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    # Count class distribution
    train_y0 = sum(1 for _, _, y in train_data if y == 0)
    train_y1 = sum(1 for _, _, y in train_data if y == 1)
    test_y0 = sum(1 for _, _, y in test_data if y == 0)
    test_y1 = sum(1 for _, _, y in test_data if y == 1)
    print(f"Training set - Y=0: {train_y0}, Y=1: {train_y1}")
    print(f"Testing set - Y=0: {test_y0}, Y=1: {test_y1}")
    print()
    
    # ========================================================================
    # Q2.1: Compute Conditional Probability Tables (CPTs)
    # ========================================================================
    print("=" * 80)
    print("Q2.1: Computing Conditional Probability Tables using Training Data")
    print("=" * 80)
    print()
    
    # 2.1.1: Compute P(Y)
    print("2.1.1 P(Y): Prior probabilities of diabetes diagnosis")
    print("-" * 80)
    prior = compute_prior(train_data)
    print(f"P(Y=0) [No Diabetes]  = {prior[0]:.6f}")
    print(f"P(Y=1) [Diabetes]     = {prior[1]:.6f}")
    print()
    
    # 2.1.2: Compute P(X1 | Y)
    print("2.1.2 P(X1 | Y): Conditional probabilities of glucose level given Y")
    print("-" * 80)
    cond_x1 = compute_conditional_prob(train_data, feature_index=0)
    
    # Remove smoothing metadata for counting
    num_unique_x1 = len([k for k in cond_x1[0].keys()])
    print(f"Number of unique glucose levels (X1) in training data: {num_unique_x1}")
    print(f"Conditional probability table P(X1 | Y) computed with Laplace smoothing")
    print()
    print("Sample conditional probabilities (first 5 glucose values):")
    sample_x1_values = sorted(list(cond_x1[0].keys()))[:5]
    for val in sample_x1_values:
        prob_y0 = cond_x1[0][val]
        prob_y1 = cond_x1[1][val]
        print(f"  X1={val:3d}: P(X1={val}|Y=0) = {prob_y0:.6f}, P(X1={val}|Y=1) = {prob_y1:.6f}")
    print()
    
    # 2.1.3: Compute P(X2 | Y)
    print("2.1.3 P(X2 | Y): Conditional probabilities of blood pressure level given Y")
    print("-" * 80)
    cond_x2 = compute_conditional_prob(train_data, feature_index=1)
    
    num_unique_x2 = len([k for k in cond_x2[0].keys()])
    print(f"Number of unique blood pressure levels (X2) in training data: {num_unique_x2}")
    print(f"Conditional probability table P(X2 | Y) computed with Laplace smoothing")
    print()
    print("Sample conditional probabilities (first 5 blood pressure values):")
    sample_x2_values = sorted(list(cond_x2[0].keys()))[:5]
    for val in sample_x2_values:
        prob_y0 = cond_x2[0][val]
        prob_y1 = cond_x2[1][val]
        print(f"  X2={val:3d}: P(X2={val}|Y=0) = {prob_y0:.6f}, P(X2={val}|Y=1) = {prob_y1:.6f}")
    print()
    
    # ========================================================================
    # Q2.2: Implementing Inference by Enumeration
    # ========================================================================
    print("=" * 80)
    print("Q2.2: Implementing Inference by Enumeration")
    print("=" * 80)
    print()
    
    # 2.2.1: Write code to answer the inference query P(Y | X1, X2)
    print("2.2.1 Inference Query: Computing P(Y | X1, X2)")
    print("-" * 80)
    print("Using the formula:")
    print("  P(Y | X1, X2) ∝ P(Y) * P(X1 | Y) * P(X2 | Y)")
    print()
    print("The conditional independence assumption allows us to decompose:")
    print("  P(X1, X2 | Y) = P(X1 | Y) * P(X2 | Y)")
    print()
    print(f"Computing posterior probabilities for all {len(test_data)} test samples...")
    
    # Compute posteriors for all test data points
    posteriors = []
    for x1, x2, _ in test_data:
        posterior = inference(x1, x2, prior, cond_x1, cond_x2)
        posteriors.append(posterior)
    
    print(f"Successfully computed P(Y | X1, X2) for {len(posteriors)} test points")
    print()
    
    # 2.2.2: Generate a lookup table for P(Y | X1, X2)
    print("2.2.2 Lookup Table: P(Y | X1, X2) for Test Data")
    print("-" * 80)
    print(f"{'X1':<8} {'X2':<8} {'P(Y=0|X1,X2)':<18} {'P(Y=1|X1,X2)':<18} {'True Y':<8}")
    print("-" * 80)
    
    # Display lookup table for all test samples (or first 30 if too many)
    display_limit = min(30, len(test_data))
    for i in range(display_limit):
        x1, x2, true_y = test_data[i]
        posterior = posteriors[i]
        print(f"{x1:<8} {x2:<8} {posterior[0]:<18.6f} {posterior[1]:<18.6f} {true_y:<8}")
    
    if len(test_data) > display_limit:
        print(f"... (showing first {display_limit} of {len(test_data)} test samples)")
    print()
    
    # ========================================================================
    # Q2.3: Generate Predictions
    # ========================================================================
    print("=" * 80)
    print("Q2.3: Generate Predictions and Compute Accuracy")
    print("=" * 80)
    print()
    
    # 2.3.1: Make predictions for each test data point
    print("2.3.1 Making Predictions")
    print("-" * 80)
    print("Prediction rule: Predict Y=1 if P(Y=1|X1,X2) > P(Y=0|X1,X2), else predict Y=0")
    print()
    
    predictions = []
    for posterior in posteriors:
        pred = predict(posterior)
        predictions.append(pred)
    
    true_labels = [y for _, _, y in test_data]
    
    # Show sample predictions
    print("Sample predictions (first 10 test samples):")
    print(f"{'X1':<8} {'X2':<8} {'P(Y=1|X1,X2)':<15} {'Predicted Y':<13} {'True Y':<10} {'Correct?'}")
    print("-" * 80)
    for i in range(min(10, len(test_data))):
        x1, x2, true_y = test_data[i]
        posterior = posteriors[i]
        pred = predictions[i]
        correct = "✓" if pred == true_y else "✗"
        print(f"{x1:<8} {x2:<8} {posterior[1]:<15.6f} {pred:<13} {true_y:<10} {correct}")
    print()
    
    # 2.3.2: Compute the model's accuracy
    print("2.3.2 Computing Model Accuracy")
    print("-" * 80)
    
    accuracy = calculate_accuracy(predictions, true_labels)
    correct_count = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    total_count = len(predictions)
    
    print(f"Total number of predictions: {total_count}")
    print(f"Number of correct predictions: {correct_count}")
    print(f"Number of incorrect predictions: {total_count - correct_count}")
    print()
    print(f"Accuracy = Number of correct predictions / Total number of predictions")
    print(f"Accuracy = {correct_count} / {total_count}")
    print(f"Accuracy = {accuracy:.6f}")
    print(f"Accuracy = {accuracy * 100:.2f}%")
    print()
    
    # Additional evaluation metrics
    print("Detailed Performance Breakdown:")
    print("-" * 80)
    tp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
    
    print(f"True Positives (correctly predicted diabetes):     {tp}")
    print(f"True Negatives (correctly predicted no diabetes):  {tn}")
    print(f"False Positives (incorrectly predicted diabetes):  {fp}")
    print(f"False Negatives (incorrectly predicted no diabetes): {fn}")
    print()
    
    print("=" * 80)
    print("Question 2 Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()