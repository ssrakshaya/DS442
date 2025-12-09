import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split


def load_dataset(filepath='Naive-Bayes-Classification-Data.csv'):

    """
    Load in the diabetes dataset from the CSV file
    Arguments: filepath - path to the csv file
    Returns: a datafram with the dataset
    """

    data = pd.read_csv(filepath)
    #Rename columns to match format of X1, X2, and Y
    data.columns = ['X1', 'X2', 'Y']
    return data


def stratified_split(data, test_ratio = 0.3, random_seed = 42):
    """
    Performing stratified split to make sure the training and testing sets are good sizes (70/30 split in this case)
    
    Arguments: 
    - data (which is hthe csv)
    - test_ratio - so 0.3 or 30% of the dataset is used for testing, while 70% fo training. 
    - random seed is a random value for reproducibility of the split

    returns:
    - train_data, test_data - which are the two dataframes with training and testing data
    """


    #VERSION UTILIZING SKLEARN
    #Separate features and target
    X = data[['X1', 'X2']]
    y = data['Y']

    #Use sklearn's train_test_split with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, stratify=y, random_state =random_seed)

    #Recombine features and target into DataFrames
    train_data = pd.concat([X_train, y_train], axis = 1).reset_index(drop=True)
    test_data = pd.concat([X_test, y_test], axis = 1).reset_index(drop=True)

    return train_data, test_data


    # #VERSION WITHOUT SK LEARN
    # np.random.seed(random_seed)
    
    # #Separate data by class
    # class_0 = data[data['Y'] == 0].copy()
    # class_1 = data[data['Y'] == 1].copy()
    
    # #Shuffle each class
    # class_0 = class_0.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # class_1 = class_1.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # #Calculate split indices
    # test_size_0 = int(len(class_0) * test_ratio)
    # test_size_1 = int(len(class_1) * test_ratio)
    
    # #Split each class
    # test_0 = class_0[:test_size_0]
    # train_0 = class_0[test_size_0:]
    
    # test_1 = class_1[:test_size_1]
    # train_1 = class_1[test_size_1:]
    
    # #Combine train and test sets
    # train_data = pd.concat([train_0, train_1], ignore_index=True)
    # test_data = pd.concat([test_0, test_1], ignore_index=True)
    
    # #Shuffle the combined sets
    # train_data = train_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # test_data = test_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # return train_data, test_data

def compute_prior_Y(train):
    """
    Computing the prior probabilities P(Y) for each class
    Arguments: train - training datafram
    returns - dictionary with P(Y=0) and P(Y=1)
    """

    total = len(train) #total training examples
    count_y1 = len(train[train['Y'] == 1]) #count how many trainign rows ahve Y=1
    count_y0 = len(train[train['Y'] == 0]) #cont how many have Y=0
    

    #Compute prior probabilities:
    #P(Y=0) = (# of rows where Y=0) / total rows
    #P(Y=1) = (# of rows where Y=1) / total rows
    priors = {
        0: count_y0 / total,
        1: count_y1 / total
    }

    return priors #dictionary with the two prior values


def compute_cpt(train, feature_column, target_column = "Y"):
    """
    compute the conditional probability table P(Feature | Y) with smoothing
    Arguments:
    - train - trainign dataframe
    - feature_column  name of the feature column, either X1 or X2
    - target_column - > name of the target column, Y1

    returns: nested dictionary which is the conditional probability table 
    - cpt[y][feature_value] = probability
    """

    cpt = defaultdict(lambda: defaultdict(float))
    
    #Get all unique feature values in training data
    unique_features = train[feature_column].unique()
    num_unique_features = len(unique_features)

    #Find the conditional probability table for each class
    for y in [0, 1]:
        #filter the training rows where Y=y
        data_y = train[train[target_column] == y]
        total_count_y = len(data_y) #number of rows where Y=y

        #count the numbr of times each feature appears given Y=y
        feature_counts = data_y[feature_column].value_counts().to_dict()

        #use laplace smoothing: P(X=x | Y=y) = (count(X=x, Y=y) + 1) / (count(Y=y) + |unique X values|)
        #laplace s used to to avoid zero probabilties which break 
        for feature_value in unique_features:
            count = feature_counts.get(feature_value, 0)  #0 if feature never appears with Y=y
            cpt[y][feature_value] = (count + 1) / (total_count_y + num_unique_features)
    
    return cpt

def compute_posterior(x1, x2, priors, cpt_x1, cpt_x2):
    """
    compute the posterior probabilities P(Y | X1=x1, X2=x2) using inference by enumeration.
    Arguments:
    - x1: glucose level vlaue
    - x2: blood pressure level vlaue 
    - priors: dictionary with P(Y)
    - cpt_x1: Conditional probability table for P(X1 | Y)
    - cpt_x2: Conditional probability table for P(X2 | Y)

    returns: dictionary with P(Y=0 | x1, x2) and P(Y=1 | x1, x2)
    """

    posteriors = {} #stores final normalized posterior values
    unnormalized = {} #stores numerator values before normalization

    #first: compute unnormalized posterior values for each class y=0 and y =1
    #Naive Bayes formula (unnormalized)
    #numerator = P(Y=y) * P(X1=x1 | Y=y) * P(X2=x2 | Y=y)
    #we do not normalize in this step
    for y in [0, 1]:
        #P(Y=y) * P(X1=x1 | Y=y) * P(X2=x2 | Y=y)
        prior_y = priors[y] #get prior probability P(Y=y)

        #Get conditional probabilities from CPT table (use small value if not in training data) so that you do not assign a zero probability
        p_x1_given_y = cpt_x1[y].get(x1, 1e-10)
        p_x2_given_y = cpt_x2[y].get(x2, 1e-10)

        #Compute the unnormalized joint probability
        unnormalized[y] = prior_y * p_x1_given_y * p_x2_given_y

    #now we can move onto normalization!
    #Step 2: Normalize to conver unnormalized numbers into actual probabilities
    #posterior formula: P(Y=y | X1, X2) = numerator_y / (numerator_0 + numerator_1) -> ensures that values sum to 1
    total = unnormalized[0] + unnormalized[1]
    
    if total > 0:
        #normally - will divide each unnormalized probability by the sume
        posteriors[0] = unnormalized[0] / total
        posteriors[1] = unnormalized[1] / total
    else:
        #Edge case: both probabilities are zero, so we use uniform distribution (split 0.5 0.5)
        posteriors[0] = 0.5
        posteriors[1] = 0.5
    
    return posteriors

def predict(test_data, priors, cpt_x1, cpt_x2):
    """
    generate predictions for test data and compute accuracyy
    Arguments: 
    - test_data: Test DataFrame
    - priors: Prior probabilities P(Y)
    - cpt_x1: CPT for P(X1 | Y)
    - cpt_x2: CPT for P(X2 | Y)
        
    Return: 
    - predictions: List of predicted labels
    - lookup_table: List of tuples (x1, x2, P(Y=1|x1,x2), P(Y=0|x1,x2))
    - accuracy: Classification accuracy
    """

    predictions = []    #Stores predicted class labels for each test example
    lookup_table = []   #Stores posterior probabilities for reporting or inspection

    #Iterate through every test example (row-by-row)
    for idx, row in test_data.iterrows():
        #Extract the feature values for this test sample
        x1 = row['X1']
        x2 = row['X2']
        
        #Compute posterior probabilities using Naive Bayes
        # posteriors[1] = P(Y=1 | X1=x1, X2=x2)
        # posteriors[0] = P(Y=0 | X1=x1, X2=x2)
        posteriors = compute_posterior(x1, x2, priors, cpt_x1, cpt_x2)
        
        #store information to the lookup table for debugging/printing
        lookup_table.append((x1, x2, posteriors[1], posteriors[0]))
        
        #Make prediction: If P(Y=1 | X1,X2) is larger, classify as 1 OR classify as 0.
        if posteriors[1] > posteriors[0]:
            predictions.append(1)
        else:
            predictions.append(0)
    
    #calculate accuracy by comparing the predicted values with the true label test_data['y']
    correct = sum(pred == true for pred, true in zip(predictions, test_data['Y']))
    accuracy = correct / len(predictions) #fraction of correct predictions
    
    return predictions, lookup_table, accuracy


def main():
    """
    Main function used to do the diabetes prediction
    """

    #load in dataset
    data = load_dataset("Naive-Bayes-Classification-Data.csv")

    #Perform stratified split
    train_data, test_data = stratified_split(data, test_ratio=0.3, random_seed=42)


    #QUESTION 2.1.1: Compute P(X)
    print("2.1.1")
    priors = compute_prior_Y(train_data)
    print(f"P(Y=0) = {priors[0]:.6f}")
    print(f"P(Y=1) = {priors[1]:.6f}")
    print()

    #QUESTION 2.1.2: Compute P(X1 | Y) (probability of x1 given y)
    print("2.1.2")
    cpt_x1 = compute_cpt(train_data, 'X1') #finding the conditional probability table
    print("P(X1 | Y) -> Conditional Probability Table for Glucose:")
    print("Format: P(X1=value | Y=class)")
    print()

    #Display some of the conditional probability table (first 10 unique X1 values for each class)
    unique_x1 = sorted(train_data['X1'].unique())[:10]
    #I have unique_x1 - a list/array of the unique possible values that X1 can take
    #I also have cpt_x1, which is a conditional probablity table for P(X1 | Y)
    for x1_val in unique_x1: #looping over every possible value of X1
        #for each value it prints P(X1 = value from loop | Y=0) or P(X1 = value from loop | Y=0)
        print(f" X1={x1_val}: P(X1={x1_val}|Y=0)={cpt_x1[0][x1_val]:.6f}, P(X1={x1_val}|Y=1)={cpt_x1[1][x1_val]:.6f}")
    print("->(CPT computed for all unique glucose values)")
    print()

    #QUESTION 2.1.3: COMPUTE P(X2 | Y)
    print("2.1.3")
    cpt_x2 = compute_cpt(train_data, 'X2') #finding the conditional probability table
    print("P(X2 | Y) - Conditional Probability Table for Blood Pressure:")
    print("Format: P(X2=value | Y=class)")
    print()
    #Display some of the conditional probability table (first 10 unique X2 values for each class)
    unique_x2 = sorted(train_data['X2'].unique())[:10]

    #unique_x2 is the list/array for unique values x2 can take
    for x2_val in unique_x2:
        #for each value it prints P(X2 = value from loop | Y=0) or P(X2 = value from loop | Y=0)
        print(f"X2={x2_val}: P(X2={x2_val}|Y=0)={cpt_x2[0][x2_val]:.6f}, P(X2={x2_val}|Y=1)={cpt_x2[1][x2_val]:.6f}")
    print("-> (CPT computed for all unique blood pressure values)")
    print()


    # QUESTION 2.2.1: Inference by Enumeration ==========
    print("2.2.1")
    print("Inference by Enumeration - Computing P(Y | X1, X2) for test data")
    print("Sample calculations for first 5 test examples:")
    print()
    for idx in range(min(5, len(test_data))): #looping thorough 5 test samples of the test data.
        row = test_data.iloc[idx]
        x1, x2 = row['X1'], row['X2']
        posteriors = compute_posterior(x1, x2, priors, cpt_x1, cpt_x2) #Finding the prior probability 
        print(f"Test example {idx+1}: X1={x1}, X2={x2}")
        print(f"  P(Y=1 | X1={x1}, X2={x2}) = {posteriors[1]:.6f}") #printing to 6 decimal places 
        print(f"  P(Y=0 | X1={x1}, X2={x2}) = {posteriors[0]:.6f}")
    print("-> (computed for all test examples)")
    print()

    #question 2.2.2: lOOKUP table
    print("2.2.2")
    print("Lookup Table for P(Y | X1, X2) on Test Data")
    print("Format: X1, X2, P(Y=1|X1,X2), P(Y=0|X1,X2)")
    print()
    predictions, lookup_table, accuracy = predict(test_data, priors, cpt_x1, cpt_x2)
    
    #Display first 10 entries of lookup table
    print("First 10 entries:")
    for i in range(min(10, len(lookup_table))): #whatever is smaller between 10 and the length of the lookup table
        x1, x2, p_y1, p_y0 = lookup_table[i]
        print(f"X1={x1}, X2={x2}: P(Y=1)={p_y1:.6f}, P(Y=0)={p_y0:.6f}")
    print(f"... (total {len(lookup_table)} test examples)")
    print()


    #QUESTION 2.3: Predictions and the accuracy of predictions
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