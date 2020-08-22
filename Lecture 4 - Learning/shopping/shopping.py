import csv
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Read the data from the csv file
    df = pd.read_csv(filename)
    # Save the revenue series as integers: 
    labels = list(df['Revenue'].astype(int))

    # Convert months' names into numbers
    # print(df.Month.value_counts()) This is a very useful command to see what is in a column of a dataframe, with that I identified that June is written as June in the dataframe
    monthsDict = {'JAN':0,'FEB':1,'MAR':2,'APR':3,'MAY':4,'JUNE':5,'JUL':6,'AUG':7,'SEP':8,'OCT':9,'NOV':10,'DEC':11}
    df.Month = df.Month.str.upper().map(monthsDict)
   
    # Convert visitortype to 1 only if it is a returning visitor, otherwise write 0
    df.VisitorType = df.VisitorType.map(lambda x: 1 if x == 'Returning_Visitor' else 0)
    # Convert weekend booleans to integers
    df.Weekend = df.Weekend.astype(int)

    # Remove the revenue column from the dataframe
    df.drop(columns = ['Revenue'], axis = 1, inplace = True)
    
    # Contruct the list of lists for the evidence
    evidence = []   
    for i in range(df.shape[0]):
        evidence.append(list(df.iloc[i]))
    
    # Return the tuple of the processed data
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model




def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    identifiedTruePositives = 0
    identifiedTrueNegatives = 0
    totalPositives = 0
    totalNegatives = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            totalPositives += 1
            if labels[i] == predictions[i]:
                identifiedTruePositives += 1
        else:
            totalNegatives += 1
            if labels[i] == predictions[i]:
                identifiedTrueNegatives += 1
    sensitivity = identifiedTruePositives / totalPositives
    specificity = identifiedTrueNegatives / totalNegatives

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
