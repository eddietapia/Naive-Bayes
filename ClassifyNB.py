from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):   
    # Create classifier
    clf = GaussianNB()
    # Fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)
    
    # If we have a test data, we can figure out the accuracy 
    # accuracy = clf.score(features_test, labels_test)

    # Return the fit classifier
    return clf