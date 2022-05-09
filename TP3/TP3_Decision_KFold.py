from sklearn.model_selection import KFold
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier


#LOAD DATASET
dataset = datasets.load_iris()

X = pd.DataFrame(data = dataset['data'], columns = dataset['feature_names'])
y = pd.DataFrame(data = dataset['target'], columns = ['target'])

# Instantiating the K-Fold cross validation object with 5 folds
k_folds = KFold(n_splits = 5, shuffle = True, random_state = 42)

# Iterating through each of the folds in K-Fold
for train_index, val_index in k_folds.split(X):
    # Splitting the training set from the validation set for this specific fold
    X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Instantiating a Decision Tree Classifier model
    clf = DecisionTreeClassifier()

    # Fitting the X_train and y_train datasets to the Decision Tree Classifier model
    clf.fit(X_train, y_train)

    # Getting inferential predictions for the validation dataset
    y_pred =  clf.predict(X_val)

    # Generating validation metrics by comparing the inferential predictions (val_preds) to the actuals (y_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    val_confusion_matrix = confusion_matrix(y_val, y_pred)

    print(f'Accuracy Score: {val_accuracy}')
    print(f'Confusion Matrix: \n{val_confusion_matrix}')