import pandas as pd
from sklearn.model_selection import KFold
from sklearn import naive_bayes, datasets
from sklearn.metrics import confusion_matrix, accuracy_score

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

    # Instantiating a Multinomial Naive_Bayes Classifier model
    nb = naive_bayes.MultinomialNB(fit_prior=True)

    # Fitting the X_train and y_train datasets to the Multinomial Naive_Bayes Classifier model
    nb.fit(X_train, y_train)

    # Getting inferential predictions for the validation dataset
    y_pred =  nb.predict(X_val)

    # Generating validation metrics by comparing the inferential predictions (val_preds) to the actuals (y_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    val_confusion_matrix = confusion_matrix(y_val, y_pred)

    print(f'Accuracy Score: {val_accuracy}')
    print(f'Confusion Matrix: \n{val_confusion_matrix}')