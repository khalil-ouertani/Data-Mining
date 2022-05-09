import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn import naive_bayes, datasets
from sklearn.metrics import confusion_matrix, accuracy_score

#LOAD DATASET
dataset = datasets.load_iris()
X = pd.DataFrame(data = dataset['data'], columns = dataset['feature_names'])
y = pd.DataFrame(data = dataset['target'], columns = ['target'])

#TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
# X_test will take random 25% rows from the dataset


# Instantiating a Multinomial Naive_Bayes Classifier model
nb = naive_bayes.MultinomialNB(fit_prior=True)

# Fitting the X_train and y_train datasets to the Multinomial Naive_Bayes Classifier model
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

#METRICS
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy : ", accuracy)
print("CONFUSION MATRIX : \n", cm)