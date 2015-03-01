import numpy as np
import pandas as pd 
import datetime
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn import cross_validation

def read_bagofwords_dat(myfile, numofemails=10000):
    bagofwords = np.fromfile(myfile, dtype=np.uint8, count=-1, sep="")
    bagofwords=np.reshape(bagofwords,(numofemails,-1))
    return bagofwords

with open('trec07p_data/Train/train_emails_classes_100.txt') as f:
    y_train_str = f.read().split('\n')

def replace(w):
    if w == 'NotSpam':
        return 0
    elif w == 'Spam':
        return 1
    else:
        print('wat')
        return -1

data = np.c_[np.asarray([replace(w) for w in y_train_str]),read_bagofwords_dat('trec07p_data/Train/train_emails_bag_of_words_100.dat', numofemails=45000)]
np.random.shuffle(data)

# testing with random 500
# data = data[0:500]

N = int(data.shape[0])

# We use K-Fold CV to evaluate each model and pick the best one.  Then, for that model, we find the actual test accuracy.
F = 5 # 5 folds
# record the number of incorrect classifications for each fold
cv_scores = pd.DataFrame(columns = ['f' + str(i) for i in range(0, 5)] + ['mean'], index=['NB', 'LOG', 'AdB'])

cv_scores = joblib.load('cv_scores.pkl')
#### Naive Bayes
f = 0
for train_index, test_index in kf:
    clf = MultinomialNB()
    start = datetime.datetime.now()
    print("Starting NB f" + str(f) + " fit")
    clf.fit(data[train_index, 1:], data[train_index, 0])
    end = datetime.datetime.now()
    delt = end - start
    # train_times.loc['NB'][f] = delt
    print("NB f" + str(f) + " fit took " + str(delt))

    start = datetime.datetime.now()
    print("Starting NB f" + str(f) + " predict")
    y_pred = clf.predict(data[test_index, 1:])
    end = datetime.datetime.now()
    delt = end - start

    cv_scores.loc['NB'][f] = sum(abs(y_pred - data[test_index, 0]))
    print("NB f" + str(f) + " predict took " + str(delt) + ": " + str(cv_scores.loc['NB'][f]) + " incorrect")
    joblib.dump(cv_scores, 'cv_scores.pkl')
    f += 1


#### Logistic Regression
f = 0
for train_index, test_index in kf:
    clf = linear_model.LogisticRegression()
    start = datetime.datetime.now()
    print("Starting LOG f" + str(f) + " fit")
    clf.fit(data[train_index, 1:], data[train_index, 0])
    end = datetime.datetime.now()
    delt = end - start
    print("LOG f" + str(f) + " fit took " + str(delt))

    start = datetime.datetime.now()
    print("Starting LOG f" + str(f) + " predict")
    y_pred = clf.predict(data[test_index, 1:])
    end = datetime.datetime.now()
    delt = end - start

    cv_scores.loc['LOG'][f] = sum(abs(y_pred - data[test_index, 0]))
    print("LOG f" + str(f) + " predict took " + str(delt) + ": " + str(cv_scores.loc['LOG'][f]) + " incorrect")
    joblib.dump(cv_scores, 'cv_scores.pkl')
    f += 1


#### AdaBoost using 50 decision trees
kf = cross_validation.KFold(N, n_folds=F)
f = 0
for train_index, test_index in kf:
    clf = AdaBoostClassifier(n_estimators=50)
    start = datetime.datetime.now()
    print("Starting AdB (50 trees) f" + str(f) + " fit")
    clf.fit(data[train_index, 1:], data[train_index, 0])
    end = datetime.datetime.now()
    delt = end - start
    print("AdB f" + str(f) + " fit took " + str(delt))

    start = datetime.datetime.now()
    print("Starting AdB f" + str(f) + " predict")
    y_pred = clf.predict(data[test_index, 1:])
    end = datetime.datetime.now()
    delt = end - start

    cv_scores.loc['AdB'][f] = sum(abs(y_pred - data[test_index, 0]))
    print("AdB f" + str(f) + " predict took " + str(delt) + ": " + str(cv_scores.loc['AdB'][f]) + " incorrect")
    joblib.dump(cv_scores, 'cv_scores.pkl')
    f += 1

# calculate the mean CV score
cv_scores['mean'] = cv_scores.iloc[:, 0:F].mean(1)

# save cv_scores again
joblib.dump(cv_scores, 'cv_scores.pkl')
###########

# now train the best classifier on the full set of training data
# (also save it)
cv_scores = joblib.load('cv_scores.pkl')
# retrieve the classifier with the highest mean CV score
best_clf_name = cv_scores.loc[:, 'mean'].idxmin()

clf = linear_model.LogisticRegression()
start = datetime.datetime.now()
print("Starting LOG full training fit")
clf.fit(data[:, 1:], data[:, 0])
end = datetime.datetime.now()
delt = end - start
print("LOG full training fit took " + str(delt))
joblib.dump(clf, 'best_clf.pkl')

# and get the final test accuracy
with open('trec07p_data/Test/test_emails_classes_0.txt') as f:
    y_test_str = f.read().split('\n')
y_test = np.asarray([replace(w) for w in y_test_str])
X_test = read_bagofwords_dat('trec07p_data/Test/test_emails_bag_of_words_0.dat', numofemails=5000)
start = datetime.datetime.now()
print("Starting LOG full testing predict")
y_pred = clf.predict(X_test)
end = datetime.datetime.now()
delt = end - start
print("LOG full testing predict took " + str(delt))
errors = sum(abs(y_pred - y_test))
test_acc = 1 - (float(errors) / len(y_test))