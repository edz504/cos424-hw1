import numpy as np
import datetime

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

data = np.c_[np.asarray([replace(w) for w in y_train_str]), read_bagofwords_dat('trec07p_data/Train/train_emails_bag_of_words_100.dat', numofemails=45000)]
np.random.shuffle(data)

# testing with random 1000
# data = data[0:1000]

# separate training and validation sets
N = int(data.shape[0])
train_index = np.array(range(0, int(0.8 * N)))
valid_index = np.array(range(int(0.8 * N), N))

cv_scores = {}
times = {}

#### Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
start = datetime.datetime.now()
clf.fit(data[train_index, 1:], data[train_index, 0])
end = datetime.datetime.now()
delt = end - start
times['NB'] = delt
y_pred = clf.predict(data[valid_index, 1:])
cv_scores['NB'] = sum(abs(y_pred - data[valid_index, 0]))
print("NB took " + str(delt))

#### KNN using k = sqrt(N)
from sklearn.neighbors import KNeighborsClassifier
k = int(np.sqrt(N))
clf = KNeighborsClassifier(n_neighbors=k)
start = datetime.datetime.now()
clf.fit(data[train_index, 1:], data[train_index, 0])
end = datetime.datetime.now()
delt = end - start
times['KNN'] = delt
y_pred = clf.predict(data[valid_index, 1:])
cv_scores['KNN'] = sum(abs(y_pred - data[valid_index, 0]))
print("KNN took " + str(delt))

#### AdaBoost using 50 decision trees
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=50)
start = datetime.datetime.now()
clf.fit(data[train_index, 1:], data[train_index, 0])
end = datetime.datetime.now()
delt = end - start
times['AdB'] = delt
y_pred = clf.predict(data[valid_index, 1:])
cv_scores['AdB'] = sum(abs(y_pred - data[valid_index, 0]))
print("AdB took " + str(delt))

from sklearn.externals import joblib
joblib.dump(cv_scores, 'cv_scores.pkl')