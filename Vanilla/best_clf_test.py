import numpy as np
import pandas as pd 
import datetime
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# load best classifier from CV
clf = joblib.load('best_clf.pkl')

def read_bagofwords_dat(myfile, numofemails=10000):
    bagofwords = np.fromfile(myfile, dtype=np.uint8, count=-1, sep="")
    bagofwords=np.reshape(bagofwords,(numofemails,-1))
    return bagofwords

def replace(w):
    if w == 'NotSpam':
        return 0
    elif w == 'Spam':
        return 1
    else:
        print('wat')
        return -1

# and get the final test accuracy
with open('trec07p_data/Test/test_emails_classes_0.txt') as f:
    y_test_str = f.read().split('\n')
y_test = np.asarray([replace(w) for w in y_test_str])
X_test = read_bagofwords_dat('trec07p_data/Test/test_emails_bag_of_words_0.dat',
    numofemails=5000)
start = datetime.datetime.now()
print("Starting LOG full testing predict")
y_pred = clf.predict(X_test)
end = datetime.datetime.now()
delt = end - start
print("LOG full testing predict took " + str(delt))

errors = sum(abs(y_pred - y_test))
test_acc = 1 - (float(errors) / len(y_test))


# Compute ROC curve
roc = roc_curve(y_test, y_pred)

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(roc, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()