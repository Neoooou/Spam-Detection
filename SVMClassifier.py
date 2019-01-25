from SpamDetection.preprocessing import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings

def tune_SVM(X, y, tuned_parameters):
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.33, random_state=44)
    scores = ['precision', 'recall']
    warnings.filterwarnings('ignore')
    for sc in scores:
        print("Tuning parameters for {0}".format(sc))
        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % sc, verbose=99)
        clf.fit(X_trn, y_trn)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_tst, clf.predict(X_tst)
        print(classification_report(y_true, y_pred))
        print()






y, msgs = get_data('spam.csv')
print('spam percentage: %.2f' % (y.count(1) / len(y)))


vectorizer = TfidfVectorizer(stop_words='english')

# vectorize data, then divide data into training and testing data set
X = vectorizer.fit_transform(msgs).toarray()
# split data into training and testing data set
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.33, random_state=44)

#tune the model in order to  find the optimized parameters
# parameters = [{'kernel':['sigmoid', 'linear', 'poly', 'rbf'], 'gamma':[1e-1,1e-2,1e-3],
#                     'C': [1, 10, 30, 50,100]}]
# tune_SVM(X, y, parameters)


clf = SVC(C=1, gamma=0.1, verbose=True, kernel='linear')
clf.fit(X_trn, y_trn)
y_pred = clf.predict(X_tst)

sc = accuracy_score(y_tst, y_pred)
print("Accuracy score %.4f" % sc)


precision = precision_score(y_tst, y_pred)
print('Precision score: %.4f ' % precision)

recall_score = recall_score(y_tst, y_pred, average='macro')
print('Recall score: %.4f' % recall_score)

f1_sc = f1_score(y_tst, y_pred)
print("F1 score: {:.4f}".format(f1_sc))

cm = confusion_matrix(y_tst, y_pred)
print("Confusion Matrix:")
print(cm)



