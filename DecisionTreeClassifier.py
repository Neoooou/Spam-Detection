from SpamDetection.preprocessing import get_data
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score





y, msgs = get_data('spam.csv')
print('spam percentage: %.2f' % (y.count(1) / len(y)))


vectorizer = TfidfVectorizer(stop_words='english')

# vectorize data, then divide data into training and testing data set
X = vectorizer.fit_transform(msgs).toarray()
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.33, random_state=44)

print("Results using Decision Tree Classifier built in Sklearn")
clf = DecisionTreeClassifier()
clf.fit(X_trn, y_trn)
y_pred = clf.predict(X_tst)

ac = accuracy_score(y_tst, y_pred)
print("Accuracy score: {:.4f}".format(ac))

cm = confusion_matrix(y_tst, y_pred)
print("Confusion Matrix:")
print(cm)

recall = recall_score(y_tst, y_pred)
print("Recall score: {:.4f}".format(recall))

precision = precision_score(y_tst, y_pred)
print("Precision: {:.4f}".format(precision))

f1_sc = f1_score(y_tst, y_pred)
print("F1 score: {:.4f}".format(f1_sc))


print("Results using Decison Tree Regressor built in Sklearn")
clf = DecisionTreeRegressor()
clf.fit(X_trn, y_trn)
y_pred = clf.predict(X_tst)
y_pred = [1 if p >= 0.5 else 0 for p in y_pred  ]
ac = accuracy_score(y_tst, y_pred)
print("Accuracy score: {:.4f}".format(ac))

cm = confusion_matrix(y_tst, y_pred)
print("Confusion Matrix:")
print(cm)

recall = recall_score(y_tst, y_pred)
print("Recall score: {:.4f}".format(recall))

precision = precision_score(y_tst, y_pred)
print("Precision score:{:.4f}".format(precision))

f1_sc = f1_score(y_tst, y_pred)
print("F1 score: {:.4f}".format(f1_sc))