from SpamDetection.preprocessing import get_data
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB





y, msgs = get_data('spam.csv')
print('spam percentage: %.2f' % (y.count(1) / len(y)))


vectorizer = TfidfVectorizer(stop_words='english')

# vectorize data, then divide data into training and testing data set
X = vectorizer.fit_transform(msgs).toarray()
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.33, random_state=44)


classifier = BernoulliNB()
classifier.fit(X_trn,y_trn)
y_pred = classifier.predict(X_tst)


sc = accuracy_score(y_tst, y_pred)
print("Accuracy score %.4f" % sc)


precision = precision_score(y_tst, y_pred)
print('Precision score: %.4f ' % precision)

recall_score = recall_score(y_tst, y_pred, average='macro')
print('Recall score: %.4f' % recall_score)


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_tst, y_pred)
print(confusion_matrix)

from sklearn.metrics import f1_score
f1 = f1_score(y_tst, y_pred)
print('f1_score: {:.2f}'.format(f1))

input_msg = [r'hey, boy, are you there? something for you']
input_X = vectorizer.transform(input_msg).toarray()
print(classifier.predict(input_X))
