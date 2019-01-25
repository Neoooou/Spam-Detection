from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from SpamDetection.preprocessing import get_data, clean_msg
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import recall_score, precision_score,confusion_matrix,f1_score,accuracy_score
import tensorflow as tf

import keras.backend as K
class NNClassifier:
    def __init__(self):
        self.load_data()
        K.clear_session()
        self.build_model()


    def encode_input(self,msg):
        msg = clean_msg(msg)
        raw_msg = [msg]
        in_x = self.vectorizer.transform(raw_msg)
        return in_x

    def make_predict(self, msg):
        with self.sess.as_default():
            with self.graph.as_default():
                pred = self.model.predict(self.encode_input(msg))
        print(pred)
        return pred[0][0] >= 0.5

    def load_data(self):
        y, msgs = get_data('spam.csv')
        print('spam percentage: %.2f' % (y.count(1) / len(y)))

        vectorizer = TfidfVectorizer(stop_words='english')

        # vectorize data, then divide data into training and testing data set
        X = vectorizer.fit_transform(msgs).toarray()
        self.X_trn, self.X_tst, self.y_trn, self.y_tst = train_test_split(X, y, test_size=0.33, random_state=44)
        self.vectorizer = vectorizer


    def build_model(self):

        model = Sequential()
        model.add(Dense(np.power(2, 6), input_dim = len(self.X_trn[0])))
        model.add(LeakyReLU(.11162758792630445))
        model.add(Dropout(.64640024226702))
        model.add(Dense(np.power(2, 7)))
        model.add(LeakyReLU(.2613309282324268))
        model.add(Dropout(.6920257899742139))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        # model.fit(X_trn, y_trn,
        #           epochs=100,
        #           batch_size=16)
        file_path = 'spam_detection_model.h5'
        model.load_weights(file_path)
        self.model = model
        # 解决 Error: tensor is not an element of this graph
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        self.sess = K.get_session()


    def evaluate_model(self,X_tst, y_tst):


        score = self.model.evaluate(X_tst, y_tst)
        print('Evaluate loss and score: {:.4f}, {:.4f} '.format(score[0], score[1]))

        y_pred = self.model.predict(X_tst)
        y_pred = [1 if y[0] >= 0.5 else 0 for y in y_pred]
        print(y_pred[:100])

        sc = accuracy_score(y_tst, y_pred)
        print("Accuracy score %.4f" % sc)

        cm = confusion_matrix(y_tst, y_pred)
        print("Confusion Matrix:")
        print(cm)

        precision = precision_score(y_tst, y_pred)
        print('Precision score: %.4f ' % precision)

        recall = recall_score(y_tst, y_pred, average='macro')
        print('Recall score: %.4f' % recall)

        f1_sc = f1_score(y_tst, y_pred)
        print("F1 score: {:.4f}".format(f1_sc))

if __name__ == '__main__':
    clf = NNClassifier()
    msg = r'XXXMobileMovieClub click the WAP http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL'
    print(clf.make_predict(msg))
