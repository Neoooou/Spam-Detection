from SpamDetection.preprocessing import get_data
from hyperopt import Trials, STATUS_OK, tpe, rand
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from hyperas.distributions import choice, uniform, conditional
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras import callbacks
from hyperas import optim
import numpy as np


def data():
    y, msgs = get_data("spam.csv")
    v = TfidfVectorizer(stop_words="english")
    X = v.fit_transform(msgs)
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=43)
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=0.2, random_state=43)
    return X_trn, X_val, X_tst, y_trn, y_val, y_tst


def create_model(X_trn, y_trn, X_val, y_val):
    model = Sequential()
    model.add(
        Dense({{choice([np.power(2,5), np.power(2,6), np.power(2,7)])}}, input_dim=X_trn.shape[1])
    )
    model.add(LeakyReLU(alpha={{uniform(0, 0.5)}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(
        Dense({{choice([np.power(2,5), np.power(2,6), np.power(2,7)])}}, input_dim=X_trn.shape[1])
    )
    model.add(
        LeakyReLU(alpha={{uniform(0, 0.5)}})
    )
    model.add(
        Dropout({{uniform(0.5, 1)}})
    )
    model.add(Dense(1, activation='sigmoid'))
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2,patience=5, min_lr=0.0001)
    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_trn, y_trn, epochs={{choice([25, 50, 75, 100])}}, batch_size={{choice([16, 32, 64])}},
              validation_data=(X_val, y_val), verbose=1, callbacks=[reduce_lr])
    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Test accuracy:', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=15,
                                          trials=Trials())
    X_train, X_val, X_test, y_train, y_val, y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    best_model.save('spam_detection_model.h5')