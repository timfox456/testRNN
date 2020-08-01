from keras.layers import Dense, LSTM, Embedding, Dropout
from keras import backend
from keras.models import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import getActivationValue,layerName, hard_sigmoid
from keract import get_activations_single_layer
import os.path


def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

class tsClass:
    def __init__(self):
        self.data = None
        self.X_orig = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.unique_chars = None
        self.char_to_int = None
        self.int_to_char = None
        self.pad = 0
        self.numAdv = 0
        self.numSamples = 0
        self.perturbations = []

    
    def load_data(self, filename, seq_len, normalise_window):

        f = open(filename, 'r').read()
        data = f.split('\n')
    
        sequence_length = seq_len + 1
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
    
        if normalise_window:
            result = self.normalise_windows(result)
    
        result = np.array(result)
    
        row = round(0.9 * result.shape[0])
        train = result[:int(row), :]
        np.random.shuffle(train)
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1]
    
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        self.X_train, self.X_test, self.y_train, self.y_test = (x_train, y_train, x_test, y_test)

        return [x_train, y_train, x_test, y_test]


    def normalise_windows(self, window_data):
        normalised_data = []
        for window in window_data:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
            normalised_data.append(normalised_window)
        return normalised_data


    def load_model(self):
        filename = 'models/Ts.h5'

        if os.path.isfile(filename):
            self.model = load_model(filename,custom_objects={'rmse': rmse})
            self.model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=[rmse])
            self.model.summary()

    def layerName(self, layer):
        layerNames = [layer.name for layer in self.model.layers]
        return layerNames[layer]

    def train_model(self):
        self.load_data()
        char_num = len(self.unique_chars) + 1
        embedding_vector_length = 8
        self.model = Sequential()
        self.model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True)) 
        self.model.add(Dropout(0.2))
        self.model.add(LSTM( layers[2], return_sequences=False)) 
        self.model.add(Dropout(0.2)) 
        self.model.add(Dense( output_dim=layers[3])) 
        self.model.add(Activation("linear"))
        self.model.compile(loss="mse", optimizer="rmsprop", metrics=[rmse])
        
        print(self.model.summary())
        self.model.fit(self.X_train,self.y_train, validation_data=(self.X_test,self.y_test), nb_epoch=300, batch_size=32)
        self.model.save('Ts.h5')

    def displayInfo(self,test):
        test = test[np.newaxis, :]
        smiles = self.vect_smile(test)
        output_value = np.squeeze(self.model.predict(test))
        print("current SMILES: ",smiles[0])
        print("current prediction: ", output_value)
        return output_value

    def updateSample(self,pred1,pred2,m,o):
        if abs(pred1-pred2) >= 1 and o == True:
            self.numAdv += 1
            self.perturbations.append(m)
        self.numSamples += 1
        self.displaySuccessRate()

    def displaySamples(self):
        print("%s samples are considered" % (self.numSamples))

    def displaySuccessRate(self):
        print("%s samples, within which there are %s adversarial examples" % (self.numSamples, self.numAdv))
        print("the rate of adversarial examples is %.2f\n" % (self.numAdv / self.numSamples))

    def displayPerturbations(self):
        if self.numAdv > 0:
            print("the average perturbation of the adversarial examples is %s" % (sum(self.perturbations) / self.numAdv))
            print("the smallest perturbation of the adversarial examples is %s" % (min(self.perturbations)))


    # calculate the lstm hidden state and cell state manually
    def cal_hidden_state(self, test):
        acx = get_activations_single_layer(self.model, np.array([test]), self.layerName(0))
        units = int(int(self.model.layers[1].trainable_weights[0].shape[1]) / 4)
        # print("No units: ", units)
        # lstm_layer = model.layers[1]
        W = self.model.layers[1].get_weights()[0]
        U = self.model.layers[1].get_weights()[1]
        b = self.model.layers[1].get_weights()[2]

        W_i = W[:, :units]
        W_f = W[:, units: units * 2]
        W_c = W[:, units * 2: units * 3]
        W_o = W[:, units * 3:]

        U_i = U[:, :units]
        U_f = U[:, units: units * 2]
        U_c = U[:, units * 2: units * 3]
        U_o = U[:, units * 3:]

        b_i = b[:units]
        b_f = b[units: units * 2]
        b_c = b[units * 2: units * 3]
        b_o = b[units * 3:]

        # calculate the hidden state value
        h_t = np.zeros((self.pad, units))
        c_t = np.zeros((self.pad, units))
        f_t = np.zeros((self.pad, units))
        h_t0 = np.zeros((1, units))
        c_t0 = np.zeros((1, units))

        for i in range(0, self.pad):
            f_gate = hard_sigmoid(np.dot(acx[i, :], W_f) + np.dot(h_t0, U_f) + b_f)
            i_gate = hard_sigmoid(np.dot(acx[i, :], W_i) + np.dot(h_t0, U_i) + b_i)
            o_gate = hard_sigmoid(np.dot(acx[i, :], W_o) + np.dot(h_t0, U_o) + b_o)
            new_C = np.tanh(np.dot(acx[i, :], W_c) + np.dot(h_t0, U_c) + b_c)
            c_t0 = f_gate * c_t0 + i_gate * new_C
            h_t0 = o_gate * np.tanh(c_t0)
            c_t[i, :] = c_t0
            h_t[i, :] = h_t0
            f_t[i, :] = f_gate

        return h_t, c_t, f_t











