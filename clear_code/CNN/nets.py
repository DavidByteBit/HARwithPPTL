from tensorflow.keras import layers
from tensorflow.keras import activations

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


class nets:

    def __init__(self):
        self.models = {"create_model1": self.create_model1,
                       "daily_activity": self.daily_activity,
                       "standard_model": self.standard_model,
                       }

    def create_model1(self, n_timesteps, n_features, n_outputs):

        print("input shape")
        print(n_timesteps)
        print(n_features)

        model = Sequential()
        model.add(Conv1D(filters=16, kernel_size=2, activation='hard_sigmoid', input_shape=(n_timesteps, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=128, kernel_size=8, activation='hard_sigmoid'))
        model.add(MaxPooling1D(pool_size=8))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def daily_activity(self, n_timesteps, n_features, n_outputs):

        model = Sequential()
        model.add(Conv1D(filters=8, kernel_size=2, activation='hard_sigmoid', input_shape=(n_timesteps, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=128, kernel_size=5, activation='hard_sigmoid'))
        model.add(Dropout(0.25))
        model.add(MaxPooling1D(pool_size=8))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())

        return model


    def standard_model(self, n_timesteps, n_features, n_outputs):
        model = Sequential()
        model.add(Conv1D(filters=16, kernel_size=2, activation='hard_sigmoid', input_shape=(n_timesteps, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=128, kernel_size=8, activation='hard_sigmoid'))
        model.add(MaxPooling1D(pool_size=8))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
