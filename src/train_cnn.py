'''
We are loading all the participants data and we are training the cnn.
'''
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Bidirectional, LSTM



def load_all_data():
    '''
    We are loading
    :return:
    '''
    number_subjects = 16
    for current_subject_nr in range(0, number_subjects):

        train_x = np.load('../data/subject' + str(current_subject_nr) + '_train_x_filtered.npy')
        train_y = np.load('../data/subject' + str(current_subject_nr) + '_train_y.npy')

        if 'train_x_all' in locals():
            train_x_all = np.concatenate((train_x, train_x_all), axis=0)
            train_y_all = np.concatenate((train_y, train_y_all), axis=0)
        else:
            train_x_all = train_x
            train_y_all = train_y

    train_x = train_x_all
    train_y = train_y_all


def setup_model():
    """
    We are setting up the model architecture.
    :return:
    """
    num_classes = 2
    input_shape = [train_x.shape[1], train_x.shape[2], train_x.shape[3]]

    model = Sequential()

    #  The first convolutional block.
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    #  The second convolutional block.
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())

    #  Adding an LSTM to extract time features.
    model.add(LSTM(20, return_sequences=True))

    #  The final dense layer.
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.1),
                  metrics=['accuracy'])

if __name__ == '__main__':
    #  We are loading the data and we concatenate the data of all subjects.
    data = load_all_data()

    #  We perform crossvalidation.


    #  We are setting up the neural net.