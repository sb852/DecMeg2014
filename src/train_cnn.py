'''
We are loading all the participants data and we are training the cnn.
'''
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Bidirectional, LSTM, Input, GRU, Reshape
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping


def split_data_into_folds(train_x, train_y):
    """
    We are splitting the measurements of individual subjects into time-safe splits. The training set is approximately
    80% of the data, the validation set is 10% and the test set is 10%. We will let the network predict future recordings
    as this gives a much more realistic estimate of the true performance of a network.
    """
    cutoff_training_data = 500
    cutoff_validation_data = 550

    val_x = train_x[cutoff_training_data:cutoff_validation_data, :, :]
    val_y = train_y[cutoff_training_data:cutoff_validation_data]

    test_x = train_x[cutoff_validation_data::, :, :]
    test_y = train_y[cutoff_validation_data::]

    train_x = train_x[0:cutoff_training_data, :, :]
    train_y = train_y[0:cutoff_training_data]

    return train_x, train_y, val_x, val_y, test_x, test_y


def load_all_data():
    '''
    We are loading the preprocessed training data and we concatenate the data.
    :return train_x_all: The training data of all subjects.
    :return train_y_all: The labels of all subjects.
    '''
    number_subjects = 16
    for current_subject_nr in range(0, number_subjects):

        train_x = np.load('../data/subject' + str(current_subject_nr) + '_train_x_filtered.npy')
        train_y = np.load('../data/subject' + str(current_subject_nr) + '_train_y.npy')

        #  We split the data into training, validation and testing data.
        train_x, train_y, val_x, val_y, test_x, test_y = split_data_into_folds(train_x, train_y)

        #  We concatenate the data of all subjects.
        if 'train_x_all' in locals():
            train_x_all = np.concatenate((train_x, train_x_all), axis=0)
            train_y_all = np.concatenate((train_y, train_y_all), axis=0)

            val_x_all = np.concatenate((val_x, val_x_all), axis=0)
            val_y_all = np.concatenate((val_y, val_y_all), axis=0)

            test_x_all = np.concatenate((test_x, test_x_all), axis=0)
            test_y_all = np.concatenate((test_y, test_y_all), axis=0)

        else:
            train_x_all = train_x
            train_y_all = train_y

            val_x_all = val_x
            val_y_all = val_y

            test_x_all = test_x
            test_y_all = test_y

    train_x_all = train_x_all[:, :, :, np.newaxis]
    val_x_all = val_x_all[:, :, :, np.newaxis]
    test_x_all = test_x_all[:, :, :, np.newaxis]

    return train_x_all, train_y_all, val_x_all, val_y_all, test_x_all, test_y_all


def setup_model(data):
    """
    We are setting up the model architecture.
    :return:
    """
    num_classes = 2
    input_shape = [data.shape[1], data.shape[2], data.shape[3]]

    model = Sequential()

    #  The first convolutional block.
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    #  The second convolutional block.
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.2))

    #  The third convolutional block.
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.2))

    #  Adding an LSTM to extract time features.
    model.add(Reshape((15, 12*32)))
    model.add(GRU(50))

    #  The final dense layer.
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.1),
                  metrics=['accuracy'])

    return model


def train_model(train_x, train_y, val_x, val_y):
    '''
    We are training the model.
    :param train_x: The training data.
    :param train_y: THe training labels.
    :param val_x: The validation data.
    :param val_y: The valiation labels.
    :return model: The trained model.
    '''

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        shear_range=0.1,
        zoom_range=[0.90, 1.1])

    #  We are performing early stopping.
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=5, verbose=False, mode='auto')

    n_epochs = 100
    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=n_epochs, batch_size=100,
              callbacks=[early_stopping], verbose=1)

    model.fit(train_x, train_y)

    return model


def perform_one_hot_encoding(labels):
    """
    We are performing one-hot encoding on the labels.
    """
    labels = np.reshape(labels, [-1, 1])
    onehot_encoder = OneHotEncoder(sparse=False)
    labels = onehot_encoder.fit_transform(labels)

    return labels


if __name__ == '__main__':
    #  We are loading the data and we concatenate the data of all subjects.
    train_x, train_y, val_x, val_y, test_x, test_y = load_all_data()

    #  We are setting up the neural net.
    model = setup_model(train_x)

    #  We are performing one-hot encoding.
    train_y = perform_one_hot_encoding(train_y)
    val_y = perform_one_hot_encoding(val_y)
    test_y = perform_one_hot_encoding(test_y)

    #  We are training the model.
    model = train_model(train_x, train_y, val_x, val_y)

