from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D


class ModelLoader():
    def __init__(self, n_labels, seq_length, model_name,
                 saved_weights=None, optimizer=None, image_size=(100, 176)):

        self.n_labels = n_labels
        self.load_model = load_model
        self.saved_weights = saved_weights
        self.model_name = model_name

        # Loads the specified model
        print('loading C3D model')
        self.input_shape = ((seq_length,) + image_size + (3,))
        self.model = self.c3d()


        metrics = ['accuracy', 'top_k_categorical_accuracy']

        # If no optimizer is given, use Adam as default
        if not optimizer:
            optimizer = Adam()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=metrics)
        print(self.model.summary())

    def c3d(self):
        """See: 'https://arxiv.org/pdf/1412.0767.pdf' """
        # Tunable parameters
        strides = (1, 1, 1)
        model = Sequential()
        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=strides, input_shape=(
            self.input_shape), border_mode='same', activation='relu'))
        # model.add(Activation('relu'))

        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=strides, padding='same', activation='softmax'))
        # model.add(Activation('softmax'))

        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=strides, padding='same', activation='relu'))
        # model.add(Activation('relu'))
        model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=strides, padding='same', activation='softmax'))
        # model.add(Activation('softmax'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_labels, activation='softmax'))

        return model

