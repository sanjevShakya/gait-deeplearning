from numpy import var
import tensorflow as tf
from gait.config import np, pd
from gait.path import get_log_file_path, get_model_file_path
from gait.utils import create_dir

keras = tf.keras
Dense = tf.keras.layers.Dense
Conv2D = tf.keras.layers.Conv2D
SeparableConv2D = tf.keras.layers.SeparableConv2D
Flatten = tf.keras.layers.Flatten
Sequential = tf.keras.Sequential
Dropout = tf.keras.layers.Dropout
MaxPooling2D = tf.keras.layers.MaxPooling2D
BatchNormalization = tf.keras.layers.BatchNormalization
TimeDistributed = tf.keras.layers.TimeDistributed
LSTM = tf.keras.layers.LSTM
AveragePooling2D = tf.keras.layers.AveragePooling2D
Activation = tf.keras.layers.Activation
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
MaxPooling1D = tf.keras.layers.MaxPooling1D
Conv1D = tf.keras.layers.Conv1D
GlobalAveragePooling1D = tf.keras.layers.GlobalAveragePooling1D
Input = tf.keras.Input
Concatenate = tf.keras.layers.Concatenate


def calculate_model_size(model):
    print(model.summary())
    var_sizes = [
        np.product(list(map(int, v.shape))) * v.dtype.size
        for v in model.trainable_variables
    ]
    print("Model size:", sum(var_sizes) / 1024, "KB")


def timeseries_shapes(train_X, train_y):
    n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]

    return (n_timesteps, n_features, n_outputs)


def build_cnn_lstm_stat(n_features, n_outputs, statistics):
    model = Sequential()
    TimeDistributed = tf.keras.layers.TimeDistributed
    input1 = Input(shape=(None, 32, n_features))
    input2 = Input(shape=(statistics.shape[1],statistics.shape[2]))

    net = TimeDistributed(
        Conv1D(8, n_features, strides=2, padding="same"))(input1)
    net = TimeDistributed(BatchNormalization())(net)
    net = TimeDistributed(Activation('relu'))(net)
    net = TimeDistributed(Dropout(0.5))(net)
    net = TimeDistributed(
        Conv1D(16, n_features*2, strides=2, padding="same"))(net)
    net = TimeDistributed(BatchNormalization())(net)
    net = TimeDistributed(Activation('relu'))(net)
    net = TimeDistributed(Dropout(0.5))(net)
    net = TimeDistributed(
        Conv1D(32, n_features * 3, strides=2, padding="same"))(net)
    net = TimeDistributed(BatchNormalization())(net)
    net = TimeDistributed(Activation('relu'))(net)
    net = TimeDistributed(Dropout(0.5))(net)
    net = TimeDistributed(GlobalAveragePooling1D())(net)
    net = TimeDistributed(Flatten())(net)

    net_combined = Concatenate()([net, input2])
    output = LSTM(100)(net_combined)
    output = Dropout(0.5)(output)
    output = Dense(100, activation="relu")(output)
    output = Dropout(0.1)(output)
    output = Dense(n_outputs, activation="softmax")(output)

    model = tf.keras.models.Model(
        inputs=[input1, input2], outputs=output)

    return model


def build_cnn_lstm(n_timesteps, n_features, n_outputs):
    model = Sequential()

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=n_features),
                              input_shape=(None, 32, n_features)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(TimeDistributed(
        Conv1D(filters=64, kernel_size=n_features)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(TimeDistributed(GlobalAveragePooling1D()))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))

    model.add(Dense(n_outputs, activation='softmax'))
    model.summary()
    return model


def build_cnn(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(Conv2D(8, n_features, padding="same",
                     activation="relu", input_shape=(n_timesteps, n_features, 1)))
    model.add(AveragePooling2D((3, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (4, 1), padding="same", activation="relu"))
    model.add(AveragePooling2D((3, 1)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(n_outputs, activation="softmax"))
    model.summary()
    return model


def build_cnn2(n_timesteps, n_features, n_outputs):
    model = Sequential()

    model.add(Conv2D(8, n_features, strides=2, padding="same",
              input_shape=(n_timesteps, n_features, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv2D(16, n_features * 2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(GlobalAveragePooling2D())
    model.add(Flatten())

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation="softmax"))

    model.summary()
    return model


def build_cnn_stats(n_timesteps, n_features, n_outputs, statistics):
    model = Sequential()
    inputs1 = Input(shape=(n_timesteps, n_features, 1))
    net = Conv2D(8, n_features, strides=2, padding="same")(inputs1)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Dropout(0.5)(net)
    net = Conv2D(16, n_features*2, strides=2, padding="same")(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Dropout(0.5)(net)
    net = Conv2D(32, n_features * 3, strides=2, padding="same")(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Dropout(0.5)(net)
    net = GlobalAveragePooling2D()(net)
    net = Flatten()(net)
    input2 = Input(shape=statistics.shape[1])
    net_combined = Concatenate()([net, input2])
    net_combined = Dense(32, activation="relu")(net_combined)
    net_combined = Dropout(0.5)(net_combined)
    net_combined = Dense(n_outputs, activation="softmax")(net_combined)

    model = tf.keras.models.Model(
        inputs=[inputs1, input2], outputs=net_combined)

    return model


def build_new_model(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(Conv2D(9, n_features, padding="same",
                     activation="relu", input_shape=(n_timesteps, n_features, 1)))
    model.add(Conv2D(32, (4, 1), padding="same", activation="relu"))
    model.add(AveragePooling2D(pool_size=3))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(n_outputs, activation="softmax"))
    model.summary()
    return model


def build_dsconv_cnn(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(Conv2D(8, n_features, padding="same",
                     activation="relu", input_shape=(n_timesteps, n_features, 1)))
    model.add(MaxPooling2D((3, 3)))
    model.add(BatchNormalization())

    model.add(SeparableConv2D(16, (4, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D((3, 1)))
    model.add(BatchNormalization())

    model.add(SeparableConv2D(32, (4, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D((3, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_outputs, activation="softmax"))
    model.summary()
    return model


def prepare_training_data_shape(train_X, test_X, variant):
    train_X_shape = train_X.shape
    test_X_shape = test_X.shape
    if variant != 'cnn_lstm':
        train_X = train_X.reshape(-1, train_X_shape[1],
                                  train_X_shape[2], 1).astype(np.float32)
        test_X = test_X.reshape(-1, test_X_shape[1],
                                test_X_shape[2], 1).astype(np.float32)
    if variant == 'cnn_lstm' or variant == 'cnn_lstm_stat':
        n_steps, n_length = 4, 32
        train_X = train_X.reshape((train_X.shape[0], n_steps, n_length, 12))
        test_X = test_X.reshape((test_X.shape[0], n_steps, n_length, 12))
    return train_X, test_X


def get_ml_model(variant, timesteps, features, outputs):
    if(variant == 'cnn'):
        return build_cnn(timesteps, features, outputs)
    elif(variant == 'cnn_lstm'):
        return build_cnn_lstm(timesteps, features, outputs)
    elif(variant == 'dsconvcnn'):
        return build_dsconv_cnn(timesteps, features, outputs)
    elif(variant == 'cnn_new'):
        return build_new_model(timesteps, features, outputs)
    elif(variant == 'build_cnn2'):
        return build_cnn2(timesteps, features, outputs)
    pass


def train_model(train_X, train_y, test_X, test_y, overlap_percent, verbose=1, epochs=10, batch_size=50, variant='cnn'):
    n_timesteps, n_features, n_outputs = timeseries_shapes(train_X, train_y)
    train_X, test_X = prepare_training_data_shape(train_X, test_X, variant)
    model = get_ml_model(variant, n_timesteps, n_features, n_outputs)
    if not model:
        raise Exception(
            'model', 'Model creation failed. Arguments not correct')
    model_filepath = get_model_file_path(
        overlap_percent,
        'best_model.{epoch:02d}-{val_loss:.2f}.hdf5')
    print('Model saved at filepath : {}'.format(model_filepath))
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_filepath,
            monitor='val_loss', save_best_only=True, mode="min"),
        # patience 20 gave a better result
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max', min_delta=0.8, patience=100)
    ]
    print(train_X.shape)
    optimizer = keras.optimizers.Adam(lr=1e-3)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    history = model.fit(train_X, train_y, epochs=epochs, verbose=verbose,
                        callbacks=callbacks_list, batch_size=batch_size,
                        validation_split=0.2
                        )
    evaluation_history = model.evaluate(
        test_X, test_y, batch_size=batch_size, verbose=verbose)

    return model, history, evaluation_history


def train_model_with_stats(train_X, train_y, test_X, test_y, trainXStats, testXStats, overlap_percent, verbose=1, epochs=10, batch_size=50):
    n_timesteps, n_features, n_outputs = timeseries_shapes(train_X, train_y)
    variant = 'cnn_stat'
    train_X, test_X = prepare_training_data_shape(train_X, test_X, variant)
    model = build_cnn_stats(n_timesteps, n_features, n_outputs, trainXStats)
    if not model:
        raise Exception(
            'model', 'Model creation failed. Arguments not correct')
    model_filepath = get_model_file_path(
        overlap_percent,
        'best_model.{epoch:02d}-{val_loss:.2f}.hdf5')
    print('Model saved at filepath : {}'.format(model_filepath))
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_filepath,
            monitor='val_loss', save_best_only=True, mode="min"),
        # patience 20 gave a better result
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max', min_delta=0.8, patience=100)
    ]
    print(train_X.shape)
    optimizer = keras.optimizers.Adam(lr=1e-3)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    history = model.fit([train_X, trainXStats], train_y, epochs=epochs, verbose=verbose,
                        callbacks=callbacks_list, batch_size=batch_size,
                        validation_split=0.2
                        )
    evaluation_history = model.evaluate(
        [test_X, testXStats], test_y, batch_size=batch_size, verbose=verbose)

    return model, history, evaluation_history


def train_model_cnn_lstm_with_stats(train_X, train_y, test_X, test_y, trainXStats, testXStats, overlap_percent, verbose=1, epochs=10, batch_size=50):
    n_timesteps, n_features, n_outputs = timeseries_shapes(train_X, train_y)
    variant = 'cnn_lstm_stat'
    train_X, test_X = prepare_training_data_shape(train_X, test_X, variant)
    model = build_cnn_lstm_stat(n_features, n_outputs, trainXStats)
    print('Train y shape:', train_y.shape)
    print('trainXStats  shape:', trainXStats.shape)
    if not model:
        raise Exception(
            'model', 'Model creation failed. Arguments not correct')
    model_filepath = get_model_file_path(
        overlap_percent,
        'best_model.{epoch:02d}-{val_loss:.2f}.hdf5')
    print('Model saved at filepath : {}'.format(model_filepath))
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_filepath,
            monitor='val_loss', save_best_only=True, mode="min"),
        # patience 20 gave a better result
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max', min_delta=0.8, patience=100)
    ]
    print(train_X.shape)
    optimizer = keras.optimizers.Adam(lr=1e-3)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    history = model.fit([train_X, trainXStats], train_y, epochs=epochs, verbose=verbose,
                        callbacks=callbacks_list, batch_size=batch_size,
                        validation_split=0.2
                        )

    return model, history
