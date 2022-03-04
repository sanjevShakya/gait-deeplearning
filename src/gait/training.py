import tensorflow as tf
from gait.config import np, pd
from gait.path import get_log_file_path, get_model_file_path
from gait.utils import create_dir

keras = tf.keras
Dense = tf.keras.layers.Dense
Conv2D = tf.keras.layers.Conv2D
Flatten = tf.keras.layers.Flatten
Sequential = tf.keras.Sequential
Dropout = tf.keras.layers.Dropout
MaxPooling2D = tf.keras.layers.MaxPooling2D
BatchNormalization = tf.keras.layers.BatchNormalization
TimeDistributed = tf.keras.layers.TimeDistributed
LSTM = tf.keras.layers.LSTM


def reshape_tensor(data):
    return tf.reshape(data, [-1, 3, 1])


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


def build_cnn_lstm(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(8, n_features, padding="same",
                                     activation="relu"), input_shape=(n_timesteps, n_features, 1)))
    model.add(TimeDistributed(MaxPooling2D((3, 3))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(
        Conv2D(32, (4, 1), padding="same", activation="relu")))
    model.add(TimeDistributed(MaxPooling2D((3, 1))))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(n_outputs, activation="softmax"))
    # model.summary()
    return model


def build_cnn(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(Conv2D(8, n_features, padding="same",
                     activation="relu", input_shape=(n_timesteps, n_features, 1)))
    model.add(MaxPooling2D((3, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (4, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D((3, 1)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(n_outputs, activation="softmax"))
    model.summary()
    return model


def prepare_training_data_shape(train_X, test_X):
    train_X_shape = train_X.shape
    test_X_shape = test_X.shape
    train_X = train_X.reshape(-1, train_X_shape[1],
                              train_X_shape[2], 1).astype(np.float32)
    test_X = test_X.reshape(-1, test_X_shape[1],
                            test_X_shape[2], 1).astype(np.float32)
    return train_X, test_X


def get_ml_model(variant, timesteps, features, outputs):
    if(variant == 'cnn'):
        return build_cnn(timesteps, features, outputs)
    elif(variant == 'cnn_lstm'):
        return build_cnn_lstm(timesteps, features, outputs)
    pass


def train_model(train_X, train_y, test_X, test_y, overlap_percent, verbose=1, epochs=10, batch_size=128, variant='cnn'):
    n_timesteps, n_features, n_outputs = timeseries_shapes(train_X, train_y)
    train_X, test_X = prepare_training_data_shape(train_X, test_X)
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
        keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
    ]
    print(train_X.shape)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    history = model.fit(train_X, train_y, epochs=epochs, verbose=verbose,
                        callbacks=callbacks_list, batch_size=batch_size,
                        validation_split=0.2
                        )
    evaluation_history = model.evaluate(
        test_X, test_y, batch_size=batch_size, verbose=verbose)

    return model, history, evaluation_history
