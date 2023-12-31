import os

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, Model, Sequential
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd


def lr_update(epoch):
    if epoch < 35:
        return 0.05
    elif epoch < 70:
        return 0.04
    elif epoch < 220:
        return 0.01
    elif epoch < 300:
        return 0.009
    elif epoch < 400:
        return 0.005
    elif epoch < 500:
        return 0.004
    else:
        return 0.001


# Update the plot_loss_curves function
def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    lr = history.history['lr']

    epochs = range(len(history.history['loss']))

    # Create a figure
    plt.figure(figsize=(20, 10))

    # Subplot 1: For loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.grid()

    # Create a second y-axis for the learning rate
    plt.twinx()
    plt.plot(epochs, lr, label='learning_rate', linestyle=':')
    plt.ylabel('Learning Rate')
    plt.legend(loc='upper right')

    # Subplot 2: For accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.grid()

    # Create a second y-axis for the learning rate
    plt.twinx()
    plt.plot(epochs, lr, label='learning_rate', linestyle=':')
    plt.ylabel('Learning Rate')
    plt.legend(loc='upper right')

    # Show the figure
    plt.show()


# Define a function to create a dataset for LSTM
def creating_datasets(train_windows, val_windows, train_labels, val_labels, batch_size=32):
    train_windows_dataset = tf.data.Dataset.from_tensor_slices(train_windows)
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)
    val_windows_dataset = tf.data.Dataset.from_tensor_slices(val_windows)
    val_labels_dataset = tf.data.Dataset.from_tensor_slices(val_labels)

    train_dataset = tf.data.Dataset.zip((train_windows_dataset, train_labels_dataset))
    val_dataset = tf.data.Dataset.zip((val_windows_dataset, val_labels_dataset))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(train_dataset)
    print(val_dataset)
    return train_dataset, val_dataset


def make_train_test_datasets(X, y, time_steps=1024, test_size=0.2):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y[i + time_steps])

    split_index = int(len(Xs) * (1 - test_size))

    X_train = np.array(Xs[:split_index])
    y_train = np.array(ys[:split_index])

    X_test = np.array(Xs[split_index:])
    y_test = np.array(ys[split_index:])

    return X_train, X_test, y_train, y_test


def df_to_datasets(batch_size, df, test_size, time_steps):
    # Initialize the oversampler
    ros = RandomOverSampler(random_state=0)
    # Separate features and target
    X = df.drop(['date', 'action'], axis=1)
    y = df['action']
    # Apply oversampling
    X_resampled, y_resampled = ros.fit_resample(X, y)
    # Normalize the features
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_resampled), columns=X_resampled.columns)
    # Initialize a LabelEncoder
    le = LabelEncoder()
    # Fit the encoder and transform the 'action' column
    y_resampled = le.fit_transform(y_resampled)
    # Convert target to one-hot encoding
    y_resampled = to_categorical(y_resampled)
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = make_train_test_datasets(X_scaled, y_resampled, time_steps=time_steps,
                                                                test_size=test_size)
    train_dataset, val_dataset = creating_datasets(X_train, X_test, y_train, y_test, batch_size=batch_size)
    return X_train, train_dataset, val_dataset


def create_model_callbacks(model_name, check_point=False, early_stopping=False, reduce_lr=False, custom_lr=True,
                           save_path="model_experiments", monitor="val_accuracy", checkpoint_verbose=1,
                           early_stop_patience=5, learning_rate_patience=500,
                           learning_rate_factor=0.1, learning_rate_verbose=1):
    callbacks = []

    if check_point:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                                            verbose=checkpoint_verbose,
                                                            save_best_only=True,
                                                            save_weights_only=False,
                                                            monitor=monitor))
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=early_stop_patience,
                                                          restore_best_weights=True))
    if reduce_lr:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, patience=learning_rate_patience,
                                                              factor=learning_rate_factor,
                                                              verbose=learning_rate_verbose))

    if custom_lr:
        callbacks.append(LearningRateScheduler(lr_update))

    return callbacks


def dense_model(X_train, size):
    # Define the Dense model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Dense(size, activation='relu'),
        Dense(size, activation='relu'),
        Flatten(),
        Dense(3, activation='softmax')  # Since we have three classes
    ], name="trading")
    return model


def trading_model(df, use_model="Dense", test_size=0.2, epochs=250, batch_size=128, time_steps=128, learning_rate=0.01,
                  checkpoint_verbose=True):
    X_train, train_dataset, val_dataset = df_to_datasets(batch_size, df, test_size, time_steps)

    size = 1024

    if use_model == "ensemble":
        n_models = 5  # Number of models in the ensemble

        models, histories = [
            ensemble_model(X_train, size, epochs, batch_size, learning_rate, train_dataset, val_dataset)
            for _ in range(n_models)]

    else:
        model = dense_model(X_train, size)

        # Compile the model with a specified learning rate
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        # Create the LearningRateScheduler callback
        lr_callback = LearningRateScheduler(lr_update)

        # Add the callback to the fit method
        history = model.fit(x=train_dataset, epochs=epochs, batch_size=batch_size, validation_data=val_dataset,
                            callbacks=[lr_callback])

        model.save("model_experiments/trading.h5")
        model = tf.keras.models.load_model(os.path.join("model_experiments", f"{model.name}.h5"))

        print(f"\nvalidation accuracy: {model.evaluate(val_dataset)[1] * 100:.2f}%")

        # Plot the learning rate versus loss
        plot_loss_curves(history)


def ensemble_model(X_train, size, epochs, batch_size, learning_rate, train_dataset, val_dataset):
    model = dense_model(X_train, size)

    model.compile(optimizer=SGD(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    lr_callback = LearningRateScheduler(lr_update)

    history = model.fit(x=train_dataset, epochs=epochs, batch_size=batch_size, validation_data=val_dataset,
                        callbacks=[lr_callback])

    return model, history
