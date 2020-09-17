import os
import time
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub


def ratings_mapper(rating):
    if rating >= 4:
        return 1
    elif rating == 3:
        return 0
    else:
        return -1


def load_dataset(file_path, num_rows):
    df = pd.read_csv(file_path, usecols=[6, 9], nrows=num_rows)
    df.columns = ['rating', 'title']

    # separate features from labels
    X = df['title'].astype(str).str.encode('ascii', 'replace').to_numpy(dtype=object)
    y = df['rating'].apply(ratings_mapper)

    labels = np.array(pd.get_dummies(y), dtype=int)
    return labels, X


## https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1
## https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1

def get_model():
    hub_layer = hub.KerasLayer('https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1',
                               output_shape=[128], input_shape=[], dtype=tf.string, name='input', trainable=False)

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax', name='output'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()
    return model


def train(epochs=5, batch_size=32, train_file='train.csv', val_file='test.csv'):
    workdir = os.getcwd()
    print("Loading training/validation data...")

    y_train, X_train = load_dataset(train_file, num_rows=1e5)
    y_val, X_val = load_dataset(val_file, num_rows=1e4)

    print("Training the model...")
    model = get_model()
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(X_val, y_val),
              callbacks=[tf.keras.callbacks.ModelCheckpoint(os.path.join(workdir, 'model_checkpoint'),
                                                            monitor='val_loss', verbose=1, save_best_model=True,
                                                            save_weights_only=False, mode='auto')])
    return model


def export_model(model, base_path='amazon_review'):
    workdir = os.getcwd()
    time_ = str(int(time.time()))
    path = os.path.join(workdir, base_path, time_)
    tf.saved_model.save(model, path)


if __name__ == '__main__':
    model = train()
    export_model(model)
