import os
import csv

import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence


def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = chunk[1:]
    return input_seq, target_seq


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def build_model(labels_len, rnn_units):  # Internet says I can feed the data directly into the LSTM, long as it's 3D
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(labels_len, activation='relu')  # I have the feeling I should be using relu, not sure.
    ])
    return model


if __name__ == "__main__":
    print("Running nn1...")
    print(tf.__version__)
    # input goes here
    training_data = []
    file_path = ''
    # mapping = {'R': [1, 0, 0], 'P': [0, 1, 0], 'S': [0, 0, 1]}
    # Let's try again using Sparse categorical cross-entropy.
    mapping = {'R': [int(0)], 'P': [int(1)], 'S': [int(2)]}
    # Open the CSV file

    with open(file_path, 'r') as file:
        # Create a CSV reader
        reader = csv.reader(file)
    raw_data = np.genfromtxt(file_path, delimiter=',', dtype=str)

    encoded_opponent_plays = np.array([mapping[item] for item in raw_data])
    # Creating Training Examples
    seq_length = 10

    # Create training examples/targets with tf's help

    examples_per_epoch = len(raw_data) // (seq_length + 1)
    # This makes tensorlike data (e.g. numpy arrays) into a tf Dataset
    dataset = tf.data.Dataset.from_tensor_slices(encoded_opponent_plays)
    # This batches the data in sequences of the desired length


    sequences = dataset.batch(seq_length + 1, drop_remainder=True)
    # Next we use these sequences of length 10+1 and split them into input & output (prev seq vs reply)


    dataset = sequences.map(split_input_target)

    # Next, we make some training batches.
    # I think my problems stem from the fact I'm not using an embedding layer... regarding the training checkpoints
    # And the need to build the model before running it... ugh what a HEADACHE!

    RNN_UNITS = 512  # I'm guessing we need less units for this task.
    BATCH_SIZE = 64
    LABELS_LEN = len(mapping)
    BUFFER_SIZE = 10000
    # Now we'll shuffle and batch our dataset to feed it into the model

    data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    # And now it's time to build the model. Wonder what's the way to do this without an embedding layer?
    # We're making a function to create the model, since we're going to pass it inputs of different size.

    model = build_model(LABELS_LEN, RNN_UNITS)
    model.build(input_shape=(64, 10, 1))  # Not sure why I have to build the model after declaring it...
    model.summary()

    # Let's test the input and output shapes match and the model is OK
    for input_example_batch, target_example_batch in data.take(1):
        example_batch_predictions = model(input_example_batch)
        print("Example batch shape:", example_batch_predictions.shape)
        # Let's examine the first prediction in the batch:
        pred = example_batch_predictions[0]
        print("First prediction from the batch: ", pred)  # This is a 2D array of length 10, & each interior array is the
        # prediction for the opponent's next play
        # And finally we look at a prediction at the first timestep
        time_pred = pred[0]
        print("Prediction at time step zero: ", time_pred)

    # Now let's build a loss function to our model. This is where things get iffy...
    # Since we're using one-hot, we're going with Cat. Crossentropy (not sparse)
    # And uuuuh... I'm not sure that I'm doing this correctly but let's go on and see what happens.

    def loss(y_true, y_pred):  # I don't think I should be using logits since I'm not using a sparse variable idk.
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # Compile the model
    model.compile(optimizer='adam', loss=loss)

    # Configure the checkpoints
    checkpoint_dir = './training_checkpoints2'
    # Filename
    checkpoint_preffix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_preffix,
        save_weights_only=True)

    # And here you start training the model
    # I don't need to modify the batch size since I'm always feeding a sequence of size 10 to the model (at least for now)
    history = model.fit(data, epochs=40, callbacks=[checkpoint_callback])  # Wondering if I can overfit the model...
    # Probably not, it's fighting a bot after all.
    # At 60 epochs, we hit loss: 0.4982
    # At 300 epochs, we hit loss: 0.4029, seems to settle around this value but could go lower of course.

    # Good, now we rebuild with a batch size of 1 and call a checkpoint
    model = build_model(LABELS_LEN, RNN_UNITS)
    # model.build(input_shape=(1, 10, 1))
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))


    # Now let's test it with a random string and see what's the output.
    test_string = [0, 1, 1, 2, 1, 0, 0, 1, 2, 1]
    test_string = tf.expand_dims(test_string, 0)
    this_prediction = model(test_string)
    this_prediction = tf.squeeze(this_prediction, 0)
    result = this_prediction[9]
    max_to_one = lambda arr: (arr == np.max(arr)).astype(int)
    result = max_to_one(result.numpy())
    print("Output from the model: ", result)
