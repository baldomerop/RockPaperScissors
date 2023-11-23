import tensorflow as tf
from tensorflow import keras
from dataManipulation import open_csv
import numpy as np
import csv

NUM_GAMES = 120
SEQ_LENGTH = 10  # I can feel this is going to be tricky. We are back to len 10.
TRAINING = True
TRAINING_ITER = 0  # If at 0, check the file_path manually
DENSE_NEURONS = 24
RNN_UNITS = 512
EPOCH_NUM = 10
# if TRAINING_ITER == 0:
#     UPDATING = False
# else:
#     UPDATING = True
UPDATING = False

# load_chk_path = './checkpoints/quincy_chkp_test'
if TRAINING_ITER == 0:
    file_path = f"./databases/abbey/db1_random_abbey_n2000.csv"
    # file_path = './databases/quincy/db7_random_quincy_n2000.csv'  # Quincy testing
else:
    file_path = f'db1_nn_abbey_n{NUM_GAMES}_0{TRAINING_ITER-1}.csv'  # Raw data
load_chk_path = f'./checkpoints/abbey/trained{TRAINING_ITER-1}'
last_saved_chk_path = f'./checkpoints/abbey/trained{TRAINING_ITER}'

# load_chk_path = f'./checkpoints/quincy/trained{TRAINING_ITER-1}'
# last_saved_chk_path = f'./checkpoints/quincy/trained{TRAINING_ITER}'
#

# OHE the data
move_to_ohe = {  # Create a dictionary to OHE the data
    "R": [1, 0, 0],
    "P": [0, 1, 0],
    "S": [0, 0, 1],
}


def build_model(seq_l, vocab_size, rnn_units, dense_neurons):
    this_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_l, vocab_size), dtype='int32'),
        tf.keras.layers.Dense(dense_neurons, activation='relu'),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return this_model


if __name__ == "__main__":
    # Import the data
    raw_data = open_csv(file_path)
    # with open(file_path, 'r') as file:
    #     # Create a CSV reader
    #     reader = csv.reader(file)
    # raw_data = np.genfromtxt(file_path, delimiter=',', dtype=str, skip_header=1)
    # raw_data = np.array(raw_data)

    data = np.array([[move_to_ohe[val] for val in row] for row in raw_data])  # Should I leave this alone?
    # Now we're getting y2, the 2nd objective.
    # Unlike Todorov, we are OHE'ing all of our data before passing it to the model.

    # First, a small function to calculate the winner of a given round:


    def calculate_winner(x):  # This variable will also be OHE
        p1, p2 = x[0], x[1]
        p1 = p1[1] + p1[2] * 2  # This converts from array to the correct ints using only arithmetic. ZOMG.
        p2 = p2[1] + p2[2] * 2
        result = p1 - p2  # 0 is a tie, 1 is a win, and -1 is a loss for p1: the opponent. Remember p1 is the opponent.
        # We are simply following Todorov here. But REMEMBER this is inverted wrt the verbose from Replit.
        if result == 0:
            return [0, 1, 0]  # tie
        if result in (1, -2):
            return [0, 0, 1]  # win
        return [1, 0, 0]  # lose


    def create_y_winner(this_data):  # This will only create the dataset for the secondary objective (y2)
        round_results = []
        for j, game in enumerate(this_data):  # Vestigial. Leave it just in case.
            score = calculate_winner(game)
            round_results.append(score)
            # n.b., we're using p1 as the one to predict, i.e. the opponent.

        return np.array(round_results)


    y_dataset_2 = create_y_winner(data)
    # Next, we need to sequence the data, since the model won't operate on single-move games, nor on games like Todorov's
    # examples_per_epoch = len(data)//(SEQ_LENGTH+1)  # This thing doesn't apply to the way we're handling data.
    total_seqs = len(data) - (SEQ_LENGTH + 1)

    # Creating the training sequences
    Xy1_sequences = np.empty((total_seqs, SEQ_LENGTH + 1, 2, 3), dtype=int)
    y2_sequences = np.empty((total_seqs, SEQ_LENGTH + 1, 3), dtype=int)
    for i, d in enumerate(data):
        if i == total_seqs:
            break
        Xy1_sequences[i] = data[i: i + (SEQ_LENGTH + 1)]
        # Xy1_sequences.append(data[i: i + (SEQ_LENGTH + 1)])
    for i, d in enumerate(y_dataset_2):
        if i == total_seqs:
            break
        y2_sequences[i] = y_dataset_2[i: i + (SEQ_LENGTH + 1)]
    x1_seq = Xy1_sequences[:, :-1, :, :]
    y1_seq = Xy1_sequences[:, 1:, 0, :]  # y1 has to be only about the opponent
    y2_seq = y2_sequences[:, :-1, :]

    # x1_seq, y1_seq, y2_seq = np.array(x1_seq), np.array(y1_seq), np.array(y2_seq)
    x1_seq, y1_seq = np.reshape(x1_seq, (total_seqs, SEQ_LENGTH, 6)), \
        np.reshape(y1_seq, (total_seqs, SEQ_LENGTH, 3))

    # batch_size = 1

    # Create a dataset from your data
    # dataset = tf.data.Dataset.from_tensor_slices((x1_seq, y1_seq, y2_seq))

    # Batch the dataset
    # dataset = dataset.batch(batch_size)

    # I'll try building the model following Russica, since Todorov's method is not working.

    model = build_model(SEQ_LENGTH, 6, RNN_UNITS, DENSE_NEURONS)
    print("Database len: ", len(x1_seq))

    if TRAINING:
        model.summary()
        opt = keras.optimizers.Adam(learning_rate=0.001)
        if UPDATING:
            model.load_weights(load_chk_path).expect_partial()
        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])
        # model.fit(x1_seq, y2_seq, epochs=20, shuffle=False, batch_size=8, verbose=2)
        model.fit(x1_seq, y1_seq, epochs=EPOCH_NUM, shuffle=False, batch_size=8, verbose=2)

        # Using an embedding layer means the input has to be ints. No thanks Jose.
        # main_input = keras.Input(shape=(10, 6), dtype='float32')  # I suppose the shape here is None, 2, 3?
        # Remember we're working with a different dataset from Todorov's. We have to fit the data to this one.
        # dense = keras.layers.Dense(64, activation='relu')(main_input)  # cf. Notes [1]
        # lstm = keras.layers.LSTM(1024, return_sequences=True)(dense)
        # main_output = keras.layers.Dense(20, activation='relu')(lstm)
        # main_output = keras.layers.Dense(6, activation='softmax')(main_output)
        # second_output = keras.layers.Dense(20, activation='relu')(lstm)
        # second_output = keras.layers.Dense(3, activation='softmax')(second_output)  # I guess we could change this to three
        # since we already have it OHE
        # model = keras.Model(inputs=[main_input, ], outputs=[main_output, second_output])
        # model.summary()
        # model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer='adam',
        #               metrics=['accuracy'], loss_weights=[1., 0.2])
        # model.fit(x1_seq, [y1_seq, y2_seq], epochs=2, shuffle=False, batch_size=8, verbose=2)


        # opt = keras.optimizers.Adam()
        # model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=tf.keras.optimizers.Adam(),
        #               metrics=['accuracy'], loss_weights=[1., 0.2])

        # model.compile(loss=[seq_loss_1, seq_loss_2], optimizer=opt, metrics=['accuracy'],
        #               loss_weights=[1., 0.2])
        # the primary objective is treated as much more important (1.0) than our secondary objective (0.2),
        # and we choose their respective loss functions - categorical crossentropy for the primary
        # (as we are choosing categories - rock, paper or scissors), and mean squared error for the number of wins.

        # for step, x_batch_train, y1_batch_train, y2_batch_train in range(len(x1_seq)), x1_seq, y1_seq, y2_seq:
        #     model.fit(x_batch_train, [y1_batch_train, y2_batch_train], epochs=20, shuffle=False, verbose=2)



        # # Now we train the model (and begin praying)
        # for i in range(len(x1_seq)):
        #     X_, y_1, y_2 = x1_seq[i], y1_seq[i], y2_seq[i]
        #     X_ = np.reshape(X_, (1, 10, 6))
        #     y_1 = np.reshape(y_1, (1, 10, 6))
        #     y_2 = np.reshape(y_2, (1, 10, 3))
        #     # verbose = 2 if (i % 100 == 0) else 0  # print only 1/25th of the time
        #
        # model.fit(X_, [y_1, y_2], epochs=1, shuffle=False, batch_size=1, verbose=2)

        # model.fit(x1_seq, [y1_seq, y2_seq], epochs=2, shuffle=False, batch_size=8, verbose=2)

        # X_, y_1, y_2 = np.reshape(x1_seq, (95, 20, 6)), np.reshape(y1_seq, (95, 20, 6)), np.reshape(y2_seq, (95, 20, 3))
        # X_, y_1, y_2 = tf.convert_to_tensor(X_), tf.convert_to_tensor(y_1), tf.convert_to_tensor(y_2)
        # # verbose = 2 if (i % 10 == 0) else 0  # print only 1/25th of the time
        # model.fit(X_, [y_1, y_2], epochs=1, shuffle=False, batch_size=20)
        #
        # Save the weights
        model.save_weights(last_saved_chk_path)
        print("Iteration number ", TRAINING_ITER)
        print("Weights saved to ", last_saved_chk_path)
    else:
        model.load_weights(load_chk_path).expect_partial()  # We are OK with working with weights only. This silences warnings.
        model.summary()

