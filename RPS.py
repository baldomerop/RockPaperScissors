# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago.
# It is not a very good player so you will need to change the code to pass the challenge.
import random
import numpy as np
import csv
# import tensorflow as tf
from nn22 import SEQ_LENGTH, TRAINING_ITER, NUM_GAMES, DENSE_NEURONS, RNN_UNITS,\
    last_saved_chk_path, move_to_ohe, build_model


model = build_model(SEQ_LENGTH, 6, RNN_UNITS, DENSE_NEURONS)
model.load_weights(last_saved_chk_path).expect_partial()
model.reset_states()  # This might be the piece we're missing to make it work. Update: nope, it didn't do anything.
# quincy_checkpoint = './checkpoints/quincy_chkp_working'
# model.load_weights(quincy_checkpoint).expect_partial()  # vs. Quincy


reply_ohe = ["P", "S", "R"]  # Note this is inverted to reply with the move that WINS vs the predicted move.


def nn_player_abbey_TEST(prev_play, opponent_history=[], player_history=[]):
    write_history = False
    file_path = f'db1_nn_abbey_n{NUM_GAMES}_0{TRAINING_ITER}.csv'  # It will create a new file if it doesn't find one.

    opponent_history.append(prev_play)
    print(opponent_history)
    if not opponent_history == ['']:
        opponent_history = opponent_history[1:]  # n.b. unlike in the other history player function, here we have to
        # slice op. history at this point for the sequence generation to work.

    if len(opponent_history) < SEQ_LENGTH:
        guess = random.choice(["R", "P", "S"])
        player_history.append(guess)

        return guess
    p1_sequence = opponent_history[-(SEQ_LENGTH):]
    p2_sequence = player_history[-(SEQ_LENGTH):]
    game_sequence = np.column_stack((p1_sequence, p2_sequence))
    # print(game_sequence)
    ohe_sequence = np.array([[move_to_ohe[val] for val in row] for row in game_sequence])
    # # print("Before reshaping, shape is ", np.shape(ohe_sequence))
    # # print("Pre-reshape sequence: ", ohe_sequence)
    # ohe_sequence = np.reshape(ohe_sequence, (1, SEQ_LENGTH, 6))
    # # print(np.shape(ohe_sequence))
    # # print("Current sequence: ", ohe_sequence)
    #
    # reply = model(ohe_sequence)
    #
    # reply = int(np.argmax(reply[0][-1]))  # You're trying to predict the opponent's next move.
    # print("Model predicted: ", reply)
    # reply = reply_ohe[reply]  # This converts the reply back to a single letter and inverts it as well.
    # player_history.append(reply)
    #
    # if write_history:
    #     if len(opponent_history) == (NUM_GAMES - 1):
    #         # history = np.column_stack((opponent_history[1:], player_history[:-1]))  # I think we have to remove the
    #         # slicing from op. history because we already did it above.
    #         history = np.column_stack((opponent_history, player_history[:-1]))  # P1, left, will be the opponent.
    #         # print("History: ", history)
    #         print("ping!")
    #         print("History saved at ", file_path)
    #         with open(file_path, 'w', newline='') as csvfile:
    #             writer = csv.writer(csvfile)
    #             writer.writerow(["Opponent (bot)", "Dave's Player"])
    #             for row in history:
    #                 writer.writerow(row)
    #

    guess = random.choice(["R", "P", "S"])
    player_history.append(guess)

    return guess

    # print("Now playing: ", reply)
    # return reply


def nn_player_abbey(prev_play, opponent_history=[], player_history=[]):
    write_history = True
    file_path = f'db1_nn_abbey_n{NUM_GAMES}_0{TRAINING_ITER}.csv'  # It will create a new file if it doesn't find one.

    opponent_history.append(prev_play)
    if not opponent_history == ['']:
        opponent_history = opponent_history[1:]  # n.b. unlike in the other history player function, here we have to
        # slice op. history at this point for the sequence generation to work.

    if len(opponent_history) < SEQ_LENGTH:
        guess = random.choice(["R", "P", "S"])
        player_history.append(guess)

        return guess

    p1_sequence = opponent_history[-(SEQ_LENGTH):]
    p2_sequence = player_history[-(SEQ_LENGTH):]
    # print(opponent_sequence)
    # print(player_sequence)
    game_sequence = np.column_stack((p1_sequence, p2_sequence))
    # print(opponent_history)
    # print(player_history)
    # print(game_sequence)
    ohe_sequence = np.array([[move_to_ohe[val] for val in row] for row in game_sequence])
    # print("Before reshaping, shape is ", np.shape(ohe_sequence))
    # print("Pre-reshape sequence: ", ohe_sequence)
    ohe_sequence = np.reshape(ohe_sequence, (1, SEQ_LENGTH, 6))
    # print(np.shape(ohe_sequence))
    # print("Current sequence: ", ohe_sequence)

    reply = model(ohe_sequence)

    reply = int(np.argmax(reply[0][-1]))  # You're trying to predict the opponent's next move.
    print("Model predicted: ", reply)
    reply = reply_ohe[reply]  # This converts the reply back to a single letter and inverts it as well.
    player_history.append(reply)

    if write_history:
        if len(opponent_history) == (NUM_GAMES - 1):
            # history = np.column_stack((opponent_history[1:], player_history[:-1]))  # I think we have to remove the
            # slicing from op. history because we already did it above.
            history = np.column_stack((opponent_history, player_history[:-1]))  # P1, left, will be the opponent.
            # print("History: ", history)
            print("ping!")
            print("History saved at ", file_path)
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Opponent (bot)", "Dave's Player"])
                for row in history:
                    writer.writerow(row)

    print("Now playing: ", reply)
    return reply


def nn_player_quincy(prev_play, opponent_history=[], player_history=[]):
    write_history = True
    file_path = f'db1_nn_quincy_n{NUM_GAMES}_test.csv'  # It will create a new file if it doesn't find one.

    opponent_history.append(prev_play)
    if not opponent_history == ['']:
        opponent_history = opponent_history[1:]  # n.b. unlike in the other history player function, here we have to
        # slice op. history at this point for the sequence generation to work.

    if len(opponent_history) <= SEQ_LENGTH:
        guess = random.choice(["R", "P", "S"])
        player_history.append(guess)

        return guess

    p1_sequence = opponent_history[-SEQ_LENGTH:]
    p2_sequence = player_history[-SEQ_LENGTH:]
    # print(opponent_sequence)
    # print(player_sequence)
    game_sequence = np.column_stack((p1_sequence, p2_sequence))
    # print(opponent_history)
    # print(player_history)
    # print(game_sequence)
    ohe_sequence = np.array([[move_to_ohe[val] for val in row] for row in game_sequence])
    ohe_sequence = np.reshape(ohe_sequence, (1, SEQ_LENGTH, 6))
    # print(np.shape(ohe_sequence))
    reply = model(ohe_sequence)

    reply = int(np.argmax(reply[0][-1]))  # You're trying to predict the opponent's next move.
    reply = reply_ohe[reply]  # This converts the reply back to a single letter and inverts it as well.
    player_history.append(reply)

    if write_history:
        if len(opponent_history) == (NUM_GAMES - 1):
            # history = np.column_stack((opponent_history[1:], player_history[:-1]))  # I think we have to remove the
            # slicing from op. history because we already did it above.
            history = np.column_stack((opponent_history, player_history[:-1]))  # P1, left, will be the opponent.
            # print("History: ", history)
            print("ping!")
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Opponent (bot)", "Dave's Player"])
                for row in history:
                    writer.writerow(row)

    return reply


def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)

    guess = "R"
    if len(opponent_history) > 2:
        guess = opponent_history[-2]

    return guess


def my_random_player(prev_play):
    return random.choice(["R", "P", "S"])


def random_player_history(prev_play, opponent_history=[], player_history=[]):
    guess = random.choice(["R", "P", "S"])
    player_history.append(guess)
    opponent_history.append(prev_play)
    file_path = 'db2_trained_abbey_n2000.csv'  # Wonder whether it will create a new file if it doesn't find one.
    if not opponent_history:
        pass
    else:
        history = np.column_stack((opponent_history[1:], player_history[:-1]))  # P1, left, will be the opponent.
        # print("History: ", history)

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Opponent (bot)", "Dave's Player"])
            for row in history:
                writer.writerow(row)
    return guess

