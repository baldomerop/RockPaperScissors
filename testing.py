from dataManipulation import open_csv
import numpy as np
from dictionaries import move_to_ohe


def testing_history(file_path, i, seq_length, this_model):  # This will serve to test the bot on discrete sequences
    prev_history = file_path
    print("Prev history path:", prev_history)
    raw_data = open_csv(file_path)
    data = np.array([[move_to_ohe[val] for val in row] for row in raw_data])  # Should I leave this alone?
    data_sequence = data[i:i+seq_length]
    ohe_sequence = np.reshape(data_sequence, (1, seq_length, 6))
    reply = this_model(ohe_sequence)
    return reply
