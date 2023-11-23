# This entrypoint file to be used in development. Start by reading README.md
from RSP_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import random_player_history, nn_player_abbey, nn_player_abbey_TEST, nn_player_quincy
from nn22 import build_model, SEQ_LENGTH, NUM_GAMES, load_chk_path


print("Starting play function...")
# play(random_player_history, quincy, 2001, verbose=True)
# play(nn_player_quincy, quincy, NUM_GAMES, verbose=False)
play(nn_player_abbey, abbey, NUM_GAMES, verbose=False)
# play(nn_player_abbey_TEST, abbey, 20, verbose=True)

# play(player, kris, 1000)
# play(random_player_history, mrugesh, 2000)
# play(player, mrugesh, 1000)

# Uncomment line below to play interactively against a bot:
# play(human, abbey, 20, verbose=True)

# Uncomment line below to play against a bot that plays randomly:
# play(human, random_player, 1000)



# Uncomment line below to run unit tests automatically
# main(module='test_module', exit=False)