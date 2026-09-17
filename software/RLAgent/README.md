# Overview
The software currently contains three files tetris_env, Tetris, and ale.
Tetris_env: This is imported from the gymnasium python library. This set up the tetris game mechanics including: game window, game controls, block physics, score counter.

Tetris: Another part imported from the gymnasium library, the code here allows the user to start up and test the enviroment. This normally happens with random actions.
ALE: Ale combines the previous two files into one, and adds to them. Our AI brain is a Deep Q-Network, which takes data given to it and decides the best action to take. PyTorch is what we use to build our Deep Q-Network, Torch provides the data to our Deep Q-Network based on the Network's previous actions. Previous attempts are saved in a JSON file so that the brain can look on the past actions. Currently the brain is not selecting choices because Keyboard controls are enabeled, but can be swapped out. To give the bot back control: comment out lines 215,216,263,264,308,309, and 310. Then comment in lines 259,260, and 262. 

The ALE file does not need the other two, but I like having them there to reference incase something breaks with the Tetris enviroment.
