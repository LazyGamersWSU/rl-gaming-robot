import json
import random
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock
from time import sleep, time

import ale_py
import gymnasium as gym
import numpy as np
import pygame
import torch
from PIL import Image
from pynput import keyboard
from torch import nn
from torch.optim import Adam

#Tells Reinforcement Learning enviroment where the tetris games are.
gym.register_envs(ale_py)

#This build the Tetris enviroment and creates the screen through a pixel array.
env = gym.make('ALE/Tetris-v5', render_mode='rgb_array', frameskip=4)

#Picks what part that processes the training.
Brain = torch.device("cpu")

#Stores the path to the training data file, so that the file can be updated with new progress.
Memory_Path = Path("tetris_dqn_rotations.pt")

#Limits how long the training is to 100 episodes (rounds of tetris).
Training_Ep = 100

#Limits each update to 32 interactions, makes each update manageable for the agent.
Update_Size = 32

#Saves the most recent 20,000 interactions for the agent to recall later.
Interactions_Saved = 20_000

#The level to which the bot cares about future rewards. 0.99 means future rewards are 99% as important as current rewards.
Reward_Caring = 0.99

#How much the agent changes per update.
Change_Rate = 1e-4

#How many steps until the main network copies over to the target network.
Steps_Til_Sync = 1_000

#How random the agent's choices are at the beginning.
Button_Mash_Start = 1.0

#How random the agent's choices are at the end, some randomness helps keep the agent from getting stuck
Button_Mash_End = 0.05

#The max amount of steps until no more goofing is allowed.
Mash_Decay = 50_000

#Simulates latency between action selection and then execution
Dummy_latency = .1

#The amount of choices the agent has.
Choices = 5

#The internal identifier of each choice.
Choice_IDs = [0, 1, 2, 3, 4]

#The AI model, made out of a Deep Q Network. It learns the best action for the given screen.
class Lazy_Gamer(nn.Module):
    
    #Runs the following commands whenever a model is created.
    def __init__(self, action_count):

        #Sets up the class using Pytorch.
        super().__init__()

        #When an object is created, the data is run through different layers in the order below.
        self.network = nn.Sequential(
            #Scans for basic shapes and edges 4 pixels at a time.
            nn.Conv2d(1, 32, kernel_size=8, stride=4),

            #Adds nonlinearity, which means the network can learn visual patterns and complex relationships.
            nn.ReLU(),

            #Scans for full columns of blocks, empty spaces, and falling shapes.
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            #Scans the board in fine detail to refine patterns.
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            #Turns 3D images into 1D for later input.
            nn.Flatten(),

            #Turns the 1D image into a feature vector to feed to the AI.
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),

            #Compares the choices to the feture vector and provides which choice would give the best result (Q-Value).
            nn.Linear(512, action_count),
        )

    #Function to call that requires a state as an input.
    def forward(self, states):

        #Take the input and run it through the layers above, return back with the choice made.
        return self.network(states)

#Creates a memory structure to save past choices and outcomes.
class Save_Data:
    def __init__(self, capacity):

        #Creates a double ended queue to save past experiences.
        self.memory = deque(maxlen=capacity)

    #Function to add another experience to the replay buffer.
    def new_memory(self, state, action, reward, next_state, done):

        #Save these 5 values as one experience in the replay buffer.
        self.memory.append((state, action, reward, next_state, done))

    #Function to recall a certain amount of updates from the replay buffer.
    def remember(self, update_size):

        #Randomly selects previous interactions from the replay buffer and returns it to the active agent.
        return random.sample(self.memory, update_size)

    #Function to check how many experiences are inside the replay buffer currently.
    def __len__(self):
        return len(self.memory)

#Function to turn a screenshot into a greyscale image for the Gamer.
def colorblind(observation):

    #Converts pixel data into a new greyscale image.
    image = Image.fromarray(observation).convert("L")

    #Resize the image so it can be processed.
    image = image.resize((84, 84), Image.Resampling.BILINEAR)

    #Copies the image and returns it to the replay buffer.
    return np.asarray(image, dtype=np.uint8).copy()

#Function to take the current data and make a choice.
def forward(policy_net, state, allowed_actions, button_mash):

    #Check to see if the Gamer is button mashing.
    if random.random() < button_mash:

        #MASH THE BUTTONS.
        return random.choice(allowed_actions)

    #Converts the image to a Pytorch tensor (array) for the image.
    image_to_tensor = torch.from_numpy(state).to(Brain).float().div(255).unsqueeze(0).unsqueeze(0)

    #Do not track gradient calculations.
    with torch.no_grad():

        #Decide how good each choice is.
        q_values = policy_net(image_to_tensor)[0]

        #Clones Q values to keep the original Q values unchanged.
        masked_q_values = q_values.clone()

        #Remove options that will not work by setting the impossible choices to negative infinity.
        masked_q_values[:] = float("-inf")

        #Compares original Q values with the impossible choices, giving us a list of all valid options.
        masked_q_values[allowed_actions] = q_values[allowed_actions]

        #Return the valid choices to the Gamer.
        return int(masked_q_values.argmax().item())

#Time to press the button and see what happens.
def step_environment(env, action):

    #Map ALE commands to choices to make.
    ale_actions = {
        #Nothing
        0: (0,),
        #Right.
        1: (2,), 
        #Left.
        2: (3,),  
        #Down.
        3: (4,),      
        #Rotate clockwise 90 degrees.
        4: (1,),       
    }

    #How much the Gamer succeeded.
    total_reward = 0.0

    #See when the game ends.
    terminated = truncated = False

    #Holds the current screen data.
    observation = None

    #Empty space to hold additional information.
    info = {}

    #Loop actions until the game is stopped.
    for ale_action in ale_actions[action]:

        #Complete one action in the Atari emulator and return the results.
        observation, reward, terminated, truncated, info = env.step(ale_action)

        #Update how much the agent succeeded.
        total_reward += float(reward)

        #Look to see if the game is closed.
        if terminated or truncated:
            break
    return observation, total_reward, terminated, truncated, info

#Show the game window with the current mode and attempt counter.
def computer_make_game(observation, score, attempt, agent_playing, display, font):

    #Creates a pygame image from the pixel data to display to humans.
    frame = pygame.surfarray.make_surface(np.transpose(observation, (1, 0, 2)))

    #Make image bigger so that more than just ants can see it.
    frame = pygame.transform.scale(frame, (frame.get_width() * 3, frame.get_height() * 3))

    #Paints over the old game to remove old pixels.
    display.fill((20, 20, 20))

    #Creates the full game image starting from the top left.
    display.blit(frame, (0, 0))

    #Saves the height of the game image.
    image_height = frame.get_height()

    #Display who is playing and the current attempt number.
    mode_text = font.render(
        "Gamer is playing" if agent_playing else "You are playing",
        True,
        (120, 220, 160) if agent_playing else (120, 190, 255),
    )
    swap_to = "Human" if agent_playing else "Gamer"
    swap_text = font.render(f"Press Space to swap to {swap_to}", True, (230, 230, 230))
    attempt_text = font.render(f"Attempt: {attempt}", True, (255, 220, 80))

    display.blit(mode_text, (12, image_height + 6))
    display.blit(swap_text, (12, image_height + 38))
    display.blit(attempt_text, (12, image_height + 72))

    #Make everything visible to humans.
    pygame.display.flip()

    #Keep it going until the game is told to quit.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            stop_requested.set()

#Time to for Gamer to reflect on his actions.
def learning_time(policy_net, target_net, replay_buffer, optimizer):

    #Check to see if we have made enough choices to learn from.
    if len(replay_buffer) < Update_Size:
        return None

    #Gather a selection of previous interactions.
    short_term = replay_buffer.remember(Update_Size)

    #Seperate the interactions into what actions were taken and what happened because of it.
    states, actions, rewards, next_states, dones = zip(*short_term)

    #Convert the current given image state into Pytorch tensors.
    image_to_tensor = torch.from_numpy(np.stack(states)).to(Brain).float().div(255).unsqueeze(1)

    #Convert the next given image state into Pytorch tensor.
    next_image_to_tensor = torch.from_numpy(np.stack(next_states)).to(Brain).float().div(255).unsqueeze(1)

    #Converts the choice made into Pytorch tensor.
    choice_to_tensor = torch.tensor(actions, device=Brain, dtype=torch.long).unsqueeze(1)

    #Converts the total reward amount Pytorch tensor.
    reward_to_tensor = torch.tensor(rewards, device=Brain, dtype=torch.float32)

    #Tells the Pytorch if the most recent choice ended the game.
    final_action = torch.tensor(dones, device=Brain, dtype=torch.float32)

    #Gathers how impactful each choice was to take.
    current_q_values = policy_net(image_to_tensor).gather(1, choice_to_tensor).squeeze(1)

    
    with torch.no_grad():

        #Estimate the best possible action for the next state.
        next_q_values = target_net(next_image_to_tensor).max(dim=1).values

        #What actions the Gamer should look for.
        target_q_values = reward_to_tensor + Reward_Caring * next_q_values * (1 - final_action)

    #How far off the Gamer is from making the right choice.
    loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)

    #Clears out old training information.
    optimizer.zero_grad()

    #How much the Gamer needs to adjust.
    loss.backward()

    #Scales down the update to keep the training stable.
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10)

    #Informs the Gamer how much it needs to change, makes the Gamer learn.
    optimizer.step()

    #Record how far off the Gamer was for later use. 
    return float(loss.item())

#Saves the progress to Gamer's long term memory.
def long_term_memory(policy_net, optimizer, episode, best_score, button_mash, total_env_steps):
    
    torch.save(
        {
            #Saves the impact of each choice made.
            "model": policy_net.state_dict(),

            #Updates the optimizer with past choices and results.
            "optimizer": optimizer.state_dict(),

            #Saves how many episodes the bot went for.
            "episode": episode,

            #Saves how well the bot did
            "best_score": best_score,

            #Save how random the Gamers inputs are
            "button_mash": button_mash,

            #Save how many environment steps happened at this point.
            "total_env_steps": total_env_steps,
        },

        #File path to store the memory to.
        Memory_Path,
    )

#List to store inputs
input_log = []

#Makes sure all simultaneous input events are recorded correctly.
input_log_lock = Lock()

#Looks to see if the game has been told to turn off.
stop_requested = Event()

#Creates a list of keyboard inputs for the game to use.
keyboard_controls = Queue()

#Starts in agent mode; Space toggles between agent and keyboard control.
agent_mode = Event()
agent_mode.set()
space_held = False

#Record current time.
started_at = time()

#Map keyboard inputs to choices ingame.
KEY_ACTIONS = {
    keyboard.Key.right: 1,
    keyboard.Key.left: 2,
    keyboard.Key.down: 3,
}
ROTATE_CLOCKWISE_COMMAND = "z"

#Record and display the choice made.
def record_input(source, event, value):
    entry = {

        #How long Tetris has been running. 
        "time": round(time() - started_at, 6),

        #Who made the choice, either keyboard or Gamer.
        "source": source,

        #How they made the choice, keyboard press or decision making.
        "event": event,

        #What choice was made.
        "value": value,
    }
    with input_log_lock:

        #Update with new choice entry.
        input_log.append(entry)

    #Print choice info to console.
    print(entry)

#Listen for keyboard inputs
def on_press(key):
    global space_held

    #Output the text for normal keys
    try:
        value = key.char

    #Output special keys as key.special_key
    except AttributeError:
        value = str(key)

    #Record what key was pressed.
    record_input("keyboard", "press", value)

    #Spacebar now swaps between the Gamer and human control.
    if key == keyboard.Key.space:

        #Does not work if you hold down the spacebar, only works on the first press.
        if not space_held:
            space_held = True

            #Swap from Gamer to human control.
            if agent_mode.is_set():
                agent_mode.clear()
                mode = "keyboard"
            else:

                #Swap from human control to Gamer.
                agent_mode.set()
                mode = "agent"
                while True:
                    try:
                        keyboard_controls.get_nowait()
                    except Empty:
                        break
            record_input("mode", "toggle", mode)
        return

    if not agent_mode.is_set():
        #If the key pressed is a valid choice, map the key to that choice.
        if key in KEY_ACTIONS:
            keyboard_controls.put(KEY_ACTIONS[key])

        #Rotate clockwise is bound to z, so it needs its own special line.
        elif value == ROTATE_CLOCKWISE_COMMAND:
            keyboard_controls.put(4)
            record_input("keyboard", "command", "rotate +90")

    #If esc is hit, close the game and end the training session.
    if key == keyboard.Key.esc:
        stop_requested.set()
        return False

#Signifies that the spacebar is no longer being held down.
def on_release(key):
    global space_held

    if key == keyboard.Key.space:
        space_held = False

def get_controller_action():
    for event in pygame.event.get():
 
        if event.type == pygame.QUIT:
            stop_requested.set()
            return None
 
        if event.type == pygame.JOYBUTTONDOWN:
            print(f"Controller button: {event.button}")
 
            if event.button == 3:       # Y
                return 4
            elif event.button == 0:     # X
                stop_requested.set()
                return None
 
        if event.type == pygame.JOYAXISMOTION:
 
            if event.axis == 0:
                if event.value > 0.5:
                    return 1       # Right
                elif event.value < -0.5:
                    return 2       # Left
 
            elif event.axis == 1 and event.value > 0.5:
                    return 3       # Down
 
    return 0

try:

    #Starts the keyboard input listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    #Starts up the Tetris window.
    pygame.init()
    
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No controller detected!")
 
    controller = pygame.joystick.Joystick(0)
    controller.init()
 
    print(f"Controller detected: {controller.get_name()}")
 
    # Added these 2 lines to test controller diagnostics
    print(f"Buttons: {controller.get_numbuttons()}")
    print(f"Axes: {controller.get_numaxes()}")

    hud_font = pygame.font.Font(None, 30)

    #Wake up the Gamer.
    policy_net = Lazy_Gamer(Choices).to(Brain)

    #Create goals for the Gamer to stride for.
    target_net = Lazy_Gamer(Choices).to(Brain)

    #Create the optimizer to smooth out learning.
    optimizer = Adam(policy_net.parameters(), lr=Change_Rate)

    #Create the replay buffer to save exeriences of this session.
    replay_buffer = Save_Data(Interactions_Saved)

    #Start the high score as low as possible.
    best_score = float("-inf")

    #Reset how much exploration happwns in the beginning.
    button_mash = Button_Mash_Start

    #Fresh environment so no steps have happened.
    total_env_steps = 0

    #Load saved memories if some exist.
    if Memory_Path.exists():

        #Loads previous Pytorch tensors into the brain.
        checkpoint = torch.load(Memory_Path, map_location=Brain)

        #Load saved choice weight into the Gamer, so it can use what it has learned.
        policy_net.load_state_dict(checkpoint["model"])

        #Load the same information into the target for a good starting point.
        target_net.load_state_dict(checkpoint["model"])

        #Load in previous optimize data, so training can go smoothly.
        optimizer.load_state_dict(checkpoint["optimizer"])

        #Load in best score Gamer has schieved so it knows what to aim to beat.
        best_score = checkpoint.get("best_score", best_score)

        #Load in how much the Gamer is supposed to explore to make sure the appropriate amount of randomness is happening.
        button_mash = checkpoint.get("button_mash", button_mash)

        #Load in total amount of environment steps so far, keeps scheduling consisten for copying over the network to target network.
        total_env_steps = checkpoint.get("total_env_steps", total_env_steps)
        print(f"Loaded checkpoint from {Memory_Path} on {Brain}.")

    #Match target network and current network incase a save file does not already exist.
    target_net.load_state_dict(policy_net.state_dict())

    #Record the amount of episodes played until the game is closed.
    for episode_number in range(1, Training_Ep + 1):
        if stop_requested.is_set():
            break

        
        obs, info = env.reset()

        #Make sure the game window is only created once instead of each episode.
        if "display" not in locals():
            display = pygame.display.set_mode((obs.shape[1] * 3, obs.shape[0] * 3 + 115))
            pygame.display.set_caption("Tetris RL Agent")

        #Converts the current image and converts it into a format the Gamer can recognize.
        state = colorblind(obs)

        #Dont end the episode upon creating it.
        terminated = truncated = False

        #Brand new attempt, so the score is zero.
        episode_score = 0.0

        #Keep track of how far off the Gamer was for this attempt.
        losses = []

        #Display everything that was created in a Pygame window for humans to see.
        computer_make_game(obs, episode_score, episode_number, agent_mode.is_set(), display, hud_font)

        #Listen for inputs as long as the game is active
        while not (terminated or truncated or stop_requested.is_set()):
            if agent_mode.is_set():
                action = forward(policy_net, state, Choice_IDs, button_mash)
                record_input("Gamer", "action", int(action))
                sleep(Dummy_latency)
            else:
                action = get_controller_action()

                if action is None:
                    break

                if action != 0:
                    print(f"CONTROLLER ACTION: {action}")
                    record_input("controller", "action", action)

            #Apply the choice to the game and produce the next game image.
            obs, reward, terminated, truncated, info = step_environment(env, action)

            #Turn the next game image into a game state for choice making and loop.
            next_state = colorblind(obs)

            #Check to see if the game has been stopped.
            done = terminated or truncated

            if agent_mode.is_set():
                #Update the Gamer's memory with a new experience.
                replay_buffer.new_memory(state, action, float(reward), next_state, done)

                #Update to see how far the gamer was from succeeding this time.
                loss = learning_time(policy_net, target_net, replay_buffer, optimizer)

                #Keep a running total to see how many times the Gamer is wrong.
                if loss is not None:
                    losses.append(loss)

            #Replace the old state with the new one for the Gamer to make a choice on.
            state = next_state

            #Update how many points the Gamer got this attempt.
            episode_score += float(reward)

            #Update the Pygame window to reflect the choices and results.
            computer_make_game(obs, episode_score, episode_number, agent_mode.is_set(), display, hud_font)

            if agent_mode.is_set():
                #Count agent-controlled steps and update its exploration schedule.
                total_env_steps += 1
                button_mash = max(
                    Button_Mash_End,
                    Button_Mash_Start - (Button_Mash_Start - Button_Mash_End) * total_env_steps / Mash_Decay,
                )

                #Update the target network at the configured step interval.
                if total_env_steps % Steps_Til_Sync == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            sleep(1 / 60)

        #See on average how off the Gamer was from scoring.
        average_loss = sum(losses) / len(losses) if losses else 0.0
        print(

            #Print out a summary of the game played to see how well Gamer did.
            f"Episode {episode_number}/{Training_Ep} | "
            f"score={episode_score}\n"
            f"Attempt: {episode_number} | best={max(best_score, episode_score)} | "
            f"button_mash={button_mash:.3f} | loss={average_loss:.4f}"
        )

        #Save the episodes score for later reporting.
        record_input("episode", "score", episode_score)

        #Check to see if the Gamer's highscore needs to be updated.
        if episode_score > best_score:
            best_score = episode_score

            #Update the long term memory with the state the Gamer got the high score in.
            long_term_memory(
                policy_net,
                optimizer,
                episode_number,
                best_score,
                button_mash,
                total_env_steps,
            )
            print(f"Saved improved checkpoint with score {best_score}.")

    #Update the Gamer's long term memory to save the progress he has made.
    long_term_memory(policy_net, optimizer, episode_number, best_score, button_mash, total_env_steps)
finally:

    #Tell the program to end.
    stop_requested.set()

    #Stop listening to inputs.
    if "listener" in locals():
        listener.stop()
        listener.join()

    #Turn off the tetris environment.
    env.close()

    #Shut her down, no more Tetris.
    pygame.quit()

    #Update the input log made for later reporting.
    with open("tetris_input_log.json", "w", encoding="utf-8") as log_file:
        json.dump(input_log, log_file, indent=2)
