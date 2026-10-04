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
env = gym.make('ALE/Tetris-v5', render_mode='rgb_array')

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

#The amount of choices the agent has.
Choices = 6

#The internal identifier of each choice.
Choice_IDs = [0, 1, 2, 3, 4]

#The AI model, made out of a Deep Q Network. It learns the best action for the given screen.
class Lazy_Gamer(nn.Module):
    
    #Runs the following commands whenever a model is created.
    def __init__(self, action_count):

        #Sets up the class using Pytorch.
        super().__init__()

        #When an object is created, the data is run through different layers in the order below.
        self.network = nn.Sequential
        (
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
    def Make_Choice(self, states):

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
def make_choice(policy_net, state, allowed_actions, button_mash):

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

#Show the game window with current attempt counter.
def computer_make_game(observation, score, attempt, display, font):

    #Creates a pygame image from the pixel data to display to humans.
    frame = pygame.surfarray.make_surface(np.transpose(observation, (1, 0, 2)))

    #Make image bigger so that more than just ants can see it.
    frame = pygame.transform.scale(frame, (frame.get_width() * 3, frame.get_height() * 3))

    #Paints over the old game to remove old pixels.
    display.fill((20, 20, 20))

    #Creates the full game image starting from the top left.
    display.blit(frame, (0, 0))

    #Creates the text to display the amount of attempts the agent has done.
    attempt_text = font.render(f"Attempt: {attempt}", True, (255, 220, 80))

    #Saves the height of the game image.
    image_height = frame.get_height()

    #Display the attempt counter on the bottom of the screen
    display.blit(attempt_text, (12, image_height + 42))

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
def long_term_memory(policy_net, optimizer, episode, best_score, button_mash, steps):
    
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
            "steps": steps,
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

    
    try:
        value = key.char
    except AttributeError:
        value = str(key)
    record_input("keyboard", "press", value)
    if key in KEY_ACTIONS:
        keyboard_controls.put(KEY_ACTIONS[key])
    elif value == ROTATE_CLOCKWISE_COMMAND:
        keyboard_controls.put(4)
        record_input("keyboard", "command", "rotate +90")

def on_release(key):
    value = getattr(key, "char", str(key))
    record_input("keyboard", "release", value)
    if key == keyboard.Key.esc:
        stop_requested.set()
        return False

try:
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    pygame.init()
    hud_font = pygame.font.Font(None, 30)
    policy_net = Lazy_Gamer(Choices).to(Brain)
    target_net = Lazy_Gamer(Choices).to(Brain)
    optimizer = Adam(policy_net.parameters(), lr=Change_Rate)
    replay_buffer = Save_Data(Interactions_Saved)
    best_score = float("-inf")
    button_mash = Button_Mash_Start
    total_steps = 0

    if Memory_Path.exists():
        checkpoint = torch.load(Memory_Path, map_location=Brain)
        policy_net.load_state_dict(checkpoint["model"])
        target_net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_score = checkpoint.get("best_score", best_score)
        button_mash = checkpoint.get("button_mash", button_mash)
        total_steps = checkpoint.get("steps", total_steps)
        print(f"Loaded checkpoint from {Memory_Path} on {Brain}.")

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    for episode_number in range(1, Training_Ep + 1):
        if stop_requested.is_set():
            break

        obs, info = env.reset()
        if "display" not in locals():
            display = pygame.display.set_mode((obs.shape[1] * 3, obs.shape[0] * 3 + 75))
            pygame.display.set_caption("Tetris RL Agent")
        state = colorblind(obs)
        terminated = truncated = False
        episode_score = 0.0
        losses = []
        computer_make_game(obs, episode_score, episode_number, display, hud_font)

        while not (terminated or truncated or stop_requested.is_set()):
            try:
                action = keyboard_controls.get(timeout=1 / 60)
                record_input("manual", "action", int(action))
            except Empty:
                #action = make_choice(policy_net, state, Choice_IDs, button_mash)
                #record_input("bot", "action", int(action))

                #obs, reward, terminated, truncated, info = step_environment(env, action)
                action = 0
                record_input("gravity", "action", action)

            obs, reward, terminated, truncated, info = step_environment(env, action)
            next_state = colorblind(obs)
            done = terminated or truncated
            replay_buffer.new_memory(state, action, float(reward), next_state, done)
            loss = learning_time(policy_net, target_net, replay_buffer, optimizer)
            if loss is not None:
                losses.append(loss)
            state = next_state
            episode_score += float(reward)
            computer_make_game(obs, episode_score, episode_number, display, hud_font)
            total_steps += 1
            button_mash = max(
                Button_Mash_End,
                Button_Mash_Start - (Button_Mash_Start - Button_Mash_End) * total_steps / Mash_Decay,
            )
            if total_steps % Steps_Til_Sync == 0:
                target_net.load_state_dict(policy_net.state_dict())
            sleep(1 / 60)

        average_loss = sum(losses) / len(losses) if losses else 0.0
        print(
            f"Episode {episode_number}/{Training_Ep} | "
            f"score={episode_score}\n"
            f"Attempt: {episode_number} | best={max(best_score, episode_score)} | "
            f"button_mash={button_mash:.3f} | loss={average_loss:.4f}"
        )
        record_input("episode", "score", episode_score)
        if episode_score > best_score:
            best_score = episode_score
            long_term_memory(
                policy_net,
                optimizer,
                episode_number,
                best_score,
                button_mash,
                total_steps,
            )
            print(f"Saved improved checkpoint with score {best_score}.")

    long_term_memory(policy_net, optimizer, episode_number, best_score, button_mash, total_steps)
finally:
    stop_requested.set()
    if "listener" in locals():
        listener.stop()
        listener.join()
    env.close()
    pygame.quit()
    with open("tetris_input_log.json", "w", encoding="utf-8") as log_file:
        json.dump(input_log, log_file, indent=2)
