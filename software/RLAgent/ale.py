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
Goofing_Amount_Start = 1.0

#How random the agent's choices are at the end, some randomness helps keep the agent from getting stuck
Goofing_Amount_End = 0.05

#The max amount of steps until no more goofing is allowed.
Goof_Decay_Rate = 50_000

#The amount of choices the agent has.
Choices = 6

#The internal identifier of each choice.
Choice_IDs = [0, 1, 2, 3, 4, 5]

#The AI model, made out of a Deep Q Network. It learns the best action for the given screen.
class Lazy_Gamer(nn.Module):
    
    #Runs the following commands whenever an object using the network is created.
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

    #Function that runs whenever the agent recieves and input
    def Make_Choice(self, states):

        #Take the input and run it through the layers above, return back with the choice made.
        return self.network(states)

#Creates a memory structure to save past choices and outcomes.
class Save_Data:
    def __init__(self, capacity):

        #Creates a double ended queue to save past experiences.
        self.memory = deque(maxlen=capacity)

    
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def preprocess(observation):
    image = Image.fromarray(observation).convert("L")
    image = image.resize((84, 84), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8).copy()


def select_action(policy_net, state, allowed_actions, epsilon):
    if random.random() < epsilon:
        return random.choice(allowed_actions)
    state_tensor = torch.from_numpy(state).to(Brain).float().div(255).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_tensor)[0]
        masked_q_values = q_values.clone()
        masked_q_values[:] = float("-inf")
        masked_q_values[allowed_actions] = q_values[allowed_actions]
        return int(masked_q_values.argmax().item())


def step_environment(env, action):
    """Translate an abstract action into one or more ALE actions."""
    ale_actions = {
        0: (0,),       # No-op: the block continues falling.
        1: (2,),       # Right.
        2: (3,),       # Left.
        3: (4,),       # Down.
        4: (1,),       # Rotate clockwise 90 degrees.
        5: (1, 1, 1),  # Rotate counterclockwise 90 degrees.
    }
    total_reward = 0.0
    terminated = truncated = False
    observation = None
    info = {}
    for ale_action in ale_actions[action]:
        observation, reward, terminated, truncated, info = env.step(ale_action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    return observation, total_reward, terminated, truncated, info


def update_game_display(observation, score, attempt, display, font):
    """Show the Atari frame with the current score and attempt counter."""
    frame = pygame.surfarray.make_surface(np.transpose(observation, (1, 0, 2)))
    frame = pygame.transform.scale(frame, (frame.get_width() * 3, frame.get_height() * 3))
    display.fill((20, 20, 20))
    display.blit(frame, (0, 0))
    score_text = font.render(f"Score: {score:.0f}", True, (255, 255, 255))
    attempt_text = font.render(f"Attempt: {attempt}", True, (255, 220, 80))
    panel_y = frame.get_height()
    display.blit(score_text, (12, panel_y + 8))
    display.blit(attempt_text, (12, panel_y + 42))
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            stop_requested.set()


def optimize_model(policy_net, target_net, replay_buffer, optimizer):
    if len(replay_buffer) < Update_Size:
        return None

    batch = replay_buffer.sample(Update_Size)
    states, actions, rewards, next_states, dones = zip(*batch)
    state_tensor = torch.from_numpy(np.stack(states)).to(Brain).float().div(255).unsqueeze(1)
    next_state_tensor = torch.from_numpy(np.stack(next_states)).to(Brain).float().div(255).unsqueeze(1)
    action_tensor = torch.tensor(actions, device=Brain, dtype=torch.long).unsqueeze(1)
    reward_tensor = torch.tensor(rewards, device=Brain, dtype=torch.float32)
    done_tensor = torch.tensor(dones, device=Brain, dtype=torch.float32)

    current_q_values = policy_net(state_tensor).gather(1, action_tensor).squeeze(1)
    with torch.no_grad():
        next_q_values = target_net(next_state_tensor).max(dim=1).values
        target_q_values = reward_tensor + Reward_Caring * next_q_values * (1 - done_tensor)

    loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
    optimizer.step()
    return float(loss.item())


def save_checkpoint(policy_net, optimizer, episode, best_score, epsilon, steps):
    torch.save(
        {
            "model": policy_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "episode": episode,
            "best_score": best_score,
            "epsilon": epsilon,
            "steps": steps,
        },
        Memory_Path,
    )

input_log = []
input_log_lock = Lock()
stop_requested = Event()
manual_actions = Queue()
started_at = time()
KEY_ACTIONS = {
    keyboard.Key.right: 1,
    keyboard.Key.left: 2,
    keyboard.Key.down: 3,
}
ROTATE_CLOCKWISE_COMMAND = "z"
ROTATE_COUNTERCLOCKWISE_COMMAND = "x"


def record_input(source, event, value):
    entry = {
        "time": round(time() - started_at, 6),
        "source": source,
        "event": event,
        "value": value,
    }
    with input_log_lock:
        input_log.append(entry)
    print(entry)


def on_press(key):
    try:
        value = key.char
    except AttributeError:
        value = str(key)
    record_input("keyboard", "press", value)
    if key in KEY_ACTIONS:
        manual_actions.put(KEY_ACTIONS[key])
    elif value == ROTATE_CLOCKWISE_COMMAND:
        manual_actions.put(4)
        record_input("keyboard", "command", "rotate +90")
    elif value == ROTATE_COUNTERCLOCKWISE_COMMAND:
        manual_actions.put(5)
        record_input("keyboard", "command", "rotate -90")

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
    epsilon = Goofing_Amount_Start
    total_steps = 0

    if Memory_Path.exists():
        checkpoint = torch.load(Memory_Path, map_location=Brain)
        policy_net.load_state_dict(checkpoint["model"])
        target_net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_score = checkpoint.get("best_score", best_score)
        epsilon = checkpoint.get("epsilon", epsilon)
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
        state = preprocess(obs)
        terminated = truncated = False
        episode_score = 0.0
        losses = []
        update_game_display(obs, episode_score, episode_number, display, hud_font)

        while not (terminated or truncated or stop_requested.is_set()):
            try:
                action = manual_actions.get(timeout=1 / 60)
                record_input("manual", "action", int(action))
            except Empty:
                #action = select_action(policy_net, state, Choice_IDs, epsilon)
                #record_input("bot", "action", int(action))

                #obs, reward, terminated, truncated, info = step_environment(env, action)
                action = 0
                record_input("gravity", "action", action)

            obs, reward, terminated, truncated, info = step_environment(env, action)
            next_state = preprocess(obs)
            done = terminated or truncated
            replay_buffer.add(state, action, float(reward), next_state, done)
            loss = optimize_model(policy_net, target_net, replay_buffer, optimizer)
            if loss is not None:
                losses.append(loss)
            state = next_state
            episode_score += float(reward)
            update_game_display(obs, episode_score, episode_number, display, hud_font)
            total_steps += 1
            epsilon = max(
                Goofing_Amount_End,
                Goofing_Amount_Start - (Goofing_Amount_Start - Goofing_Amount_End) * total_steps / Goof_Decay_Rate,
            )
            if total_steps % Steps_Til_Sync == 0:
                target_net.load_state_dict(policy_net.state_dict())
            sleep(1 / 60)

        average_loss = sum(losses) / len(losses) if losses else 0.0
        print(
            f"Episode {episode_number}/{Training_Ep} | "
            f"score={episode_score}\n"
            f"Attempt: {episode_number} | best={max(best_score, episode_score)} | "
            f"epsilon={epsilon:.3f} | loss={average_loss:.4f}"
        )
        record_input("episode", "score", episode_score)
        if episode_score > best_score:
            best_score = episode_score
            save_checkpoint(
                policy_net,
                optimizer,
                episode_number,
                best_score,
                epsilon,
                total_steps,
            )
            print(f"Saved improved checkpoint with score {best_score}.")

    save_checkpoint(policy_net, optimizer, episode_number, best_score, epsilon, total_steps)
finally:
    stop_requested.set()
    if "listener" in locals():
        listener.stop()
        listener.join()
    env.close()
    pygame.quit()
    with open("tetris_input_log.json", "w", encoding="utf-8") as log_file:
        json.dump(input_log, log_file, indent=2)
