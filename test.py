"""
This file is for testing purposes and has many of the same functions in main.py
"""

import sys
import time
from collections import defaultdict
from typing import Any

import gym
import numpy as np
import pygame
from ale_py import ALEInterface, ALEState, LoggerMode, roms
from gym.utils.play import play

from main import game_eval
from expert_agent import test_expert_agent, ExpertAgent
from neuroevolution import test_neat, run_neat
from subtask_dictionary import subtask_dict
from utils import (save_state, save_specific_state, load_specific_state, load_latest_state, take_action,
                   convert_game_name, find_most_recent_file)

# ANSI escape codes for colors
RESET_COLOR = "\033[0m"
RED_COLOR = "\033[91m"
YELLOW_COLOR = "\033[93m"
GREEN_COLOR = "\033[92m"
CYAN_COLOR = '\033[96m'
global previous_ram


def display_ram_info(action: Any, ale_env: ALEInterface, action_counts: defaultdict[Any, int],
                     ram_changes: defaultdict[Any, np.ndarray[Any, np.dtype]]
                     ) -> tuple[defaultdict[Any, int], defaultdict[Any, np.ndarray[Any, np.dtype]]]:
    """
    Displays the RAM and info about the amount of times different bytes change
    :param action: The action the user just took
    :param ale_env: The ale_env the user is playing in
    :param action_counts: The counts of how many each type of action the user has taken
    :param ram_changes: The previous changes in the ram
    :return: Tuple containing: action_counts, ram_changes
    """
    global previous_ram

    # Get RAM data
    ram = ale_env.getRAM()

    if previous_ram is None:
        previous_ram = np.zeros_like(ram)

    # Track changes for the current action
    action_counts[action] += 1
    ram_changes[action] += (ram != previous_ram)

    # Determine frequently changing bytes
    frequently_changing = ram_changes[action] / action_counts[action] > 0.25
    rarely_changing = ram_changes[action] / action_counts[action] < 0.05
    noop_frequently_changing = ram_changes[0] / action_counts[0] > 0.25 if action_counts[0] > 0 else np.zeros_like(
        frequently_changing)

    ram_str = ''
    for i, byte in enumerate(ram):
        byte_str = f'{i:03d}:'
        if frequently_changing[i] and not noop_frequently_changing[i]:
            byte_str += f'{GREEN_COLOR}{byte:03d}{RESET_COLOR} '
        elif frequently_changing[i]:
            byte_str += f'{YELLOW_COLOR}{byte:03d}{RESET_COLOR} '
        elif byte != previous_ram[i]:
            byte_str += f'{RED_COLOR}{byte:03d}{RESET_COLOR} '
        elif rarely_changing[i]:
            byte_str += f'{RESET_COLOR}{byte:03d}{RESET_COLOR} '
        else:
            byte_str += f'{CYAN_COLOR}{byte:03d}{RESET_COLOR} '
        ram_str += byte_str

    previous_ram = ram

    sys.stdout.write('\r' + ram_str.ljust(8 * len(ram)) + ' ' * 20 + str((ram[42], ram[43])))
    sys.stdout.flush()
    return action_counts, ram_changes


def play_game(game='MontezumaRevenge-v4') -> None:
    """
    Creates and runs a gym environment for the user to play a game in. It additionally shows the RAM and colors the
    different bytes based on their frequency of changing and under which action.
    :param game: The name of the game (e.g. 'MonteumaRevenge-v4')
    :return: Non
    """
    global previous_ram
    env: gym.Env = gym.make(game, render_mode='rgb_array')
    keys_to_action = {
        (pygame.K_SPACE,): 1,
        (pygame.K_UP,): 2,
        (pygame.K_RIGHT,): 3,
        (pygame.K_LEFT,): 4,
        (pygame.K_DOWN,): 5,
        (pygame.K_UP, pygame.K_RIGHT): 6,
        (pygame.K_UP, pygame.K_LEFT): 7,
        (pygame.K_DOWN, pygame.K_RIGHT): 8,
        (pygame.K_DOWN, pygame.K_LEFT): 9,
        (pygame.K_UP, pygame.K_SPACE): 10,
        (pygame.K_RIGHT, pygame.K_SPACE): 11,
        (pygame.K_LEFT, pygame.K_SPACE): 12,
        (pygame.K_DOWN, pygame.K_SPACE): 13,
        (pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE): 14,
        (pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE): 15,
        (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE): 16,
        (pygame.K_DOWN, pygame.K_LEFT, pygame.K_SPACE): 17,
    }
    ale_env: ALEInterface = env.unwrapped.ale  # Access the ALE environment
    previous_ram = None
    ram_changes: defaultdict[Any, np.ndarray[Any, np.dtype]] = defaultdict(lambda: np.zeros(128, dtype=int))
    action_counts: defaultdict[Any, int] = defaultdict(int)  # Track the number of times each action was pressed

    def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info) -> None:
        """
        Prints the RAM bytes in the console every frame, each byte formatted with different colors
        :param obs_t: Not used.
        :param obs_tp1: Not used.
        :param action: Action taken by the user.
        :param rew: Not used.
        :param terminated: Not used.
        :param truncated: Not used.
        :param info: Not used.
        :return: None
        """
        display_ram_info(action, ale_env, action_counts, ram_changes)

    play(env, keys_to_action=keys_to_action, callback=callback)


def ale_init(game: str, suppress: bool = True, repeat_action_probability: int = 0, visualize: bool = False,
             frame_skip: int = 0, seed: int = 123, load_state=None) -> ALEInterface:
    """
    Takes a game and loads it.
    :param game: Name of the game
    :param suppress: Whether to suppress info about the game running to the console
    :param repeat_action_probability: Probability to repeat the action the next frame, regardless of the agent's choice
    :param visualize: Whether to visualize the game interaction
    :param frame_skip: Number of times to repeat an action without observing
    :param seed: Random seed
    :param load_state: File to load a save state from
    :return: An ALEInterface with a game loaded
    """
    ale: ALEInterface = ALEInterface()

    if suppress:
        ale.setLoggerMode(LoggerMode.Error)

    ale.setFloat('repeat_action_probability', repeat_action_probability)
    ale.setBool('display_screen', visualize)
    ale.setInt('frame_skip', frame_skip)
    if seed is not None:
        ale.setInt('random_seed', seed)

    game = convert_game_name(game, True)
    rom = getattr(roms, game)
    ale.loadROM(rom)

    if load_state is not None:
        env_data: ALEState = load_specific_state(load_state)
        ale.restoreState(env_data)
        if not suppress:
            print(f"Game state loaded from {load_state}")
        return ale

    return ale


def human_input(ale: ALEInterface, fps=60) -> tuple[int, ALEInterface]:
    """
    Waits to achieve a certain FPS and returns the action of the user
    :param ale: ALEInterface with game loaded
    :param fps: Wanted FPS of the game
    :return: User action
    """
    keys_to_action = {
        (pygame.K_SPACE,): 1,
        (pygame.K_UP,): 2,
        (pygame.K_RIGHT,): 3,
        (pygame.K_LEFT,): 4,
        (pygame.K_DOWN,): 5,
        (pygame.K_UP, pygame.K_RIGHT): 6,
        (pygame.K_UP, pygame.K_LEFT): 7,
        (pygame.K_DOWN, pygame.K_RIGHT): 8,
        (pygame.K_DOWN, pygame.K_LEFT): 9,
        (pygame.K_UP, pygame.K_SPACE): 10,
        (pygame.K_RIGHT, pygame.K_SPACE): 11,
        (pygame.K_LEFT, pygame.K_SPACE): 12,
        (pygame.K_DOWN, pygame.K_SPACE): 13,
        (pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE): 14,
        (pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE): 15,
        (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE): 16,
        (pygame.K_DOWN, pygame.K_LEFT, pygame.K_SPACE): 17,
    }

    # Initialize pygame and create a small window to capture events
    pygame.init()
    pygame.display.set_mode((100, 100))
    pygame.display.set_caption("Human Input Capture")

    clock = pygame.time.Clock()  # Create a clock object to control the frame rate

    pressed_keys = pygame.key.get_pressed()
    active_keys = tuple(key for key, pressed in enumerate(pressed_keys) if pressed)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return 0, ale  # Exit function if window is closed
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:  # Save state
                env_data: ALEState = ale.cloneState()
                save_state(env_data)
                print("Game state saved.")
            elif event.key in {pygame.K_l, pygame.K_w}:  # Load latest state
                env_data: ALEState = load_latest_state()
                ale.restoreState(env_data)
                print("Game state loaded.")
            elif event.key == pygame.K_d:  # Load specific state
                try:
                    choice: str = input("Load with custom filename (y/n)?  ").lower()
                    if choice == 'y':
                        filename: str = input("Filename:  ")
                    else:
                        filename: str = 'save-state-' + input("Save State #:  ")
                    env_data: ALEState = load_specific_state(filename)
                    ale.restoreState(env_data)
                    print(f"Game state loaded from {filename}")
                except (ValueError, TypeError) as e:
                    print(f"Unable to load state. Error: {e}")
            elif event.key == pygame.K_a:  # Save specific state
                try:
                    env_data: ALEState = ale.cloneState()
                    filename: str = input("Save to file:  ")
                    save_specific_state(env_data, filename)
                    print(f"Game state saved to {filename}")
                except (ValueError, TypeError) as e:
                    print(f"Unable to save state. Error: {e}")

        pygame.display.update()
        clock.tick(fps)
        return 0, ale

    for key_combo in keys_to_action:
        if all(pressed_keys[key] for key in key_combo) and len(active_keys) == len(key_combo):
            action = keys_to_action[key_combo]
            clock.tick(fps)
            return action, ale  # Return the action without closing the window

    return 0, ale


def terminate(incentive, show_death_message=False, death_message="Dead", punishment=100) -> tuple[bool, float]:
    """
    Terminates/kills an agent playing a game (e.g. Montezuma's Revenge) and gives a punishment for that
    :param incentive: The current incentive given to the agent
    :param show_death_message: Whether to print a death message to the console when the agent's process terminates
    :param death_message: The message to print to the console when the agent's process terminates
    :param punishment: The punishment given to the agent for being terminated before its time ends
    :return: A tuple containing the 'end' boolean, which terminates the agent's process and the new incentive
    """
    if show_death_message:
        print(f'\n{death_message}')
    incentive -= punishment
    end: bool = True
    return end, incentive


def add_incentive(ram, last_life: bool, last_action: int, death_clock: int, show_death_message: bool = False,
                  give_incentive: bool = True, i: int = 0) -> tuple[float, bool, bool, int]:
    """
    Takes in the game state and adds an incentive to the environment reward. This function also kills/terminates the
    agent's process if it stalls for more than 5 seconds or dies on its last life.
    :param i: run_frames iteration
    :param ram: The RAM of the game environment
    :param last_life: Whether the agent is on its last life
    :param last_action: The last action the agent took
    :param death_clock: The number of frames the agent has taken the 'NOOP' action
    :param show_death_message: Whether to print a death message to the console when the agent dies
    :param give_incentive: Whether to give the agent an incentive in addition to its reward
    :return: A typle containing: the new incentive for the agent, whether the agent is on its last life, the 'end'
    boolean that dictates whther to terminate the agent, and the amount of frames the agent has taken the 'NOOP' action
    """
    incentive: float = 0
    end: bool = False
    if not give_incentive:
        return incentive, last_life, end, death_clock

    if last_action == 0:
        death_clock += 1
    else:
        death_clock = 0
    if death_clock > 60 * 6:
        incentive, end = terminate(incentive, show_death_message, 'Dead - Stalling', 200)

    match ram[58]:
        case 0:
            if ram[55] == 0:
                last_life = True
            if ram[55] > 0 and last_life:
                incentive, end = terminate(incentive, show_death_message, 'Dead - Last Life', 15)
        case _:
            incentive += ram[58] * .001

    if ram[3] != 7 and ram[3] != 1:
        end, incentive = terminate(incentive, show_death_message, death_message=f'Dead - Wrong Screen ({ram[3]})',
                                   punishment=200)

    if ram[3] == 7 and last_action not in {0, 1, 2, 5}:
        incentive += max(0, (.1 * (ram[42] / 255) ** 2 - (i * .00005)))
    return incentive, last_life, end, death_clock


def run_frames(frames=60 * 30, info=False, frames_per_step=1, game='MontezumaRevenge', suppress=True,
               visualize=False, show_death_message=False, load_state=None) -> float:
    """
    Runs a given game for a specified number of frames based on user input
    :param frames: Number of frames/steps to run the game for
    :param info: Whether to print the total reward over time to the console
    :param frames_per_step: The number of frames the user's chosen action is applied to before it gets to decide again
    :param game: The name of the game the agent is playing (e.g. 'MontezumaRevenge')
    :param suppress: Whether to suppress the ALE initialization text in the console (may not work)
    :param visualize: Whether the agent's gameplay will be shown to the user in a seperate window
    :param show_death_message: Whether to print the cause of death to the console
    :param load_state: File to load a save state from
    :return: The total reward over all steps the user recieved
    """
    global previous_ram
    previous_ram = None
    ale: ALEInterface = ale_init(game, suppress=suppress, visualize=visualize, load_state=load_state)
    reward: float = 0
    last_action: int = 0
    last_life: bool = False
    death_clock: int = 0
    ram_changes: defaultdict[Any, np.ndarray[Any, np.dtype]] = defaultdict(lambda: np.zeros(128, dtype=int))
    action_counts: defaultdict[Any, int] = defaultdict(int)  # Track the number of times each action was pressed

    for i in range(frames):
        inputs = ale.getRAM().reshape(1, -1)[0]
        temp_tuple = add_incentive(inputs, last_life, last_action, death_clock, show_death_message, False, i)
        incentive, last_life, end, death_clock = temp_tuple
        reward += incentive

        if end:
            break

        if i % frames_per_step == 0:
            action_index, ale = human_input(ale)
            reward += take_action(action_index, ale)
            last_action = action_index
        else:
            reward += take_action(last_action, ale)

        action_counts, ram_changes = display_ram_info(last_action, ale, action_counts, ram_changes)

    if info:
        print(f'Total Reward: {reward}')
    return reward


def main() -> None:
    """
    The main function of the program
    :return: None
    """
    choice = input(
        "Test NEAT (N/n), play game (G/g), or test best agent (A/a), or visualize a generation(V/v)?  ").lower()
    t0 = time.perf_counter()
    if choice == 'n':
        test_neat()
    elif choice == 'g':
        choice = input("Gym environment (G/g) or same as agents(A/a)?  ").lower()
        t0 = time.perf_counter()
        if choice == 'g':
            play_game(game='ALE/MontezumaRevenge-ram-v5')
        elif choice == 'a':
            run_frames(frames=60 ** 3, info=True, frames_per_step=1, visualize=True, show_death_message=True)
        else:
            print("Invalid choice. Try again.")
    elif choice == 'a':
        test_expert_agent(subtask=input('Subtask:  '))
    elif choice == 'v':
        subtask = input("Subtask:  ")
        subtask_scenarios: dict = subtask_dict[subtask]
        base_filename = 'successful-genome'
        base_filename = f'{base_filename}-{subtask}'
        checkpoint_name = f'neat-checkpoint-{subtask}'
        expert_config: str = 'config-feedforward-expert'
        successful_genomes = []  # list(set(load_specific_state(file) for file in find_all_files(base_filename)))
        run_neat(expert_config, eval_func=game_eval, checkpoints=False, checkpoint_interval=1, generations=1,
                 checkpoint=find_most_recent_file(f'neat-checkpoint-{subtask}'), insert_genomes=False,
                 genomes=successful_genomes, base_filename=base_filename, base_checkpoint_filename=checkpoint_name,
                 extra_inputs=[{'visualize': False, 'subtask': subtask, 'info': False,
                                'subtask_scenarios': subtask_scenarios}, ExpertAgent])
    else:
        print("Invalid choice. Try again.")
    t1 = time.perf_counter()
    te = t1 - t0
    print(f'Time: {te:.2f}s')


if __name__ == "__main__":
    print("Program Started")
    main()
