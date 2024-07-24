import time

from ale_py import ALEInterface, roms, Action
from collections import defaultdict
import cv2
import gym
from gym.utils.play import play
import neat
import numpy as np
import os
import pygame
import subprocess
import sys

from neuroevolution import test_neat

# ANSI escape codes for colors
RESET_COLOR = "\033[0m"
RED_COLOR = "\033[91m"
YELLOW_COLOR = "\033[93m"
GREEN_COLOR = "\033[92m"
CYAN_COLOR = '\033[96m'


def play_game(game='MontezumaRevenge-v4'):
    env = gym.make(game, render_mode='rgb_array')
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
    ale_env = env.unwrapped.ale  # Access the ALE environment
    previous_ram = None
    ram_changes = defaultdict(lambda: np.zeros(128, dtype=int))  # Track changes per action
    action_counts = defaultdict(int)  # Track the number of times each action was pressed
    
    def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        nonlocal previous_ram
        
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
        
        sys.stdout.write('\r' + ram_str.ljust(8 * len(ram)))
        sys.stdout.flush()
    
    # Bytes 42 and 43 are x and y pos
    # Byte 26 is the door sprite and blue thingies, 117 is off, 124 is door, 163-165 is blue on animation
    # Byte 30 is the ladder climb sprite, 0 if not on ladder, 62 or 82 on ladder
    # Byte 32 - 21 is not on ladder, 102 is on ladder
    # Byte 89 is last movement when moving, else 0
    # Byte 90 - 18 is on ladder, 19 is off ladder
    # Byte 91 is the level/screen/room?
    
    play(env, keys_to_action=keys_to_action, callback=callback)


def load_rom_suppressed(game) -> None:
    # Create a temporary script to load the ROM
    script_content = f"""
import os
from ale_py import ALEInterface, roms

def load_rom():
    ale = ALEInterface()
    available_roms = roms.get_all_rom_ids()
    if '{game}' in available_roms:
        rom_path = roms.get_rom_path('{game}')
        ale.loadROM(rom_path)
    else:
        raise ValueError(f'ROM for game {game} not supported.\\nSupported ROMs: {{available_roms}}')

if __name__ == "__main__":
    load_rom()
"""
    
    script_path = "temp_script.py"
    with open(script_path, "w") as script_file:
        script_file.write(script_content)
    
    # Run the temporary script in a subprocess
    try:
        result = subprocess.run(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    finally:
        os.remove(script_path)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to load ROM: {result.stderr}\nReturn Code: {result.returncode}")
    else:
        print(result.stdout)


def convert_game_name(game_name, to_camel_case=True) -> str:
    if to_camel_case:
        if '_' not in game_name: return game_name
        words: list[str] = game_name.split('_')
        capitalized_words: list[str] = [word.capitalize() for word in words]
        return ''.join(capitalized_words)
    else:
        converted_name: str = ''.join(['_' + i.lower() if i.isupper() else i for i in game_name]).lstrip('_')
        return converted_name


def ale_init(game, suppress=False) -> ALEInterface:
    ale: ALEInterface = ALEInterface()
    
    if suppress:
        game = convert_game_name(game, False)
        load_rom_suppressed(game)
    else:
        game = convert_game_name(game, True)
        rom = getattr(roms, game)
        ale.loadROM(rom)
    return ale


def human_input(fps=60) -> int:
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
    screen = pygame.display.set_mode((100, 100))
    pygame.display.set_caption("Human Input Capture")
    
    clock = pygame.time.Clock()  # Create a clock object to control the frame rate
    
    while True:
        pressed_keys = pygame.key.get_pressed()
        active_keys = tuple(key for key, pressed in enumerate(pressed_keys) if pressed)
        
        for key_combo in keys_to_action:
            if all(pressed_keys[key] for key in key_combo) and len(active_keys) == len(key_combo):
                action = keys_to_action[key_combo]
                clock.tick(fps)
                return action  # Return the action without closing the window
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return 0  # Exit function if window is closed
        
        clock.tick(fps)
        pygame.display.update()
        return 0


def take_action(output: int, ale: ALEInterface) -> int:
    # Take an action and get the new state
    legal_actions: list[Action] = ale.getLegalActionSet()
    action = legal_actions[output]  # Choose an action (e.g., NOOP)
    reward: int = ale.act(action)
    return reward


def terminate(incentive, death_message="Dead") -> tuple[bool, float]:
    print(death_message)
    incentive -= 100
    end: bool = True
    return end, incentive


def add_incentive(ram, last_life, last_action, death_clock) -> tuple[float, bool, bool, int]:
    incentive: float = 0
    end: bool = False
    
    if last_action == 0:
        death_clock += 1
    else:
        death_clock = 0
    if death_clock > 60 * 60: incentive, end = terminate(incentive, death_message='Dead - Stalling')
    
    match ram[58]:
        case 0:
            if ram[55] == 0: last_life = True
            if ram[55] > 0 and last_life: incentive, end = terminate(incentive, death_message='Dead - Last Life')
        case _:
            incentive += ram[58] * .001
    
    match ram[66]:
        case 13:
            incentive += 0.3
        case 12:
            incentive += 0.6
        case 14:
            incentive += 0.9
    
    incentive -= .0001 * ram[43]
    return incentive, last_life, end, death_clock


def run_frames(frames=100, info=False, frames_per_step=1, game='MontezumaRevenge', suppress=False,
               display_frames=False) -> float:
    ale: ALEInterface = ale_init(game, suppress)
    reward: float = 0
    last_action: int = 0
    last_life: bool = False
    death_clock: int = 0
    
    for i in range(frames):
        inputs = ale.getRAM().reshape(1, -1)[0]
        incentive, last_life, end, death_clock = add_incentive(inputs, last_life, last_action, death_clock)
        if end: break
        reward += incentive
        
        if i % frames_per_step == 0:
            output = human_input()
            reward += take_action(output, ale)
            last_action = output
        else:
            reward += take_action(last_action, ale)
        
        if display_frames:
            cv2.imshow('Image', ale.getScreenRGB())
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    if display_frames: cv2.destroyAllWindows()
    if info: print(f'Total Reward: {reward}')
    return reward


def main():
    choice = input("Test NEAT (N/n) or play game (G/g)?  ").lower()
    t0 = time.perf_counter()
    if choice == 'n':
        test_neat()
    elif choice == 'g':
        choice = input("Gym environment (G/g) or same as agents(A/a)?  ").lower()
        t0 = time.perf_counter()
        if choice == 'g': play_game()
        elif choice == 'a': run_frames(frames=60*60, info=True, frames_per_step=2, display_frames=True)
        else: print("Invalid choice. Try again.")
    else:
        print("Invalid choice. Try again.")
    t1 = time.perf_counter()
    te = t1 - t0
    print(f'Time: {te:.2f}s')


if __name__ == "__main__":
    print("Program Started")
    main()
