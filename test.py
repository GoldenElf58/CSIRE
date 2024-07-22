from ale_py import ALEInterface, roms
from collections import defaultdict
import gym
from gym.utils.play import play
import neat
import numpy as np
import os
import pygame
import subprocess
import sys


# ANSI escape codes for colors
RESET_COLOR = "\033[0m"
RED_COLOR = "\033[91m"
YELLOW_COLOR = "\033[93m"
GREEN_COLOR = "\033[92m"
CYAN_COLOR = '\033[96m'


def play_game(game='MontezumaRevenge-v4'):
    env = gym.make(game, render_mode='rgb_array')

    # Define the keys to actions mapping
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

    def callback(obs_t, obs_tp1, action, rew, done, info, keys):
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
        noop_frequently_changing = ram_changes[0] / action_counts[0] > 0.25 if action_counts[0] > 0 else np.zeros_like(frequently_changing)

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
    # Byte 26 is the door sprite and blue thingys, 117 is off, 124 is door, 163-165 is blue on animation
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


def main():
    play_game()


if __name__ == "__main__":
    print("Program Started")
    main()
