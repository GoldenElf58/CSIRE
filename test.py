import os
import subprocess
from ale_py import ALEInterface, roms
import gymnasium as gym
from gym.utils.play import play
import pygame


def play_game():
    env = gym.make('MontezumaRevenge-v4', render_mode='rgb_array')

    # Define the keys to actions mapping
    keys_to_action = {
        (pygame.K_UP,): 2,
        (pygame.K_RIGHT,): 3,
        (pygame.K_LEFT,): 4,
        (pygame.K_DOWN,): 5,
        (pygame.K_UP, pygame.K_RIGHT): 6,
        (pygame.K_UP, pygame.K_LEFT): 7,
        (pygame.K_DOWN, pygame.K_RIGHT): 8,
        (pygame.K_DOWN, pygame.K_LEFT): 9,
    }

    play(env, keys_to_action=keys_to_action)


def load_rom_suppressed(game):
    rom_path = os.path.join('C:\\Users\\Owen\\PycharmProjects\\CSIRE\\.venv\\Lib\\site-packages\\AutoROM\\roms',
                            f"{game}.bin")

    # Ensure the ROM file exists
    if not os.path.exists(rom_path):
        raise FileNotFoundError(f"ROM file not found: {rom_path}")

    # Create a temporary script to load the ROM
    script_content = f"""
import os
from ale_py import ALEInterface

def load_rom():
    ale = ALEInterface()
    ale.loadROM(r'{rom_path}')

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
        raise RuntimeError(f"Failed to load ROM: {result.stderr}")


def ale_init(game, suppress=True):
    ale = ALEInterface()
    if suppress:
        load_rom_suppressed(game)
    else:
        rom = getattr(roms, game)
        ale.loadROM(rom)

    return ale


def main():
    roms_list = dir(roms)
    print(roms_list)
    ale_init("montezuma_revenge")
    print("Game initialized successfully!")


if __name__ == "__main__":
    main()
