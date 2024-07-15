import gym
import pygame
from gym.utils.play import play


def main():
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

    play(env, keys_to_action)


if __name__ == "__main__":
    main()