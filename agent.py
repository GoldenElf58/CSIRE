from typing import Callable

import neat
import numpy as np
import torch
from ale_py import ALEInterface, ALEState, LoggerMode, roms
from logs import logger
from neat import Config, DefaultGenome

from autoencoder import Autoencoder
from utils import (convert_game_name, get_action_index, load_specific_state, run_neat_model, take_action,
                   find_most_recent_file, normalize_list)


class Agent:
    def __init__(self,
                 genome: DefaultGenome,
                 config: Config,
                 index: int,
                 frames: int = 60 * 30,
                 info: bool = False,
                 frames_per_step: int = 1,
                 game: str = 'MontezumaRevenge',
                 suppress: bool = True,
                 visualize: bool = False,
                 show_death_message: bool = False,
                 seed: int = 123,
                 frame_skip: int = 1,
                 repeat_action_probability: int = 0,
                 load_state: str | None = None,
                 stall_length: int = 60 * 6,
                 stall_punishment: int = 100,
                 give_incentive: bool = True,
                 useless_action_set: set | None = None,
                 use_autoencoder: bool = False,
                 autoencoder: Autoencoder | None = None) -> None:
        if useless_action_set is None:
            useless_action_set: set[int] = {0}
        self.index: int = index
        self.genome: DefaultGenome = genome
        self.config: Config = config
        self.net: neat.nn.FeedForwardNetwork = neat.nn.FeedForwardNetwork.create(genome, config)
        self.frames: int = frames
        self.info: bool = info
        self.frames_per_step: int = frames_per_step
        self.game: str = game
        self.suppress: bool = suppress
        self.visualize: bool = visualize
        self.show_death_message: bool = show_death_message
        self.seed: int = seed
        self.frame_skip: int = frame_skip
        self.repeat_action_probability: float = repeat_action_probability
        self.load_state: str = load_state
        self.stall_length: float = stall_length
        self.stall_punishment: float = stall_punishment
        self.give_incentive: bool = give_incentive
        self.useless_action_set: set[int] = useless_action_set
        self.use_autoencoder: bool = use_autoencoder
        self.autoencoder: Autoencoder | None = autoencoder
        self.x: int = 0
        self.y: int = 0
        self.last_x: int = 0
        self.last_y: int = 0
        self.prev_y = 0
        self.i: int | None = None
        self.ram: None | list[int] = None
        self.ram_state: None | torch.FloatTensor = None
        self.inputs: None | list[int] = None
        self.outputs: None | list[float] = None
        self.incentive: float = 0
        self.end: bool = False
        self.last_life: bool = False
        self.last_action: int = 0
        self.death_clock: int = 0
        self.reward: float = 0
        self.game_reward: float = 0
        self.ale: ALEInterface | None = None

    def ale_init(self) -> None:
        """
        Loads the ALEInterface for the Agent
        :return: An ALEInterface with a game loaded
        """
        if self.suppress:
            ALEInterface.setLoggerMode(LoggerMode.Error)

        self.ale: ALEInterface = ALEInterface()

        self.ale.setFloat('repeat_action_probability', self.repeat_action_probability)
        self.ale.setBool('display_screen', self.visualize)
        self.ale.setInt('frame_skip', self.frame_skip)
        if self.seed is not None:
            self.ale.setInt('random_seed', self.seed)

        game = convert_game_name(self.game, True)
        rom = getattr(roms, game)
        self.ale.loadROM(rom)
        self.load_new_state()  # Will only load state if load_state is not None

    def load_new_state(self) -> None:
        """
        Loads the new state based on the load_state variable
        :return: None
        """
        if self.load_state is not None:
            env_data: ALEState = load_specific_state(self.load_state)
            self.ale.restoreState(env_data)
            if not self.suppress:
                logger.info(f"Game state loaded from {self.load_state}")

    def set_inputs(self, inputs) -> None:
        """Sets the inputs and normalizes them

        :param inputs: The inputs to be set to
        :return: None
        """
        self.inputs = normalize_list(inputs, 1/255)
        if self.use_autoencoder:
            if self.autoencoder is None:
                logger.warning("Autoencoder is None")
            with torch.no_grad():
                code, _ = self.autoencoder(self.ram_state)
            self.inputs = code.numpy()

    def get_outputs(self) -> list[float] | None:
        """Gets the outputs of the Agent

        :return: The outputs of the agent
        """
        return self.outputs

    def run(self) -> None:
        """Runs the neat model

        :return: None
        """
        self.outputs = run_neat_model(self.net, self.inputs)

    def terminate(self, death_message="Dead", punishment=-100) -> None:
        """Terminates/kills an agent playing a game (e.g. Montezuma's Revenge) and gives a punishment for that

        :param death_message: The message to print to the console when the agent's process terminates
        :param punishment: The punishment given to the agent for being terminated before its time ends
        :return: None
        """
        if self.show_death_message:
            logger.info(f'\n{death_message}')
        self.incentive += punishment
        self.end: bool = True

    def add_incentive(self) -> None:
        """
        Takes in the game state and adds an incentive to the environment reward. This function also kills/terminates the
        agent's process if it stalls for more than self.stall_length frames or dies on its last life.
        :return: None
        """
        self.incentive: float = 0
        lives: int = self.ram[58]
        death_scene_countdown: int = self.ram[55]
        self.x = float(self.ram[42])
        self.y = float(self.ram[43])

        if self.last_action in self.useless_action_set or (self.x == self.last_x and abs(self.y - self.last_y) < 25):
            self.death_clock += 1
        else:
            self.death_clock = 0
        if abs(self.y - self.prev_y) > 25:
            self.prev_y = self.y
        if self.death_clock >= self.stall_length:
            self.terminate(death_message='Dead - Stalling', punishment=-100)

        if death_scene_countdown == 1:
            self.incentive -= 5
            if lives == 0:
                self.last_life = True
        elif death_scene_countdown > 1 and self.last_life and lives == 0:
            self.terminate(death_message='Dead - Last Life', punishment=-20)
        self.incentive -= (.01 - lives * .001)

        if not self.give_incentive:
            self.incentive = 0
        self.last_x = self.x
        self.last_y = self.y
        self.reward += self.incentive

    def add_game_reward(self, game_reward):
        self.reward += game_reward

    def run_frames(self) -> tuple[float, int]:
        """ A function that lets an agent play a given game for a given number of steps.

        :return: A tuple containing: (The total reward over all steps the agent recieved, The genome's index)
        """
        for self.i in range(self.frames):
            self.ram = self.ale.getRAM()
            self.ram_state = np.array(self.ram, dtype=np.float32) / 255.0
            self.ram_state = torch.FloatTensor(self.ram_state)
            self.ram = self.ram.reshape(1, -1)[0]
            self.set_inputs(self.ram)
            self.add_incentive()

            if self.end:
                break

            if self.i % self.frames_per_step == 0:
                self.run()
                action_index: int = get_action_index(self.outputs)
                game_reward: float = take_action(action_index, self.ale)
                self.last_action = action_index
            else:
                game_reward: float = take_action(self.last_action, self.ale)

            self.game_reward += game_reward
            self.add_game_reward(game_reward)

        return self.reward, self.index

    def display_info(self):
        """Displays information about the testing if self.info is True

        :return: None
        """
        if self.info:
            logger.info(f'Index: {self.index}')
            logger.info(f'Load State: {self.load_state}')
            logger.info(f'Total Reward: {self.reward:.2f}')
            logger.info(f'Total Game Reward: {self.game_reward}')

    def test_agent(self) -> tuple[float, int]:
        """Tests the agent

        :return: A tuple containing: (total reward, genome index)
        """
        self.ale_init()
        self.run_frames()
        self.display_info()
        return self.reward, self.index


def test_agent(agent_type: Callable = Agent, subtask='beam', kwargs: dict | None = None,
               config_name: str = "config-feedforward") -> None:
    """Tests a chosen agent or the most recent one

    :param agent_type: The type of Agent to be tested
    :param kwargs: Any additional parameters for the testing
    :return: None
    """
    if kwargs is None:
        kwargs: dict = {}
    choice = input("Specific file? (y/n)  ").lower()
    if choice == 'y':
        genome = load_specific_state(input("Load genome from:  "))
    else:
        file = find_most_recent_file(f'successful-genome-{subtask}')
        genome = load_specific_state(file)
        logger.debug(f"Genome loaded from {file}")
    visualize = input("Visualize (y/n)?  ").lower()
    visualize = visualize == 'y'
    config: Config = Config(DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_name)
    agent = agent_type(genome, config, -1, visualize=visualize, frames=60 * 30, frames_per_step=2, suppress=False,
                       show_death_message=True, info=True, **kwargs)
    agent.test_agent()


def main() -> None:
    test_agent()


if __name__ == "__main__":
    main()
