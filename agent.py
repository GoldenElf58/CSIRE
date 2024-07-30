from typing import Any, Callable

import neat
from ale_py import ALEInterface, ALEState, LoggerMode, roms
from neat import Config, DefaultGenome

from utils import (convert_game_name, get_action_index, load_specific_state, run_neat_model, take_action,
                   find_most_recent_file)


class Agent:
    def __init__(self,
                 genome: DefaultGenome,
                 config: Config,
                 index: Any,
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
                 load_state: Any = None,
                 stall_length: int = 60 * 6,
                 stall_punishment: int = 100,
                 give_incentive: bool = True,
                 useless_action_set: Any | None = None) -> None:
        if useless_action_set is None:
            useless_action_set: set[int] = {0}
        self.index: int = index
        self.genome = genome
        self.config = config
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
        self.i: int or None = None
        self.ram: None or list[int] = None
        self.inputs: None or list[int] = None
        self.outputs: None or list[float] = None
        self.incentive: float = 0
        self.end: bool = False
        self.last_life: bool = False
        self.last_action: int = 0
        self.death_clock: int = 0
        self.reward: float = 0
        self.game_reward: float = 0
        self.ale: ALEInterface or None = None

    def ale_init(self):
        """
        Loads the ALEInterface for the Agent
        :return: An ALEInterface with a game loaded
        """
        self.ale: ALEInterface = ALEInterface()

        if self.suppress:
            self.ale.setLoggerMode(LoggerMode.Error)

        self.ale.setFloat('repeat_action_probability', self.repeat_action_probability)
        self.ale.setBool('display_screen', self.visualize)
        self.ale.setInt('frame_skip', self.frame_skip)
        if self.seed is not None:
            self.ale.setInt('random_seed', self.seed)

        game = convert_game_name(self.game, True)
        rom = getattr(roms, game)
        self.ale.loadROM(rom)

        if self.load_state is not None:
            env_data: ALEState = load_specific_state(self.load_state)
            self.ale.restoreState(env_data)
            if not self.suppress:
                print(f"Game state loaded from {self.load_state}")
            return self.ale

        return self.ale

    def set_inputs(self, inputs):
        self.inputs = inputs

    def get_outputs(self) -> list[float] or None:
        return self.outputs

    def run(self) -> None:
        """
        Runs the neat model
        :return: None
        """
        self.outputs = run_neat_model(self.net, self.inputs)

    def terminate(self, death_message="Dead", punishment=100) -> None:
        """
        Terminates/kills an agent playing a game (e.g. Montezuma's Revenge) and gives a punishment for that
        :param death_message: The message to print to the console when the agent's process terminates
        :param punishment: The punishment given to the agent for being terminated before its time ends
        :return: None
        """
        if self.show_death_message:
            print(f'\n{death_message}')
        self.incentive -= punishment
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

        if self.last_action in self.useless_action_set:
            self.death_clock += 1
        else:
            self.death_clock = 0
        if self.death_clock >= self.stall_length:
            self.terminate(death_message='Dead - Stalling', punishment=100)

        match lives:
            case 0:
                if death_scene_countdown == 0:
                    self.last_life = True
                if death_scene_countdown > 0 and self.last_life:
                    self.terminate(death_message='Dead - Last Life', punishment=15)
            case _:
                self.incentive += lives * .001

        if not self.give_incentive:
            self.incentive = 0
        self.reward += self.incentive

    def run_frames(self) -> tuple[float, int]:
        """
        A function that lets an agent play a given game for a given number of steps.
        :return: A tuple containing: (The total reward over all steps the agent recieved, The genome's index)
        """
        self.ale_init()

        for self.i in range(self.frames):
            self.inputs = self.ram = self.ale.getRAM().reshape(1, -1)[0]
            self.add_incentive()

            if self.end:
                break

            if self.i % self.frames_per_step == 0:
                self.run()
                action_index: int = get_action_index(self.outputs)
                game_reward: float = take_action(action_index, self.ale)
                self.game_reward += game_reward
                self.reward += game_reward
                self.last_action = action_index
            else:
                game_reward: float = take_action(self.last_action, self.ale)
                self.game_reward += game_reward
                self.reward += game_reward

        if self.info:
            print(f'Total Reward: {self.reward}')
            print(f'Total Game Reward {self.game_reward}')
        return self.reward, self.index


def test_agent(agent_type: Callable = Agent):
    choice = input("Specific file? (y/n)  ").lower()
    if choice == 'y':
        genome = load_specific_state(input("Load genome from:  "))
    else:
        file = find_most_recent_file('successful-genome')
        genome = load_specific_state(file)
        print(f"Genome loaded from {file}")
    config: neat.config.Config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                                    "config-feedforward")
    agent = agent_type(genome, config, 0, visualize=True, frames=60 * 30, frames_per_step=2, suppress=False,
                       show_death_message=True, load_state='beam-0', info=True)
    agent.run_frames()


def main() -> None:
    test_agent()


if __name__ == "__main__":
    main()
