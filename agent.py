from ale_py import ALEInterface, ALEState, LoggerMode, roms

from utils import convert_game_name, get_action_index, load_specific_state, run_neat_model, take_action


class Agent:
    def __init__(self, net, frames=100, info=False, frames_per_step=1, game='MontezumaRevenge', suppress=True,
                 visualize=False, show_death_message=False, seed=123, frame_skip=0, repeat_action_probability=0,
                 load_state=None):
        self.net = net
        self.frames = frames
        self.info = info
        self.frames_per_step = frames_per_step
        self.game = game
        self.suppress = suppress
        self.visualize = visualize
        self.show_death_message = show_death_message
        self.seed = seed
        self.frame_skip = frame_skip
        self.repeat_action_probability = repeat_action_probability
        self.load_state = load_state
        self.ram = None
        self.inputs = None
        self.outputs = None
        self.incentive: float = 0
        self.end: bool = False
        self.last_life: bool = False
        self.last_action: int = 0
        self.death_clock: int = 0
        self.reward: float = 0
        self.give_incentive: bool = True
        self.ale = self.ale_init()

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
        agent's process if it stalls for more than 5 seconds or dies on its last life.
        :return: None
        """
        lives = self.ram[58]
        death_scene_countdown = self.ram[55]
        room_number = self.ram[3]

        if self.last_action == 0:
            self.death_clock += 1
        else:
            self.death_clock = 0
        if self.death_clock > 60 * 6:
            self.terminate(death_message='Dead - Stalling', punishment=200)

        match lives:
            case 0:
                if death_scene_countdown == 0:
                    self.last_life = True
                if death_scene_countdown > 0 and self.last_life:
                    self.terminate(death_message='Dead - Last Life', punishment=15)
            case _:
                self.incentive += lives * .001

        if room_number != 7 and room_number != 1:
            self.terminate(death_message=f'Dead - Wrong Screen ({room_number})', punishment=200)

        if room_number == 7 and self.last_action not in {0, 1, 2, 5}:
            self.incentive += .1 * (self.ram[42] / 255) ** 2
        if not self.give_incentive:
            self.incentive = 0

    def run_frames(self):
        """
        A function that lets an agent play a given game for a given number of steps.
        :return: The total reward over all steps the agent recieved
        """

        for i in range(self.frames):
            self.ram = self.ale.getRAM().reshape(1, -1)[0]
            self.inputs = self.ram
            self.add_incentive()
            self.reward += self.incentive
            if self.end:
                break

            if i % self.frames_per_step == 0:
                self.run()
                action_index: int = get_action_index(self.outputs)
                self.reward += take_action(action_index, self.ale)
                self.last_action = action_index
            else:
                self.reward += take_action(self.last_action, self.ale)

        if self.info:
            print(f'Total Reward: {self.reward}')
        return self.reward
