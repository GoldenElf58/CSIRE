from typing import Any

from agent import Agent, test_agent
from utils import find_all_files, distance


class ExpertAgent(Agent):
    def __init__(self,
                 *args: Any,
                 useless_action_set: set | None = None,
                 room_set: set | None = None,
                 subtask: str = 'beam',
                 subtask_scenarios: dict[str, dict[str, list[list[int]] | set[int]]] | None = None,
                 additional_inputs: int = 5,
                 **kwargs: Any) -> None:
        if useless_action_set is None:
            useless_action_set = {0, 1, 2, 5}
        if room_set is None:
            room_set = {7}
        if subtask_scenarios is None:
            subtask_scenarios = {
                'beam-0': {
                    'room_set': {7, 13},
                    'subtask_goals': [[130, 252], [77, 134]]
                }, 'beam-1': {
                    'room_set': {11, 12},
                    'subtask_goals': [[0, 235]]
                }, 'beam-2': {
                    'room_set': {12, 13},
                    'subtask_goals': [[152, 235]]
                }, 'beam-3': {
                    'room_set': {0, 4},
                    'subtask_goals': [[25, 252], [77, 134]]
                }
            }

        self.load_states: list = find_all_files(subtask)
        self.load_state: str = self.load_states[0]
        self.room_set: set = room_set
        self.subtask: str = subtask
        self.subtask_scenarios: dict[str, dict[str, list[list[int]] | set[int]]] = subtask_scenarios
        self.scenario_goals: list[list[int]] = self.subtask_scenarios[self.load_state]['subtask_goals']
        self.goal_index: int = 0
        self.x_goal: int = 0
        self.y_goal: int = 0
        self.additional_inputs: int = additional_inputs
        super().__init__(*args, useless_action_set=useless_action_set, load_state=self.load_state, **kwargs)

    def run(self) -> None:
        self.inputs = [*self.inputs, self.x_goal, self.y_goal, *[0 for _ in range(self.additional_inputs)]]
        super().run()

    def run_with_current_inputs(self) -> None:
        super().run()

    def add_incentive(self) -> None:
        super().add_incentive()
        self.incentive: float = 0
        room_number: int = self.ram[3]
        x: int = self.ram[42]
        y: int = self.ram[43]
        distance_to_goal = distance(x, y, self.x_goal, self.y_goal)

        if self.subtask == 'beam':
            if room_number not in self.room_set and self.i > 0:
                self.terminate(death_message=f'Dead - Wrong Screen ({room_number})', punishment=-200)

            if room_number in self.room_set and self.last_action not in self.useless_action_set:
                self.incentive += ((1 - (distance_to_goal / 300)) * 0.5) ** 2

        if distance_to_goal < 5 and self.i > 0:
            if self.set_goal() == 0:
                self.incentive += 20
            else:
                self.terminate(death_message=f'Dead - Reached Goal', punishment=20)

        if not self.give_incentive:
            self.incentive = 0
        self.reward += self.incentive

    def set_goal(self):
        """
        Sets the x and y goal for the agent
        :return: None
        """
        self.scenario_goals = self.subtask_scenarios[self.load_state]['subtask_goals']
        if self.load_state not in self.subtask_scenarios.keys():
            return -1  # load state not documented
        if self.goal_index > len(self.scenario_goals) - 1:
            return -2  # goal index too high
        self.x_goal = self.scenario_goals[self.goal_index][0]
        self.y_goal = self.scenario_goals[self.goal_index][1]
        self.goal_index += 1
        return 0

    def set_room_set(self):
        if self.load_state not in self.subtask_scenarios.keys():
            return -1  # load state not documented
        self.room_set = self.subtask_scenarios[self.load_state]['room_set']

    def test_agent(self) -> tuple[float, int]:
        self.ale_init()
        self.reward: float = 0
        for self.load_state in self.load_states:
            self.end = False
            self.goal_index = 0
            self.last_life = False
            if self.set_goal() == -1 or self.set_room_set() == -1:
                continue
            self.load_new_state()
            self.run_frames()
        return self.reward, self.index


def test_expert_agent(kwargs=None) -> None:
    """
    Tests an ExpertAgent
    :param kwargs: Additional controls/parameters for the ExpertAgent
    :return: None
    """
    if kwargs is None:
        subtask_scenarios: dict = {
            'beam-0': {
                'room_set': {7, 13},
                'subtask_goals': [[130, 252], [77, 134]]
            }, 'beam-1': {
                'room_set': {11, 12},
                'subtask_goals': [[0, 235]]
            }, 'beam-2': {
                'room_set': {12, 13},
                'subtask_goals': [[152, 235]]
            }, 'beam-3': {
                'room_set': {0, 4},
                'subtask_goals': [[25, 252], [77, 134]]
            }
        }
        kwargs = {'subtask_scenarios': subtask_scenarios}
    test_agent(ExpertAgent, kwargs=kwargs)


def main() -> None:
    test_expert_agent()


if __name__ == "__main__":
    main()
