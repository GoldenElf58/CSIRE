from typing import Any

from agent import Agent, test_agent
from subtask_dictionary import subtask_dict
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
            useless_action_set = {0}
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
        self.rewards: list[float] = []
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
        subtask_goal = self.subtask_scenarios[self.load_state]['subtask_goals'][self.goal_index - 1]
        if len(subtask_goal) == 3:
            if subtask_goal[2] == room_number:
                distance_to_goal = distance(self.x, self.y, self.x_goal, self.y_goal)
            else:
                distance_to_goal = 215
        else:
            distance_to_goal = distance(self.x, self.y, self.x_goal, self.y_goal)

        if room_number not in self.room_set and self.i > 0:
            self.terminate(death_message=f'Dead - Wrong Screen ({room_number})', punishment=-200)

        if room_number in self.room_set and self.last_action not in self.useless_action_set:
            self.incentive += ((1 - (distance_to_goal / 215)) * 0.1) ** 2

        if distance_to_goal <= 5 and self.i > 0:
            if self.set_goal() == 0:
                self.incentive += 20 * (1 - self.i / self.frames)
            else:
                self.terminate(death_message=f'Dead - Reached Final Goal',
                               punishment=20 + 20 * (1 - self.i / self.frames))

        if not self.give_incentive:
            self.incentive = 0
        self.reward += self.incentive

    def set_goal(self):
        """Sets the x and y goal for the agent
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
        self.rewards: list[float] = []
        for self.load_state in self.load_states:
            self.end = False
            self.goal_index = 0
            self.last_life = False
            self.death_clock = 0
            goal_result, room_result = self.set_goal(), self.set_room_set()
            if goal_result == -2 or room_result == -2:
                raise RuntimeError(f"Goal Result was: {goal_result}, Room Result was: {room_result}")
            elif goal_result == -1 or room_result == -1:
                continue
            self.load_new_state()
            self.run_frames()
            if self.info:
                print(f'Load State: {self.load_state}')
                print(f'Subtask Scenarios: {self.subtask_scenarios}')
                print(f'Goal Index: {self.goal_index}')
                print(f'Goal: {self.x_goal, self.y_goal}')
            self.rewards.append(self.reward)
            self.reward = 0
        self.rewards.sort()
        self.reward = sum(self.rewards) + self.rewards[0] - self.rewards[-1] / 2  # Punishes agent more for worse runs
        return self.reward, self.index


def test_expert_agent(subtask='beam', config_name='config-feedforward-expert', kwargs=None) -> None:
    """Tests an ExpertAgent

    :param subtask: The subtask for the agent to be trained on
    :param kwargs: Additional controls/parameters for the ExpertAgent
    :return: None
    """
    if kwargs is None:
        subtask_scenarios: dict = subtask_dict[subtask]
        kwargs = {'subtask_scenarios': subtask_scenarios, 'subtask': subtask}
    test_agent(ExpertAgent, config_name=config_name, kwargs=kwargs)


def main() -> None:
    test_expert_agent()


if __name__ == "__main__":
    main()
