from typing import Any

from agent import Agent, test_agent
from utils import find_all_files, find_most_recent_file


class ExpertAgent(Agent):
    def __init__(self,
                 *args: Any,
                 useless_action_set: set | None = None,
                 subtask: str = 'beam',
                 additional_inputs: int = 5,
                 **kwargs: Any) -> None:
        if useless_action_set is None:
            useless_action_set = {0, 1, 2, 5}
        self.subtask: str = subtask
        self.additional_inputs = additional_inputs
        self.load_states: list = find_all_files(subtask)
        super().__init__(*args, useless_action_set=useless_action_set, load_state=self.load_states[0], **kwargs)

    def run(self) -> None:
        self.inputs = [*self.inputs, *[0 for _ in range(self.additional_inputs)]]
        super().run()

    def add_incentive(self) -> None:
        super().add_incentive()
        self.incentive: float = 0
        room_number: int = self.ram[3]

        if self.subtask == 'beam':
            if room_number != 7 and room_number != 1:
                self.terminate(death_message=f'Dead - Wrong Screen ({room_number})', punishment=200)

            if room_number == 7 and self.last_action not in self.useless_action_set:
                self.incentive += max(0, (.2 * (self.ram[42] / 255) ** 2 - (self.i / self.frames * .05))) * 2

        if not self.give_incentive:
            self.incentive = 0
        self.reward += self.incentive

    def test_agent(self) -> tuple[float, int]:
        self.reward: float = 0
        for self.load_state in self.load_states:
            self.run_frames()
        return self.reward, self.index


def test_expert_agent():
    test_agent(ExpertAgent)


def main() -> None:
    test_expert_agent()


if __name__ == "__main__":
    main()
