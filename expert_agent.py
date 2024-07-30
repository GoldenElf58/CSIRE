from typing import Any

from agent import Agent, test_agent


class ExpertAgent(Agent):
    def __init__(self,
                 *args: Any,
                 useless_action_set: Any | None = None,
                 **kwargs: Any) -> None:
        if useless_action_set is None:
            useless_action_set = {0, 1, 2, 5}
        super().__init__(*args, useless_action_set, **kwargs)

    def run(self) -> None:
        self.inputs = [*self.inputs, *[0 for _ in range(5)]]
        super().run()

    def add_incentive(self) -> None:
        super().add_incentive()
        self.incentive: float = 0
        room_number: int = self.ram[3]

        if room_number != 7 and room_number != 1:
            self.terminate(death_message=f'Dead - Wrong Screen ({room_number})', punishment=200)

        if room_number == 7 and self.last_action not in self.useless_action_set:
            self.incentive += max(0, (.2 * (self.ram[42] / 255) ** 2 - (self.i / self.frames * .05))) * 2

        if not self.give_incentive:
            self.incentive = 0
        self.reward += self.incentive


def test_expert_agent():
    test_agent(ExpertAgent)


def main() -> None:
    test_expert_agent()


if __name__ == "__main__":
    main()
