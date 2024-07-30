from typing import Any

from agent import Agent, test_agent
from expert_agent import ExpertAgent
from utils import run_neat_model, divide_list, average_elements_at_indexes


class MasterAgent(Agent):
    def __init__(self,
                 *args: Any,
                 expert_agents: list[ExpertAgent],
                 useless_action_set: Any | None = None,
                 **kwargs: Any) -> None:
        if useless_action_set is None:
            useless_action_set = {0}
        self.expert_agents: list[ExpertAgent] = expert_agents
        super().__init__(*args, useless_action_set, **kwargs)

    def run(self) -> None:
        initial_outputs = run_neat_model(self.net, self.inputs)
        individual_inputs: list[list[float]] = divide_list(initial_outputs, len(self.expert_agents))
        for i, expert_agent in enumerate(self.expert_agents):
            inputs = [*self.inputs, *individual_inputs[i]]
            expert_agent.set_inputs(inputs)
            expert_agent.run()
        individual_outputs = [expert_agent.get_outputs() for expert_agent in self.expert_agents]
        self.outputs = average_elements_at_indexes(individual_outputs)


def test_master_agent():
    test_agent(MasterAgent)


def main() -> None:
    test_master_agent()


if __name__ == "__main__":
    main()
