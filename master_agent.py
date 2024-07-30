from typing import Any

from neat import Config, DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation

from agent import Agent, test_agent
from expert_agent import ExpertAgent
from utils import run_neat_model, divide_list, average_elements_at_indexes


class MasterAgent(Agent):
    def __init__(self,
                 genome: DefaultGenome,
                 *args: Any,
                 expert_genomes: list[DefaultGenome],
                 expert_config_name: str,
                 useless_action_set: Any | None = None,
                 **kwargs: Any) -> None:
        if useless_action_set is None:
            useless_action_set = {0}
        expert_config: Config = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet,
                                       DefaultStagnation, expert_config_name)

        self.expert_agents: list[ExpertAgent] = [ExpertAgent(expert_genome, expert_config, 0) for expert_genome in
                                                 expert_genomes]
        super().__init__(genome, *args, useless_action_set, **kwargs)

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
