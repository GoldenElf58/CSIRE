from typing import Any

from neat import Config, DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation

from agent import Agent, test_agent
from expert_agent import ExpertAgent
from subtask_dictionary import subtask_dict
from utils import run_neat_model, divide_list, average_elements_at_indexes, normalize_list, load_latest_state


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
        super().__init__(genome, *args, useless_action_set=useless_action_set, **kwargs)

    def run(self) -> None:
        initial_outputs = run_neat_model(self.net, self.inputs)
        assert len(initial_outputs) == 49
        individual_inputs: list[list[float]] = divide_list(initial_outputs[0:47], len(self.expert_agents) + 1)
        individual_inputs[-1].append(initial_outputs[-1])
        for i, expert_agent in enumerate(self.expert_agents):
            inputs = [*self.inputs, *individual_inputs[i][:7]]  # RAM inputs and master agent inputs
            expert_agent.set_inputs(inputs)
            expert_agent.run_with_current_inputs()
        individual_outputs = [normalize_list(expert_agent.get_outputs(), individual_inputs[i][7]) for i, expert_agent in
                              enumerate(self.expert_agents)]
        individual_outputs.append(normalize_list(individual_inputs[-1][:8], individual_outputs[-1][7]))
        self.outputs = average_elements_at_indexes(individual_outputs)


def test_master_agent():
    expert_genomes = []
    for subtask in subtask_dict.keys():
        expert_genomes.append(load_latest_state(f'successful-genome-{subtask}'))
    expert_config_name = 'config-feedforward-expert'
    kwargs = {'expert_config_name': expert_config_name, 'expert_genomes': expert_genomes}
    test_agent(MasterAgent, subtask='master', kwargs=kwargs, config_name='config-feedforward-master')


def main() -> None:
    test_master_agent()


if __name__ == "__main__":
    main()
