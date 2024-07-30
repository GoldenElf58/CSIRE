import os
from typing import Callable

from neat import DefaultGenome

from agent import Agent
from expert_agent import ExpertAgent
from master_agent import MasterAgent
from neuroevolution import run_neat
from utils import find_most_recent_file, load_specific_state, find_all_files, run_in_parallel


def game_eval(genomes, config, func_params=None, run_func=Agent.run_frames, agent_type: Callable = Agent) -> None:
    """
    The evaluation function for a set of genomes. Takes in the genomes and sets their fitness.
    :param genomes: A list of the genomes to be tested
    :param config: The configuration of the genomes
    :param func_params: The parameters to be passed to the 'run_frames' function
    :param run_func: The function to test the genomes in
    :param agent_type: The type of agent (e.g. ExpertAgent, MasterAgent, Agent
    :return: None
    """
    if func_params is None:
        func_params = {}
    args = []
    for i, (_, genome) in enumerate(genomes):
        kwargs = {'frames': 60 * 30, 'frames_per_step': 2} | func_params
        agent: Agent = agent_type(genome, config, i, **kwargs)
        args.append([agent])
    results = run_in_parallel(run_func, args=args, iterations=len(args))
    for result in results:
        genomes[result[1]][1].fitness = float(result[0])


def train_expert(subtask: str = 'beam', base_filename: str = 'successful-genome',
                 expert_config: str = 'config-feedforward-expert'):
    """
    Trains an expert for a given subtask
    :param subtask: The subtask to be trained on
    :param base_filename: The base filename before the subtask name
    :param expert_config: The name of the configuration file for the expert agents
    :return:
    """
    if subtask is not None:
        base_filename = f'{base_filename}-{subtask}'
    successful_genomes = list(set(load_specific_state(file) for file in find_all_files(base_filename)))
    successful_genome = run_neat(expert_config, eval_func=game_eval, checkpoints=True, checkpoint_interval=1,
                                 checkpoint=find_most_recent_file('neat-checkpoint'), insert_genomes=True,
                                 genomes=successful_genomes, generations=2, base_filename=base_filename,
                                 extra_inputs=[{'visualize': False, 'agent_type': ExpertAgent, 'subtask': subtask}])
    return successful_genome


def train_master(expert_agents: list[DefaultGenome], base_filename: str = 'successful-genome-master',
                 expert_config: str = 'config-feedforward-expert',
                 master_config: str = 'config-feedforward-master') -> DefaultGenome:
    successful_genomes = list(set(load_specific_state(file) for file in find_all_files(base_filename)))
    successful_genome = run_neat(master_config, eval_func=game_eval, checkpoints=True, checkpoint_interval=1,
                                 checkpoint=find_most_recent_file('neat-checkpoint'), insert_genomes=True,
                                 genomes=successful_genomes, generations=2, base_filename=base_filename,
                                 extra_inputs=[
                                     {'visualize': False, 'agent_type': MasterAgent, 'expert_agents': expert_agents,
                                      'expert_config_name': expert_config}])
    return successful_genome


def main() -> None:
    """
    The main function of the program
    :return: None
    """
    expert_config: str = 'config-feedforward-expert'
    master_config: str = 'config-feedforward-master'
    subtasks: list[str] = ['beam']
    successful_genomes: dict[str, DefaultGenome] = {}
    for subtask in subtasks:
        successful_genomes[subtask] = train_expert(subtask, expert_config)
    expert_agents = list(successful_genomes.values())
    train_master(expert_agents, expert_config=expert_config,
                 master_config=master_config)


if __name__ == "__main__":
    print('Program Started')
    main()
