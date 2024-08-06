from typing import Any, Type

from neat import DefaultGenome

from agent import Agent
from expert_agent import ExpertAgent
from logs import logger, setup_logging
from master_agent import MasterAgent
from neuroevolution import run_neat
from subtask_dictionary import subtask_dict
from utils import find_most_recent_file, run_in_parallel, find_highest_file_number, load_latest_state

worst_fitnesses: list[Any | float] = []
average_fitnesses: list[Any | float] = []
best_fitnesses: list[Any | float] = []


def game_eval(genomes, config, func_params=None, agent_type: Type[Agent] = Agent, run_func=None) -> None:
    """
    The evaluation function for a set of genomes. Takes in the genomes and sets their fitness.
    :param genomes: A list of the genomes to be tested
    :param config: The configuration of the genomes
    :param func_params: The parameters to be passed to the 'run_frames' function
    :param run_func: The function to test the genomes in
    :param agent_type: The type of agent (e.g. ExpertAgent, MasterAgent, Agent
    :return: None
    """
    global worst_fitnesses
    global average_fitnesses
    global best_fitnesses
    if run_func is None:
        run_func = agent_type.test_agent
    if func_params is None:
        func_params = {}
    args = []
    for i, (_, genome) in enumerate(genomes):
        kwargs = {'frames': 60 * 30, 'frames_per_step': 2} | func_params
        agent: Agent = agent_type(genome, config, i, **kwargs)
        args.append([agent])
    results: list[tuple[float, int]] = run_in_parallel(run_func, args=args, iterations=len(args))
    worst = float('inf')
    best = float('-inf')
    average = 0
    for result in results:
        current_fitness = float(result[0])
        genomes[result[1]][1].fitness = current_fitness
        if current_fitness > best:
            best = current_fitness
        if current_fitness < worst:
            worst = current_fitness
        average += current_fitness
    average /= len(results)
    worst_fitnesses.append(round(worst, 2))
    best_fitnesses.append(round(best, 2))
    average_fitnesses.append(round(average, 2))


def train_expert(subtask: str = 'beam', subtask_scenarios: dict = None, base_filename: str = 'successful-genome',
                 expert_config: str = 'config-feedforward-expert', checkpoint_name=None, generations=1_000):
    """Trains an expert for a given subtask

    :param subtask: The subtask to be trained on
    :param subtask_scenarios: The different scenarios a subtask has
    :param base_filename: The base filename before the subtask name
    :param expert_config: The name of the configuration file for the expert agents
    :param checkpoint_name: Name of the checkpoint file
    :param generations: Number of generations to train agent
    :return:
    """
    logger.info(f" {'=' * (23 + len(subtask))}\nTraining Expert Genome {subtask}\n{'=' * (23 + len(subtask))}")
    if subtask_scenarios is None:
        subtask_scenarios = subtask_dict['beam']

    if subtask is not None:
        base_filename = f'{base_filename}-{subtask}'
    if checkpoint_name is None:
        checkpoint_name = f'neat-checkpoint-{subtask}'
    successful_genomes = []  # list(set(load_specific_state(file) for file in find_all_files(base_filename)))
    best_genome = run_neat(expert_config, eval_func=game_eval, checkpoints=True, checkpoint_interval=50,
                           checkpoint=find_most_recent_file(f'neat-checkpoint-{subtask}'), insert_genomes=False,
                           genomes=successful_genomes, generations=generations, base_filename=base_filename,
                           base_checkpoint_filename=checkpoint_name,
                           extra_inputs=[{'visualize': False, 'subtask': subtask, 'info': False,
                                          'subtask_scenarios': subtask_scenarios}, ExpertAgent])
    return best_genome


def train_master(expert_genomes: list[DefaultGenome], base_filename: str = 'successful-genome-master',
                 expert_config: str = 'config-feedforward-expert', checkpoint_name: str = 'neat-checkpoint-master',
                 master_config: str = 'config-feedforward-master') -> DefaultGenome:
    """
    Trains an expert for a given subtask
    :param expert_genomes: The previously trained expert genomes
    :param base_filename: The base filename before the subtask name
    :param expert_config: The name of the configuration file for the expert agents
    :param master_config: The name of the configuration file for the master agent
    :param checkpoint_name: Name of the checkpoint file
    :return:
    """
    logger.info(f"{15 * '='}\nTraining Master\n{15 * '='}")
    successful_genomes = []  # list(set(load_specific_state(file) for file in find_all_files(base_filename)))
    best_genome = run_neat(master_config, eval_func=game_eval, checkpoints=True, checkpoint_interval=50,
                           checkpoint=find_most_recent_file(checkpoint_name), insert_genomes=False,
                           genomes=successful_genomes, generations=1_000, base_filename=base_filename,
                           base_checkpoint_filename=checkpoint_name,
                           extra_inputs=[{'visualize': False, 'expert_genomes': expert_genomes,
                                          'expert_config_name': expert_config}, MasterAgent])
    return best_genome


def main() -> None:
    """
    The main function of the program
    :return: None
    """
    subtasks: list[str] = list(subtask_dict.keys())
    expert_config: str = 'config-feedforward-expert'
    master_config: str = 'config-feedforward-master'
    successful_genomes: dict[str, DefaultGenome] = {}
    for subtask in subtasks:
        generations = 1_000
        highest_file_number = find_highest_file_number(f'neat-checkpoint-{subtask}')
        if highest_file_number is None:
            pass
        elif highest_file_number >= generations:
            successful_genomes[subtask] = load_latest_state(base_filename=f'successful-genome-{subtask}')
            logger.warning(f"Skipping {subtask} because {highest_file_number} is the most recent checkpoint number")
            continue
        else:
            generations = generations - highest_file_number
            logger.warning(f"Running {subtask} for {generations} generations")

        worst_fitnesses.append(subtask)
        best_fitnesses.append(subtask)
        average_fitnesses.append(subtask)
        successful_genomes[subtask] = train_expert(subtask, subtask_scenarios=subtask_dict[subtask],
                                                   expert_config=expert_config, generations=generations)
        logger.debug(f'Best: {best_fitnesses}\nAverage: {average_fitnesses}\nWorst: {worst_fitnesses}')
    worst_fitnesses.append('master')
    best_fitnesses.append('master')
    average_fitnesses.append('master')
    expert_agents = list(successful_genomes.values())
    train_master(expert_agents, expert_config=expert_config, master_config=master_config)
    logger.debug(f'Best: {best_fitnesses}\nAverage: {average_fitnesses}\nWorst: {worst_fitnesses}')


if __name__ == "__main__":
    setup_logging()
    logger.debug('========= Program Started =========')
    main()
