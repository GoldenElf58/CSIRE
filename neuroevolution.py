"""
This file is used for the genetic algorithm NEAT
"""

import os
import time
import traceback

from neat import (Config, DefaultGenome, StdOutReporter, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation,
                  Checkpointer, Population, StatisticsReporter, nn)

from expert_agent import ExpertAgent
from logs import logger
from utils import save_state


class TimedReporter(StdOutReporter):
    def __init__(self, show_species_detail, interval=5):
        super().__init__(show_species_detail)
        self.interval = interval
        self.last_time = time.time()
        self.should_print = False

    def start_generation(self, generation):
        self.generation = generation
        current_time = time.time()
        if current_time - self.last_time >= self.interval or generation in {0, 999}:
            self.should_print = True
            self.last_time = current_time
            super().start_generation(generation)
        else:
            self.should_print = False

    def post_evaluate(self, config, population, species, best_genome):
        if self.should_print:
            super().post_evaluate(config, population, species, best_genome)

    def end_generation(self, config, population, species_set):
        if self.should_print:
            super().end_generation(config, population, species_set)

    def info(self, msg):
        if self.should_print:
            super().info(msg)


def XOR_eval(genomes, config, input_output_pairs=None):
    """
    An example XOR evaluation function that can be used for evaluating different genomes
    :param genomes: The genomes to be evaluated
    :param config: The configuration of the genomes
    :param input_output_pairs: The pairs of inputs and correct outputs
    :return: A list of the fitnesses of the genomes
    """
    if input_output_pairs is None:
        logger.warn("No input-output pairs given")
        return
    for genome_id, genome in genomes:
        net = nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 4.0  # Max fitness

        for [inputs], [expected_outputs] in input_output_pairs:
            output = net.activate(inputs)
            genome.fitness -= sum((output[i] - expected_outputs[i]) ** 2 for i in range(len(output)))


def adjust_genome_node_ids(genome, start_id):
    node_mapping = {}
    for i, node_key in enumerate(sorted(genome.nodes.keys())):
        node_mapping[node_key] = start_id + i

    new_nodes = {}
    for old_id, new_id in node_mapping.items():
        new_nodes[new_id] = genome.nodes[old_id]
    genome.nodes = new_nodes

    new_connections = {}
    for conn_key, conn in genome.connections.items():
        try:
            new_key = (node_mapping[conn_key], node_mapping[conn_key[1]])
        except KeyError as e:
            logger.error(f"Error in connection keys: {conn_key}")
            logger.error(f"Node mapping: {node_mapping}")
            raise e
        new_connections[new_key] = conn
    genome.connections = new_connections


def get_highest_node_id(population):
    highest_node_id = -1
    for genome_id, genome in population.items():
        for node_id in genome.nodes.keys():
            highest_node_id = max(highest_node_id, node_id)
    return highest_node_id


def insert_genomes_into_population(p, genomes):
    if p.population:
        highest_node_id = get_highest_node_id(p.population)
    else:
        highest_node_id = -1

    for genome in genomes:
        if highest_node_id >= 0:
            adjust_genome_node_ids(genome, highest_node_id + 1)
        highest_node_id += len(genome.nodes)
        p.population[genome.key] = genome
        p.species.genome_to_species[genome.key] = genome


def insert_genome(population, genome, start_id):
    """Inserts a genome into a population, ensuring no negative node keys.

    Args:
        population: The population to insert into.
        genome: The genome to insert.
        start_id: The starting ID for newly added nodes.

    Returns:
        The updated starting ID for subsequent node additions.
    """
    # Find minimum node key in the genome
    min_node_key = min(genome.nodes.keys())

    # Adjust node keys if minimum is negative
    if min_node_key < 0:
        offset = abs(min_node_key)
        genome.nodes = {k + offset: v for k, v in genome.nodes.items()}
        genome.connections = {(k1 + offset, k2 + offset): v for (k1, k2), v in genome.connections.items()}

    # Insert the genome with adjusted keys
    population.nodes = {**population.nodes, **genome.nodes}
    population.connections = {**population.connections, **genome.connections}
    return start_id + len(genome.nodes)


def run_neat(config_path, extra_inputs: list | None = None, eval_func=XOR_eval, detail=True, display_best_genome=False,
             display_best_output=True, display_best_fitness=True, checkpoint=None, checkpoints=True,
             checkpoint_interval=100, generations=1_000, insert_genomes=False, genomes=None, report_interval=2,
             save_best_genome=True, base_filename='successful-genome',
             base_checkpoint_filename='neat-checkpoint') -> DefaultGenome:
    """Runs NEAT based on many parameters:

    :param config_path: The path to the configuration file (e.g. 'config-feedforward')
    :param extra_inputs: Extra inputs to be given to the evaluation function
    :param eval_func: The evaluation function
    :param detail: Whether to show more or less detail about each generation in the console
    :param display_best_genome: Whether to display the best genome at the end of training in the console
    :param display_best_output: Whether to display the output of the best genome at the end of training (only for XOR)
    :param display_best_fitness: Whether to display the fitness of the best genome at the end of training in the conole
    :param checkpoint: A previous checkpoint that will be loaded in, instead of a new generation
    :param checkpoints: Whether to create checkpoints of the generations
    :param checkpoint_interval: The number of generations minus one between each checkpoint
    :param generations: The number of generations to train the genomes
    :param insert_genomes: Whether to insert genomes into the start generation - WARNING: Can lead to errors
    :param genomes: The genomes to be inserted into the start generation
    :param report_interval: Minimum seconds between generation reporting
    :param save_best_genome: DANGEROUS - Whether to save the best genome when the simulation ends - DANGEROUS
    :param base_filename: The base file name for the file the best genome will be saved in
    :param base_checkpoint_filename: The base filename for neat checkpoints
    :return: The best genome in the final generation
    """
    config: Config = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, config_path)

    if checkpoint and os.path.exists(checkpoint):
        logger.info("NEAT Checkpoint Loaded")
        p = Checkpointer.restore_checkpoint(checkpoint)
        p.config = config
    else:
        p = Population(config)

    if insert_genomes and genomes is not None:
        insert_genomes_into_population(p, genomes)

    p.add_reporter(TimedReporter(detail, interval=report_interval))
    p.add_reporter(StatisticsReporter())
    if checkpoints:
        p.add_reporter(Checkpointer(checkpoint_interval, filename_prefix=f'{base_checkpoint_filename}-'))

    def eval_func_compressed(eval_genomes, eval_config):
        if extra_inputs is None:
            eval_func(eval_genomes, eval_config)
        else:
            eval_func(eval_genomes, eval_config, *extra_inputs)

    winner = p.run(eval_func_compressed, generations)

    if display_best_genome:
        logger.info(f'\nBest genome:\n{winner}')

    if display_best_fitness:
        logger.info(f'\nBest Fitness: {winner.fitness:.3f}')

    if save_best_genome:
        save_state(winner, base_filename)
    else:
        choice: str | None = None
        while choice not in {'Y', 'N', 'n'}:
            choice = input("Are you sure you don't want to save the best genome? (Y/n)  ")
            if choice.lower() == 'n':
                save_state(winner, base_filename)
            elif choice != 'Y':
                logger.warn("Invalid choice. Try again.")

    if display_best_output:
        # Show output of the most fit genome against training data.
        logger.info('\nOutput:')
        winner_net = nn.FeedForwardNetwork.create(winner, config)
        if eval_func == XOR_eval:
            try:
                for [xi], [xo] in extra_inputs[0]:
                    output = winner_net.activate(xi)
                    logger.info(f"input {xi}, expected output {xo}, got {[round(x, 2) for x in output]}")
            except (RuntimeError, ValueError) as e:
                logger.warn(f'Could not display input output pairs.\nError type: {type(e)}.\nError: {e}')
                traceback.print_exc()
        else:
            try:
                kwargs = {'frames': 60 * 30, 'frames_per_step': 2, 'suppress': False} | extra_inputs[0]
                kwargs['visualize'] = True
                kwargs['info'] = True
                agent: ExpertAgent = ExpertAgent(winner, config, -1, **kwargs)
                agent.test_agent()
            except Exception as e:
                logger.warn(f"Could not visualize successful genome.\nError type: {type(e)}.\nError: {e}")
                traceback.print_exc()

    return winner


def test_neat() -> None:
    """
    A function that tests the NEAT algorithm on the XOR function
    :return: None
    """
    input_output_pairs = [
        ([(0, 0)], [(0,)]),
        ([(0, 1)], [(1,)]),
        ([(1, 0)], [(1,)]),
        ([(1, 1)], [(0,)])
    ]

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-test')
    run_neat(config_path, extra_inputs=[input_output_pairs], save_best_genome=False, checkpoints=False)


def main() -> None:
    """
    The main function of the program
    :return: None
    """
    test_neat()


if __name__ == "__main__":
    logger.info("Program Started")
    main()
