"""
This file is used for the genetic algorithm NEAT
"""

import os
import time

import neat
import neat.genes

from agent import Agent
from utils import save_state


class TimedReporter(neat.StdOutReporter):
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


def XOR_eval(genomes, config, input_output_pairs):
    """
    An example XOR evaluation function that can be used for evaluating different genomes
    :param genomes: The genomes to be evaluated
    :param config: The configuration of the genomes
    :param input_output_pairs: The pairs of inputs and correct outputs
    :return: A list of the fitnesses of the genomes
    """
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 4.0  # Max fitness
        
        for [inputs], [expected_outputs] in input_output_pairs:
            output = net.activate(inputs)
            genome.fitness -= sum((output[i] - expected_outputs[i]) ** 2 for i in range(len(output)))


def create_genome_from_string(genome_str: str) -> neat.DefaultGenome:
    """
    WARNING: This function seems to not work correctly
    Returns a renome based on a string that defines the genome. The user may have copied this from the console.
    :param genome_str: The genome in the format of a string
    :return: The genome in the format of the DefaultGenome class
    """
    genome = neat.DefaultGenome(key=0)
    genome.connections.clear()
    genome.nodes.clear()
    
    lines = genome_str.strip().split('\n')
    for line in lines:
        if line.startswith('\t') or line.startswith(' '):
            line = line.strip()
            if line.startswith('DefaultNodeGene'):
                key = int(line.split('(')[1].split(')')[0])
                attributes = line.split('=')[1].split(',')
                bias = float(attributes[0].split('=')[1])
                response = float(attributes[1].split('=')[1])
                activation = attributes[2].split('=')[1].strip()
                aggregation = attributes[3].split('=')[1].strip()
                node_gene = neat.genes.DefaultNodeGene(key)
                node_gene.bias = bias
                node_gene.response = response
                node_gene.activation = activation
                node_gene.aggregation = aggregation
                genome.nodes[key] = node_gene
            elif line.startswith('DefaultConnectionGene'):
                key = eval(line.split('(')[1].split(')')[0])
                attributes = line.split('=')[1].split(',')
                weight = float(attributes[0].split('=')[1])
                enabled = attributes[1].split('=')[1].strip() == 'True'
                connection_gene = neat.genes.DefaultConnectionGene(key)
                connection_gene.weight = weight
                connection_gene.enabled = enabled
                genome.connections[key] = connection_gene
    return genome


def run_neat(config_path, extra_inputs: list or None = None, eval_func=XOR_eval, detail=True, display_best_genome=True,
             display_best_output=True, display_best_fitness=True, checkpoint=None, checkpoints=False,
             checkpoint_interval=100, generations=1_000, insert_genomes=False, genomes=None):
    """
    Runs NEAT based on many parameters:
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
    :param insert_genomes: Whether to insert genomes into the start generation
    :param genomes: The genomes to be inserted into the start generation
    :return: The best genome in the final generation
    """
    config: neat.config.Config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                                    config_path)
    
    if checkpoint and os.path.exists(checkpoint):
        print("NEAT Checkpoint Loaded")
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
        p.config = config
    else:
        p = neat.Population(config)
    
    if insert_genomes and genomes is not None:
        for genome in genomes:
            p.population[genome.key] = genome
            p.species.genome_to_species[genome.key] = genome
    
    p.add_reporter(TimedReporter(detail, interval=2))
    p.add_reporter(neat.StatisticsReporter())
    if checkpoints:
        p.add_reporter(neat.Checkpointer(checkpoint_interval))
    
    def eval_func_compressed(eval_genomes, eval_config):
        if extra_inputs is None:
            eval_func(eval_genomes, eval_config)
        else:
            eval_func(eval_genomes, eval_config, *extra_inputs)
    
    winner = p.run(eval_func_compressed, generations)
    
    if display_best_genome or winner.fitness >= config.fitness_threshold:
        print(f'\nBest genome:\n{winner}')
    
    if display_best_fitness:
        print(f'\nBest Fitness: {winner.fitness:.3f}')

    save_state(winner, 'successful-genome')

    if display_best_output:
        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        try:
            for [xi], [xo] in extra_inputs[0]:
                output = winner_net.activate(xi)
                print(f"input {xi}, expected output {xo}, got {[round(x, 2) for x in output]}")
        except (RuntimeError, ValueError) as e:
            print(f'Could not display input output pairs. Error: {e}')

        try:
            kwargs = {'frames': 60 * 30, 'frames_per_step': 2, 'info': True, 'suppress': False} | extra_inputs[0]
            kwargs['visualize'] = True
            agent: Agent = Agent(winner, config, 0, **kwargs)
            agent.run_frames()
        except Exception as e:
            print(f"Could not visualize successful genome. Error: {e}")
    
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
    run_neat(config_path, extra_inputs=[input_output_pairs])


def main() -> None:
    """
    The main function of the program
    :return: None
    """
    test_neat()


if __name__ == "__main__":
    print("Program Started")
    main()
