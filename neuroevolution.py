import neat
import os
import time


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
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 4.0  # Max fitness
        
        for [inputs], [expected_outputs] in input_output_pairs:
            output = net.activate(inputs)
            genome.fitness -= sum((output[i] - expected_outputs[i]) ** 2 for i in range(len(output)))


def run_neat(config_path, input_output_pairs, detail=True, display_best_genome=False, display_best_output=True,
             display_best_fitness=True, checkpoints=False):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    p = neat.Population(config)
    
    p.add_reporter(TimedReporter(detail, interval=2))
    p.add_reporter(neat.StatisticsReporter())
    if checkpoints:
        p.add_reporter(neat.Checkpointer())
    
    # Use a lambda function to pass the inputs and outputs to the eval_genomes_wrapper
    def eval_func(genomes, eval_config):
        XOR_eval(genomes, eval_config, input_output_pairs)
    
    winner = p.run(eval_func, 1_000)
    
    if display_best_genome or winner.fitness > config.fitness_threshold:
        # Display the winning genome.
        print(f'\nBest genome:\n{winner}')
    
    if display_best_fitness:
        print(f'\nBest Fitness: {winner.fitness:.3f}')
    
    if display_best_output:
        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        for [xi], [xo] in input_output_pairs:
            output = winner_net.activate(xi)
            print(f"input {xi}, expected output {xo}, got {[round(x, 2) for x in output]}")
    
    return winner


def test_neat():
    input_output_pairs = [
        ([(0, 0)], [(0,)]),
        ([(0, 1)], [(1,)]),
        ([(1, 0)], [(1,)]),
        ([(1, 1)], [(0,)])
    ]
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-test')
    run_neat(config_path, input_output_pairs)


def main():
    test_neat()


if __name__ == "__main__":
    print("Program Started")
    main()
