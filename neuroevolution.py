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


def run_neat(config_path, extra_inputs=None, eval_func=XOR_eval, detail=True, display_best_genome=False,
             display_best_output=True, display_best_fitness=True, checkpoints=False, iterations=1_000,
             checkpoint_interval=100, checkpoint=None):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    if checkpoint and os.path.exists(checkpoint):
        print("NEAT Checkpoint Loaded")
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
        p.config = config
    else:
        p = neat.Population(config)
    
    p.add_reporter(TimedReporter(detail, interval=2))
    p.add_reporter(neat.StatisticsReporter())
    if checkpoints:
        p.add_reporter(neat.Checkpointer(checkpoint_interval))
    
    def eval_func_compressed(genomes, eval_config):
        if extra_inputs is None:
            eval_func(genomes, eval_config)
        else:
            eval_func(genomes, eval_config, *extra_inputs)
    
    winner = p.run(eval_func_compressed, iterations)
    
    if display_best_genome or winner.fitness >= config.fitness_threshold:
        # Display the winning genome.
        print(f'\nBest genome:\n{winner}')
    
    if display_best_fitness:
        print(f'\nBest Fitness: {winner.fitness:.3f}')
    
    if display_best_output:
        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        try:
            for [xi], [xo] in extra_inputs[0]:
                output = winner_net.activate(xi)
                print(f"input {xi}, expected output {xo}, got {[round(x, 2) for x in output]}")
        except RuntimeError as e:
            print(f'Could not display input output pairs. Error: {e}')
        except ValueError as e:
            print(f'Could not display input output pairs. Error: {e}')
    
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
    run_neat(config_path, extra_inputs=[input_output_pairs])


def main():
    test_neat()


if __name__ == "__main__":
    print("Program Started")
    main()
