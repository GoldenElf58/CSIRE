import concurrent.futures
import itertools
import logging
import os
import sys
import threading
import time
import traceback
from typing import Callable

from agent import Agent
from neuroevolution import run_neat
from utils import find_most_recent_file, load_specific_state, find_all_files


def clear() -> None:
    """
    Prints 50 lines to the console to essentially 'clear' it
    :return: None
    """
    sys.stdout.write('\r' + '\n' * 25 + '\r')
    sys.stdout.flush()


def progress_bar(percent: float, bar_length=50) -> str:
    """
    Returns a progress bar based on a percentage and bar length.
    :param percent: Percentage of bar
    :param bar_length: Length of bar
    :return: Progress bar
    """
    progress_length = int(bar_length * percent)
    bar = '█' * progress_length + '░' * (bar_length - progress_length)
    return bar


def load(stop_event, total_iterations, current_iteration, results, t0) -> None:
    """
    Shows a loading animation and other information while other tasks are running.
    :param stop_event: When this function terminates (e.g. when the other tasks are over)
    :param total_iterations: Total iterations into task (e.g. 50 iterations of function x)
    :param current_iteration: Number of finished iterations (e.g. 23 iterations have been completed)
    :param results: The fitnesses of the NEAT models, once they are finished
    :param t0: The start time of the function
    :return: None
    """
    loader = itertools.cycle(['|', '/', '-', '\\'])
    while not stop_event.is_set():
        percent_complete = (current_iteration[0] / total_iterations)
        bar = progress_bar(percent_complete)
        best = max(results, key=lambda x: x[0])[0] if len(results) > 0 else 0
        time_elapsed = time.time() - t0
        eta = time_elapsed * (1 / max(percent_complete, 4 / total_iterations) - 1)
        print(
            f"\r{next(loader)} {bar} - {percent_complete * 100:.1f}%, ETA: {eta // 60:.0f}m {eta % 60:.0f}s {best:.3f}",
            end='')
        time.sleep(0.5)  # Adjust the delay for visual effect


def run_in_parallel(function: Callable, args: None or list[list] = None, kwargs: None or list[dict] = None,
                    iterations: int = 100) -> list:
    """
    Takes in a function, keyword arguments for that function, and a number of iterations, and runs that function in
    parallel with those arguments for the given number of iterations
    :param function: The function to be run in parallel
    :param args: A list of lists of the arguments to be passed to each function
    :param kwargs: A list of dictionaries of the arguments to be passed to each function
    :param iterations: The number of times the function needs to be run
    :return: A list of the results of each individual run of the function
    """
    results = []
    stop_event = threading.Event()
    current_iteration = [0]
    t0 = time.time()

    # Start the loading sign in a separate thread
    loader_thread = threading.Thread(target=load, args=(stop_event, iterations, current_iteration, results, t0))
    loader_thread.start()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        if kwargs is None:
            if args is None:
                futures = [executor.submit(function) for _ in range(iterations)]
            else:
                futures = [executor.submit(function, *args[i]) for i in range(iterations)]
        else:
            if args is None:
                futures = [executor.submit(function, **kwargs[i]) for i in range(iterations)]
            else:
                futures = [executor.submit(function, *args[i], **kwargs[i]) for i in range(iterations)]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except KeyboardInterrupt:
                sys.exit()
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()
                print(e)
                results.append(None)
            current_iteration[0] += 1  # Increment the iteration count

    # Stop the loading sign
    stop_event.set()
    loader_thread.join()
    clear()
    return results


def game_eval(genomes, config, func_params=None, run_func=Agent.run_frames) -> None:
    """
    The evaluation function for a set of genomes. Takes in the genomes and sets their fitness.
    :param genomes: A list of the genomes to be tested
    :param config: The configuration of the genomes
    :param func_params: The parameters to be passed to the 'run_frames' function
    :param run_func: The function to test the genomes in
    :return: None
    """
    if func_params is None:
        func_params = {}
    args = []
    for i, (_, genome) in enumerate(genomes):
        kwargs = {'frames': 60 * 30, 'frames_per_step': 2} | func_params
        agent: Agent = Agent(genome, config, i, **kwargs)
        args.append([agent])
    results = run_in_parallel(run_func, args=args, iterations=len(args))
    for result in results:
        genomes[result[1]][1].fitness = float(result[0])


def setup_logging() -> logging.Logger:
    """
    Sets up the loggin configuration and returns a logger.
    :return: A logger that will be used
    """
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('stdout2.log')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    logging.getLogger().addHandler(stdout_handler)
    logging.getLogger().addHandler(file_handler)

    logger: logging.Logger = logging.getLogger()
    return logger


def main() -> None:
    """
    The main function of the program
    :return: None
    """
    successful_genomes = list(set(load_specific_state(file) for file in find_all_files('successful-genome')))
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run_neat(config_path, eval_func=game_eval, checkpoints=True, checkpoint_interval=1,
             checkpoint=find_most_recent_file('neat-checkpoint'), insert_genomes=True, genomes=successful_genomes,
             extra_inputs=[{'visualize': False, 'load_state': 'beam-0'}])


if __name__ == "__main__":
    print('Program Started')
    main()

