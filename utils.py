import concurrent.futures
import itertools
from math import sqrt
import os
import pickle
import sys
import threading
import time
from typing import Any, Callable
import warnings

import neat
from ale_py import Action, ALEInterface


def clear(lines: int = 25) -> None:
    """
    Prints lines number of lines to the console to essentially 'clear' it
    :param lines: Number of lines to clear
    :return: None
    """
    sys.stdout.write('\r' + '\n' * lines + '\r')
    sys.stdout.flush()


def progress_bar(percent: float, bar_length=50, bar_loaded='█', bar_unloaded='░') -> str:
    """
    Returns a progress bar based on a percentage and bar length.
    :param percent: Percentage of bar
    :param bar_length: Length of bar
    :param bar_loaded: Character when that segment of the bar is loaded
    :param bar_unloaded: Character when that segment of the bar is not loaded
    :return: Progress bar
    """
    progress_length = int(bar_length * percent)
    bar = bar_loaded * progress_length + bar_unloaded * (bar_length - progress_length)
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
                stop_event.set()
                loader_thread.join()
                raise e
                # print("An error occurred:")
                # traceback.print_exc()
                # print(e)
                # results.append(None)
            current_iteration[0] += 1  # Increment the iteration count

    # Stop the loading sign
    stop_event.set()
    loader_thread.join()
    clear()
    return results


def find_most_recent_file(base_filename='neat_checkpoint'):
    """
    Returns the name of the most recent file of a specific base filename (based on the number at the end of the file
    name (e.g. 'neat-checkpoint-172')
    :param base_filename: Base filename (e.g. 'neat-checkpoint' or 'save-state')
    :return: Name of the most recent checkpoint file
    """
    files = os.listdir('.')
    latest_file_num = None
    latest_file_name = None
    for file in files:
        if file.startswith(base_filename + '-'):
            try:
                current_file_num = int(file.split('-')[-1])
                if latest_file_num is None or current_file_num > latest_file_num:
                    latest_file_num = current_file_num
                    latest_file_name = file
            except ValueError:
                pass

    return latest_file_name


def find_all_files(base_filename='successful-genome'):
    """
    Finds all files that start with a base filename
    :param base_filename: The base filename that a file starts with
    :return: A list of names of files that start with the base filename
    """
    files = []
    for file in os.listdir('.'):
        if file.startswith(base_filename + '-'):
            files.append(file)

    return files


def save_state(data: Any, base_filename: str = "save-state") -> None:
    """Saves data to a file with an incrementing number in the filename.
        :param data: The data to save.
        :param base_filename: The base filename (without extension).
        :return: None
    """
    largest_number = -1
    for filename in os.listdir('.'):  # Get list of files in current directory [1]
        if filename.startswith(base_filename + '-'):
            try:
                number = int(filename[len(base_filename) + 1:].split('.')[0])
                largest_number = max(largest_number, number)
            except ValueError:
                pass  # Ignore files with non-numeric extensions

    next_number = largest_number + 1
    filename = f"{base_filename}-{next_number}"
    save_specific_state(data, filename)


def save_specific_state(data: Any, filename: str, choice: str = 'Y'):
    if filename in os.listdir('.'):
        choice: str = input("This file already exists. Overwrite it? (Y/n)\n")
    if choice != 'Y':
        return
    filepath = os.path.join('.', filename)
    with open(filepath, "wb") as file:
        pickle.dump(data, file)


def load_latest_state(base_filename="save-state") -> Any | None:
    """Loads data from the most recent file with the given base filename.
        :param base_filename: The base filename (without extension).
        :return: The loaded data, or None if no matching files are found.
    """
    latest_filename: str = find_most_recent_file(base_filename)

    if latest_filename is not None:
        return load_specific_state(latest_filename)

    return None  # No matching files found


def load_specific_state(filename: str) -> Any | None:
    """
    Loads a game state with a specific filename.
    :param filename: Filename with game state to be loaded
    :return: Game state
    """
    if filename in os.listdir('.'):
        filepath = os.path.join('.', filename)
        with open(filepath, 'rb') as file:
            return pickle.load(file)

    return None  # No matching files found


def convert_game_name(game_name: str, to_camel_case=True) -> str:
    """
    Converts a game name (e.g. 'MontezumaRevenge') to or from camel case
    :param game_name: String (e.g. 'MontezumaRevenge')
    :param to_camel_case: Dictates whether the string is being converted to or from camel case
    :return: String (e.g. 'montezuma_revenge') - Returns converted game name
    """
    if to_camel_case:
        if '_' not in game_name:
            return game_name
        words: list[str] = game_name.split('_')
        capitalized_words: list[str] = [word.capitalize() for word in words]
        return ''.join(capitalized_words)
    else:
        converted_name: str = ''.join(['_' + i.lower() if i.isupper() else i for i in game_name]).lstrip('_')
        return converted_name


def take_action(action_index, ale: ALEInterface) -> int:
    """
    Takes an action in a given ALEInterface
    :param action_index: Index of action to be taken (e.g. 1 is JUMP)
    :param ale: ALEInterface with game loaded
    :return: Reward from that action
    """
    # Take an action and get the new state
    legal_actions: list[Action] = ale.getLegalActionSet()
    action = legal_actions[action_index]  # Choose an action (e.g., NOOP)
    reward: int = ale.act(action)
    return reward


def get_action_index(output: list[float]) -> int:
    """
    Takes in the outputs of the NEAT model and returns the action corresponding to the output with the highest value
    :param output: The output of the NEAT model
    :return: The index of the action to be taken
    """
    action_index: int = output.index(max(output))
    if action_index >= 6:
        action_index += 5
    return action_index


def divide_list(lst, n) -> list[list[Any]]:
    """Divides the list lst into n equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def average_elements_at_indexes(lst: list[list[float]]) -> list[float]:
    """Compute the average of elements at each index in a list of lists."""
    if not lst:
        return []

    num_lists = len(lst)
    num_elements = len(lst[0])
    averages = [0] * num_elements

    for inner_list in lst:
        for i in range(num_elements):
            averages[i] += inner_list[i]

    return [total / num_lists for total in averages]


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


def run_neat_model(model, inputs) -> list[float]:
    """
    Runs a NEAT neural network and returns the results
    :param model: FeedForwardNetwork; The NEAT model
    :param inputs: Inputs to NEAT model (e.g. environment observations)
    :return: Model results
    """
    return model.activate(inputs)


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    except RuntimeWarning as e:
        print(f"RuntimeWarning caught: {e}, Coordinates: {(x1, y1), (x2, y2)}")
        return float('inf')
