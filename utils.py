import os
import pickle
from typing import Any

from ale_py import Action, ALEInterface


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


def load_latest_state(base_filename="save-state"):
    """Loads data from the most recent file with the given base filename.
        :param base_filename: The base filename (without extension).
        :return: The loaded data, or None if no matching files are found.
    """
    latest_filename = find_most_recent_file(base_filename)

    if latest_filename is not None:
        return load_specific_state(latest_filename)

    return None  # No matching files found


def load_specific_state(filename: str):
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


def run_neat_model(model, inputs) -> list[float]:
    """
    Runs a NEAT neural network and returns the results
    :param model: FeedForwardNetwork; The NEAT model
    :param inputs: Inputs to NEAT model (e.g. environment observations)
    :return: Model results
    """
    return model.activate(inputs)


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
