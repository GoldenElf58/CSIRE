from ale_py import ALEInterface, roms
import concurrent.futures
import cv2
import itertools
import neat
import os
import subprocess
import sys
import threading
import time

from tf_utils import create_model
from neuroevolution import run_neat

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress informational messages


def load_rom_suppressed(game) -> None:
    # Create a temporary script to load the ROM
    script_content = f"""
import os
from ale_py import ALEInterface, roms

def load_rom():
    ale = ALEInterface()
    available_roms = roms.get_all_rom_ids()
    if '{game}' in available_roms:
        rom_path = roms.get_rom_path('{game}')
        ale.loadROM(rom_path)
    else:
        raise ValueError(f'ROM for game {game} not supported.\\nSupported ROMs: {{available_roms}}')

if __name__ == "__main__":
    load_rom()
"""
    
    script_path = "temp_script.py"
    with open(script_path, "w") as script_file:
        script_file.write(script_content)
    
    # Run the temporary script in a subprocess
    try:
        result = subprocess.run(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    finally:
        os.remove(script_path)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to load ROM: {result.stderr}\nReturn Code: {result.returncode}")
    else:
        print(result.stdout)


def convert_game_name(game_name, to_camel_case=True) -> str:
    if to_camel_case:
        if '_' not in game_name: return game_name
        words: list[str] = game_name.split('_')
        capitalized_words: list[str] = [word.capitalize() for word in words]
        return ''.join(capitalized_words)
    else:
        converted_name: str = ''.join(['_' + i.lower() if i.isupper() else i for i in game_name]).lstrip('_')
        return converted_name


def ale_init(game, suppress=False) -> ALEInterface:
    ale: ALEInterface = ALEInterface()
    
    if suppress:
        game = convert_game_name(game, False)
        load_rom_suppressed(game)
    else:
        game = convert_game_name(game, True)
        rom = getattr(roms, game)
        ale.loadROM(rom)
    return ale


def run_neat_model(model, inputs) -> list[float]:
    return model.activate(inputs)


def take_action(output, ale) -> int:
    # Take an action and get the new state
    legal_actions = ale.getLegalActionSet()
    action_index = output.index(max(output))
    action = legal_actions[action_index]  # Choose an action (e.g., NOOP)
    reward = ale.act(action)
    return reward


def clear() -> None:
    sys.stdout.write('\r' + '\n' * 50 + '\r')
    sys.stdout.flush()


def progress_bar(percent, bar_length=50) -> str:
    progress_length = int(bar_length * percent)
    bar = '█' * progress_length + '░' * (bar_length - progress_length)
    return bar


def load(stop_event, total_iterations, current_iteration, results, t0) -> None:
    loader = itertools.cycle(['|', '/', '-', '\\'])
    while not stop_event.is_set():
        percent_complete = (current_iteration[0] / total_iterations)
        bar = progress_bar(percent_complete)
        best = max(results) if len(results) > 0 else 0
        time_elapsed = time.time() - t0
        eta = time_elapsed * (1 / max(percent_complete, .01) - 1)
        print(
            f"\r{next(loader)} {bar} - {percent_complete * 100:.1f}%, ETA: {eta // 60:.0f}m {eta % 60:.0f}s {best:.3f}",
            end='')
        time.sleep(0.5)  # Adjust the delay for visual effect


def run_in_parallel(function, kwargs: None or list[dict] = None, iterations=100) -> list[float]:
    results = []
    stop_event = threading.Event()
    current_iteration = [0]
    t0 = time.time()
    
    # Start the loading sign in a separate thread
    loader_thread = threading.Thread(target=load, args=(stop_event, iterations, current_iteration, results, t0))
    loader_thread.start()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        if kwargs is None:
            futures = [executor.submit(function) for _ in range(iterations)]
        else:
            futures = [executor.submit(function, **kwargs[i]) for i in range(iterations)]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except KeyboardInterrupt:
                sys.exit()
            except Exception as e:
                print(f"Function raised an exception: {e}")
                results.append(None)
            current_iteration[0] += 1  # Increment the iteration count
    
    # Stop the loading sign
    stop_event.set()
    loader_thread.join()
    clear()
    return results


def terminate(incentive, death_message="Dead") -> tuple[bool, float]:
    print(death_message)
    incentive -= 100
    end: bool = True
    return end, incentive


def add_incentive(ram, last_life, last_action, death_clock) -> tuple[float, bool, bool, int]:
    incentive: float = 0
    end: bool = False
    
    if last_action == [0]:
        death_clock += 1
    else:
        death_clock = 0
    if death_clock > 60 * 5: incentive, end = terminate(incentive, death_message='Dead - Stalling')
    
    match ram[58]:
        case 0:
            if ram[55] == 0: last_life = True
            if ram[55] > 0 and last_life: incentive, end = terminate(incentive, death_message='Dead - Last Life')
        case _:
            incentive += ram[58] * .001
    
    match ram[66]:
        case 13:
            incentive += 0.3
        case 12:
            incentive += 0.6
        case 14:
            incentive += 0.9
    
    incentive -= .0001 * ram[43]
    return incentive, last_life, end, death_clock


def run_frames(frames=100, info=False, frames_per_step=1, game='MontezumaRevenge', suppress=False, model=create_model(),
               display_frames=False, activation=neat.nn.FeedForwardNetwork.activate) -> float:
    ale: ALEInterface = ale_init(game, suppress)
    reward: float = 0
    last_action: list[float] = [0]
    last_life: bool = False
    death_clock: int = 0
    
    for i in range(frames):
        inputs = ale.getRAM().reshape(1, -1)[0]
        incentive, last_life, end, death_clock = add_incentive(inputs, last_life, last_action, death_clock)
        if end: break
        reward += incentive
        
        if i % frames_per_step == 0:
            output = run_neat_model(model, inputs)
            reward += take_action(output, ale)
            last_action = output
        else:
            reward += take_action(last_action, ale)
        
        if display_frames:
            cv2.imshow('Image', ale.getScreenRGB())
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    if display_frames: cv2.destroyAllWindows()
    if info: print(f'Total Reward: {reward}')
    return reward


def game_eval(genomes, config, func_params=None, run_func=run_frames) -> None:
    if func_params is None:
        func_params = {}
    kwargs = []
    for _, genome in genomes:
        net: neat.nn.FeedForwardNetwork = neat.nn.FeedForwardNetwork.create(genome, config)
        kwargs.append({'frames': 60 * 60, 'frames_per_step': 3, 'model': net,
                       'activation': neat.nn.FeedForwardNetwork.activate} | func_params)
    results = run_in_parallel(run_func, kwargs=kwargs, iterations=len(kwargs))
    for i, [_, genome] in enumerate(genomes):
        genome.fitness = results[i]


def find_most_recent_checkpoint():
    files = os.listdir('.')
    best_file_num = None
    best_file_name = None
    for file in files:
        if 'neat-checkpoint' in file:
            current_file_num = int(file.split('-')[-1])
            if best_file_num is None or current_file_num > best_file_num:
                best_file_num = current_file_num
                best_file_name = file
    return best_file_name


def main() -> None:
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run_neat(config_path, eval_func=game_eval, checkpoints=True, checkpoint_interval=1,
             checkpoint=find_most_recent_checkpoint(), extra_inputs=[{'display_frames': True}])


if __name__ == "__main__":
    print('Program Started')
    main()
