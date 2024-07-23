from ale_py import ALEInterface, roms
import concurrent.futures
import cv2
import itertools
import neat
import numpy as np
import os
from PIL import Image
import subprocess
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
import threading
import time

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


def load_tflite_model() -> tuple[list, list, tf.lite.Interpreter]:
    # Load the TFLite model and allocate tensors
    model_path: str = '2.tflite'
    interpreter: tf.lite.Interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details: list = interpreter.get_input_details()
    output_details: list = interpreter.get_output_details()
    return input_details, output_details, interpreter


def load_image(img, input_details) -> np.array:
    img_pil = Image.fromarray(img)  # Convert numpy array to PIL Image
    target_size = (input_details[0]['shape'][2], input_details[0]['shape'][1])  # Get target size from input_details
    resized_img_pil = img_pil.resize(target_size, Image.LANCZOS)  # Resize the image with high-quality down sampling
    resized_img = np.array(resized_img_pil)  # Convert back to numpy array
    
    # Ensure the image has the correct number of channels
    if len(resized_img.shape) == 2:  # Grayscale image
        resized_img = np.expand_dims(resized_img, axis=-1)
    if resized_img.shape[-1] == 1 and input_details[0]['shape'][-1] == 3:  # Convert grayscale to RGB
        resized_img = np.repeat(resized_img, 3, axis=-1)
    if resized_img.shape[-1] == 3 and input_details[0]['shape'][-1] == 1:  # Convert RGB to grayscale
        resized_img = np.dot(resized_img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        resized_img = np.expand_dims(resized_img, axis=-1)
    
    input_data = np.expand_dims(resized_img, axis=0)  # Add batch dimension
    
    return input_data


def display_bounding_boxes(img, boxes, classes, scores, scale_factor=3) -> None:
    # Draw the boxes and labels on the image
    for i in range(len(boxes)):
        if scores[i] > 0.0:  # You can set a threshold for displaying boxes
            ymin, xmin, ymax, xmax = boxes[i]
            start_point = (int(xmin * img.shape[1]), int(ymin * img.shape[0]))
            end_point = (int(xmax * img.shape[1]), int(ymax * img.shape[0]))
            color = (255, 0, 0)  # BGR
            thickness = 1
            
            # Draw the rectangle
            img = cv2.rectangle(img, start_point, end_point, color, thickness)
            
            # Draw the label
            label = f'{int(classes[i])}, {scores[i]:.2f}'
            print(int(classes[i]), scores[i])
            img = cv2.putText(img, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1, cv2.LINE_AA)
    
    # Resize the image to make it bigger
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    
    # Display the image
    cv2.imshow("Image with Bounding Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_tflite(img, input_details, output_details, interpreter, info=False, display=False) -> tuple[
    list[float], list[int], list[list[float]]]:
    input_data = load_image(img, input_details)
    
    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get the Results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
    
    if info:
        print("Boxes:", boxes)
        print("Classes:", classes)
        print("Scores:", scores)
    
    if display:
        display_bounding_boxes(img, boxes, classes, scores)
    
    return classes, scores, boxes


def create_model() -> models.Sequential:
    model: models.Sequential = models.Sequential([
        layers.Input(shape=(128,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(18, activation='softmax')
    ])
    return model


def run_model(model, inputs) -> list[float]:
    return list(model.predict(inputs, verbose=0)[0])


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


def run_frames(frames=100, info=False, frames_per_step=1, game='MontezumaRevenge', suppress=False, model=create_model(),
               activation=models.Sequential.predict) -> int:
    ale = ale_init(game, suppress)
    reward = 0
    last_action = 0
    last_life = False
    for i in range(frames):
        inputs = ale.getRAM().reshape(1, -1)[0]
        match inputs[58]:
            case 0:
                if inputs[55] == 0: last_life = True
                if inputs[55] > 0 and last_life:
                    print('Dead')
                    reward -= 100
                    break
            case _:
                reward += inputs[58] * .001
        match inputs[66]:
            case 13:
                reward += 0.3
            case 12:
                reward += 0.6
            case 14:
                reward += 0.9
        reward -= .0001 * inputs[43]
        if i % frames_per_step == 0:
            output = run_neat_model(model, inputs)
            reward += take_action(output, ale)
            last_action = output
        else:
            reward += take_action(last_action, ale)
    if info:
        print(f'Total Reward: {reward}')
    # print(gc.collect())
    return reward


def show_frames(frames=100, info=False, frames_per_step=1, game='MontezumaRevenge', suppress=False,
                model=create_model(), activation=models.Sequential.predict) -> int:
    ale = ale_init(game, suppress)
    reward = 0
    last_action = 0
    number = 60 * 60  # Number of frames to capture
    output_file = 'output.avi'  # Output video file name
    
    # Get the screen size from the first frame
    img = ale.getScreenRGB()
    height, width, layers = img.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))
    last_life = False
    for i in range(frames):
        inputs = ale.getRAM().reshape(1, -1)[0]
        match inputs[58]:
            case 0:
                if inputs[55] == 0: last_life = True
                if inputs[55] > 0 and last_life:
                    print('Dead')
                    reward -= 100
                    break
            case _:
                reward += inputs[58] * .001
        match inputs[66]:
            case 13:
                reward += 0.3
            case 12:
                reward += 0.6
            case 14:
                reward += 0.9
        reward -= .0001 * inputs[43]
        if i % frames_per_step == 0:
            output = run_neat_model(model, inputs)
            reward += take_action(output, ale)
            last_action = output
        else:
            reward += take_action(last_action, ale)
        frame = ale.getScreenRGB()  # Capture the frame from ALE
        out.write(frame)  # Write the frame to the video file
        
        # If you want to display the frame while saving to video
        cv2.imshow('Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    if info:
        print(f'Total Reward: {reward}')
    # print(gc.collect())
    return reward


def game_eval(genomes, config, run_func=run_frames) -> None:
    kwargs = []
    for _, genome in genomes:
        net: neat.nn.FeedForwardNetwork = neat.nn.FeedForwardNetwork.create(genome, config)
        kwargs.append(
            {'frames': 60 * 60, 'frames_per_step': 3, 'model': net, 'activation': neat.nn.FeedForwardNetwork.activate})
    results = run_in_parallel(run_func, kwargs=kwargs, iterations=len(kwargs))
    for i, [_, genome] in enumerate(genomes):
        genome.fitness = results[i]


def main() -> None:
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run_neat(config_path, eval_func=game_eval, checkpoints=True, checkpoint_interval=1, checkpoint='neat-checkpoint-11',
             extra_inputs=show_frames)
    # t0 = time.perf_counter()
    # results = run_in_parallel(run_steps)
    # t1 = time.perf_counter()
    # t = t1 - t0
    # print(f'Time: {t//60:.0f}m {t%60:.1f}s')
    # print(f'Results: {results}')


if __name__ == "__main__":
    print('Program Started')
    main()
