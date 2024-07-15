from ale_py import ALEInterface, roms
import cv2
import gym
import neat
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models


def ale_init(game):
    # Initialize ALE
    ale = ALEInterface()
    
    # Use the built-in ROM
    rom = getattr(roms, game)
    ale.loadROM(rom)
    
    return ale


def load_tflite_model():
    # Load the TFLite model and allocate tensors
    model_path: str = '2.tflite'
    interpreter: tf.lite.Interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details: list = interpreter.get_input_details()
    output_details: list = interpreter.get_output_details()
    return input_details, output_details, interpreter


def load_image(img, input_details):
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


def display_bounding_boxes(img, boxes, classes, scores, scale_factor=3):
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


def run_tflite(img, input_details, output_details, interpreter, info=False, display=False):
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


def create_model():
    model = models.Sequential([
        layers.Input(shape=(60,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(9, activation='softmax')
    ])
    return model


def run_model(model, input1, input2, input3):
    # Ensure all inputs are 2D arrays
    input1 = np.expand_dims(input1, axis=0) if len(input1.shape) == 1 else input1
    input2 = np.expand_dims(input2, axis=0) if len(input2.shape) == 1 else input2
    input3_flattened = input3.reshape(1, -1) if len(input3.shape) == 2 else input3.reshape(1, -1)
    
    # Concatenate inputs
    concatenated_inputs = np.concatenate([input1, input2, input3_flattened], axis=1)
    
    # Ensure the input shape is (1, 60)
    concatenated_inputs = concatenated_inputs.reshape((1, 60))
    
    # Run the model
    output = model.predict(concatenated_inputs, verbose=0)
    return list(output[0])


def take_action(output, ale):
    # Take an action and get the new state
    legal_actions = ale.getLegalActionSet()
    action_index = output.index(max(output))
    action = legal_actions[action_index]  # Choose an action (e.g., NOOP)
    reward = ale.act(action)
    return reward


def main():
    input_details, output_details, interpreter = load_tflite_model()
    ale = ale_init('MontezumaRevenge')
    print(ale.getRAM())
    model = create_model()

    steps = 1_000
    reward = 0
    for i in range(steps):
        input1, input2, input3 = run_tflite(ale.getScreenRGB(), input_details, output_details, interpreter)
        output = run_model(model, input1, input2, input3)
        reward += take_action(output, ale)
    print(f'Total Reward: {reward}')
    run_tflite(ale.getScreenRGB(), input_details, output_details, interpreter, info=True, display=True)
    

if __name__ == "__main__":
    main()
