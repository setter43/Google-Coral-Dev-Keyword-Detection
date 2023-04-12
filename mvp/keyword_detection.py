# python keyword_detection.py --model soundclassifier_with_metadata.tflite --labels label.txt --duration 1
# Logitech USB Headset: Audio (hw:2,0)


import numpy as np
import sounddevice as sd
import resampy
import tflite_runtime.interpreter as tflite
import argparse

def load_labels(label_file):
    with open(label_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def predict(input_data, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def record_audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = np.append(audio_buffer, indata)
    
    
    
    
def continuous_listen(window_size, model_path, labels_path):
    global audio_buffer
    audio_buffer = np.array([])

    # Load the TensorFlow Lite model with the TPU
    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[
            tflite.load_delegate('libedgetpu.so.1')
        ]
    )
    interpreter.allocate_tensors()
    input_shape = interpreter.get_input_details()[0]['shape']

    # Load labels
    labels = load_labels(labels_path)

    # Set the USB microphone as the default device
    device_name =  'Logitech USB Headset: Audio (hw:2,0)'  # Update this with the correct device name
    devices = sd.query_devices()
    for device in devices:
        if device['name'] == device_name and device['max_input_channels'] > 0:
            sd.default.device = device['index']
            break
    else:
        print("No input device matching was found.")
        return

    with sd.InputStream(samplerate=fs, channels=1, callback=record_audio_callback):
        while True:

            if len(audio_buffer) >= window_size:
                window = audio_buffer[:window_size]
                audio_buffer = audio_buffer[window_size:]

                # Resample the recorded audio to match the model's input size
                input_length = input_shape[1]

                resampled_audio = resampy.resample(window, 16000, input_length)
                # Ensure the audio data is in the correct shape and data type
                input_data = np.array(resampled_audio, dtype=np.float32).reshape(input_shape)

                output_data = predict(input_data, interpreter)
                detected_keyword = np.argmax(output_data)
                if output_data[0, detected_keyword] > 0.5:  # Adjust the threshold as needed
                    print('Detected keyword:', labels[detected_keyword])

                #0 Background Noise
                # 1 Fire Mission
                # 2 Hello
                # 3 Over
                # 4 Zero
                
                # if (labels[detected_keyword] == labels[0]):#Background Noise
                #     print("Fizz Buzz " + labels[0])

                # if (labels[detected_keyword] == labels[1]):#Fire Mission
                #     print("ALERT!!!" + labels[1])
                #     print("Send Over!")

                # if(labels[detected_keyword] == labels[2]):#Hello
                #     print("I heard " + labels[2])

                label_actions = {
                    labels[0]: lambda: print("Fizz Buzz " + labels[0]),
                    labels[1]: lambda: (
                        print("ALERT!!!" + labels[1]),
                        print("Send Over!")
                    ),
                    labels[2]: lambda: print("I heard " + labels[2]),
                    # Add more labels and corresponding actions here
                }

                action = label_actions.get(labels[detected_keyword])
                if action:
                    action()



if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="soundclassifier_with_metadata.tflite", help='Path of the TFLite model.', required=False)
    parser.add_argument('--labels', default="label.txt", help='Path of the label file.', required=False)
    parser.add_argument('--duration', help='Duration of each recording in seconds.', type=float, default=1.0, required=False)
    args = parser.parse_args()
    model_path=args.model
    labels_path=args.labels
    duration=args.duration
    print("Duration :",duration)
    # Parameters for the sliding window
    fs = 16000
    window_size =  int(duration * fs) # 1 second of audio data
    continuous_listen(window_size, model_path, labels_path)


##python3 keyword_detection.py --model soundclassifier_with_metadata.tflite --labels label.txt --duration 1