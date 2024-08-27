import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from inference_sdk import InferenceHTTPClient
from PIL import Image
import sys
import serial

# Function to setup retry mechanism for HTTP requests
def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    return session

# Set up the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="jiKiP6e9LP32ERaOGQ1S"
)

# Function to zoom the image
def zoom_frame(frame, zoom_factor):
    height, width = frame.shape[:2]
    new_height, new_width = int(height / zoom_factor), int(width / zoom_factor)
    y1, x1 = (height - new_height) // 2, (width - new_width) // 2
    y2, x2 = y1 + new_height, x1 + new_width
    cropped_frame = frame[y1:y2, x1:x2]
    zoomed_frame = cv2.resize(cropped_frame, (width, height))
    return zoomed_frame

# Function to get the next file index for saving images
def get_next_file_index():
    index_file = 'D:/Artificial Intelligence/last_index.txt'
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index = int(f.read().strip()) + 1
    else:
        index = 1
    with open(index_file, 'w') as f:
        f.write(str(index))
    return index

def main():
    # Check if the script was called with the "start" argument
    if len(sys.argv) != 2 or sys.argv[1] != "start":
        print("Usage: python wasteclassifier.py start")
        return

    # Open a connection to the webcam
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Wait for 2 seconds before starting classification
    time.sleep(2)

    # Set zoom factor if needed
    zoom_factor = 2

    # Initialize variables to track frame and detection times
    last_detection_time = time.time()  # Track when waste was last detected

    # Initialize counters
    organic_count = 0
    recyclable_count = 0
    total_waste_processed = 0

    plt.ion()  # Turn on interactive mode for matplotlib

    # Initialize serial communication with Arduino
    arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)  # Replace 'COM4' with your Arduino port

    # Capture a single frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        cap.release()
        return

    # Zoom the frame
    frame = zoom_frame(frame, zoom_factor)

    # Get the next file index for saving
    file_index = get_next_file_index()
    file_path = f'D:/Artificial Intelligence/captured_frame_{file_index}.jpg'
    
    # Save the frame as an image file
    cv2.imwrite(file_path, frame)

    # Process the saved image with Roboflow API
    try:
        # Use Roboflow API to infer the frame
        result = CLIENT.infer(file_path, model_id="garbage-classification-3/2")

        # Process the inference result
        waste_detected = False
        has_biodegradable = False
        has_recyclable = False
        waste_class = None
        
        if result and "predictions" in result:
            for prediction in result["predictions"]:
                waste_class = prediction["class"]
                if waste_class == "BIODEGRADABLE":
                    has_biodegradable = True
                    organic_count += 1
                else:
                    has_recyclable = True
                    recyclable_count += 1

                # Draw bounding box and label with predicted class and accuracy
                x = int(prediction['x'])
                y = int(prediction['y'])
                width = int(prediction['width'])
                height = int(prediction['height'])
                confidence = prediction['confidence']
                color = (0, 0, 255) if waste_class == "BIODEGRADABLE" else (255, 0, 0)
                cv2.rectangle(frame, (x - width // 2, y - height // 2), (x + width // 2, y + height // 2), color, 2)
                text = f"{waste_class} {confidence * 100:.2f}%"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_x = x - text_size[0] // 2
                text_y = y + text_size[1] // 2
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Determine the signal to send
            if has_biodegradable:
                signal = 1  # Signal for biodegradable
                print("Sending signal: 1 for Organic")
                arduino.write(bytes(str(signal), 'utf-8'))
                time.sleep(2)  # Wait for 2 seconds
                arduino.write(bytes(str(2), 'utf-8'))  # Send idle signal
            elif has_recyclable:
                signal = 3  # Signal for recyclable
                print("Sending signal: 3 for Recyclable")
                arduino.write(bytes(str(signal), 'utf-8'))
                time.sleep(2)  # Wait for 2 seconds
                arduino.write(bytes(str(2), 'utf-8'))  # Send idle signal
            else:
                signal = 2  # Default idle signal if no waste detected
                print("Sending idle signal: 2")
                arduino.write(bytes(str(signal), 'utf-8'))
            
            last_detection_time = time.time()  # Update last detection time

            # Print detected waste class
            if waste_class:
                print(f"Waste Class Detected: {waste_class}")

        # Check if it's time to send an idle signal
        current_time = time.time()
        if current_time - last_detection_time >= 5:
            # Send the idle signal to Arduino
            print("Sending idle signal: 2")
            arduino.write(bytes(str(2), 'utf-8'))
            last_detection_time = current_time  # Update last detection time

        # Update total waste processed count
        total_waste_processed = organic_count + recyclable_count

        # Display counters on the frame
        cv2.putText(frame, f"Organic: {organic_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Recyclable: {recyclable_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Waste Processed: {total_waste_processed}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the updated frame with detection details
        updated_file_path = f'D:/Artificial Intelligence/processed_frame_{file_index}.jpg'
        cv2.imwrite(updated_file_path, frame)

        # Open the updated image
        img = Image.open(updated_file_path)
        img.show()

        # Convert frame to RGB format for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the live video stream using matplotlib
        plt.imshow(frame_rgb)
        plt.title("Live Video Stream")
        plt.draw()
        plt.pause(0.001)  # Adjust the pause duration if needed

    except Exception as e:
        print(f"Processing failed: {e}")

    # Ensure Arduino is in idle state before halting
    print("Sending final idle signal: 2")
    arduino.write(bytes(str(2), 'utf-8'))
    time.sleep(2)  # Wait for 2 seconds to ensure idle signal is processed

    # Release the webcam and close all resources
    cap.release()
    plt.close()  # Close matplotlib window

    # Print final counts
    print(f"Final Counts - Organic: {organic_count}, Recyclable: {recyclable_count}, Total Waste Processed: {total_waste_processed}")

# Run the main function
if __name__ == "__main__":
    main()
