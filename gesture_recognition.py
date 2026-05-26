import tkinter as tk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import cv2
from PIL import Image, ImageTk  # For converting OpenCV image to Tkinter compatible format
import threading
from collections import deque, Counter  # ✅ For smoothing

# 1.Thumbs down
# 2.Victory
# 3.Thumbs up
# 4.Pointing
# 5.fist closed 
# 6.open palm

# STEP 1: Initialize GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Global variables
recognition_running = False
frame_to_process = None
processed_frame = None
lock = threading.Lock()

# ✅ Gesture smoothing history
gesture_history = deque(maxlen=15)

# Function to get stable gesture
def get_stable_gesture(current_gesture):
    if current_gesture:
        gesture_history.append(current_gesture.category_name)

    if len(gesture_history) == gesture_history.maxlen:
        most_common, count = Counter(gesture_history).most_common(1)[0]
        if count > 0.6 * gesture_history.maxlen:  # at least 60% agreement
            return most_common
    return None

# Function to annotate the frame with gesture and landmarks
def annotate_frame(frame, top_gesture, hand_landmarks):
    h, w, _ = frame.shape
    
    # Display the recognized gesture
    if top_gesture:
        text = f'Gesture: {top_gesture}'
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw hand landmarks
    if hand_landmarks:
        for landmark_set in hand_landmarks:
            for landmark in landmark_set:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

# Function to process the frames for gesture recognition
def process_gestures():
    global frame_to_process, processed_frame, recognition_running

    while recognition_running:
        if frame_to_process is not None:
            with lock:
                frame = frame_to_process.copy()

            # Convert frame to MediaPipe image format (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Recognize gestures in the current frame
            recognition_result = recognizer.recognize(mp_image)

            # Extract top gesture and hand landmarks
            top_gesture = recognition_result.gestures[0][0] if recognition_result.gestures else None

            # ✅ Confidence filter
            if top_gesture and top_gesture.score < 0.7:
                top_gesture = None

            hand_landmarks = recognition_result.hand_landmarks if recognition_result.hand_landmarks else []

            # ✅ Apply smoothing
            stable_gesture = get_stable_gesture(top_gesture) if top_gesture else None

            # Annotate frame with results
            annotate_frame(frame, stable_gesture if stable_gesture else None, hand_landmarks)

            with lock:
                processed_frame = frame

# Function to capture frames and update Tkinter window
def update_frame():
    global frame_to_process, processed_frame

    ret, frame = cap.read()
    if not ret:
        print("Unable to retrieve frame. Exiting...")
        return

    # Resize for faster processing (optional)
    frame = cv2.resize(frame, (640, 480))

    with lock:
        frame_to_process = frame

    # If processed frame is available, display it
    with lock:
        display_frame = processed_frame if processed_frame is not None else frame

    # Convert frame (OpenCV format) to Image format for Tkinter
    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

    # Update the image on the Tkinter label
    label_img.imgtk = imgtk
    label_img.configure(image=imgtk)

    # Call update_frame again after 10 ms
    label_img.after(10, update_frame)

# Start recognition
def start_recognition():
    global recognition_running
    recognition_running = True
    start_button.config(state=tk.DISABLED, bg="#4CAF50", fg="white")
    stop_button.config(state=tk.NORMAL, bg="#F44336", fg="white")

    # Start a separate thread for gesture processing
    threading.Thread(target=process_gestures, daemon=True).start()

# Stop recognition
def stop_recognition():
    global recognition_running
    recognition_running = False
    stop_button.config(state=tk.DISABLED, bg="#9E9E9E", fg="black")
    start_button.config(state=tk.NORMAL, bg="#4CAF50", fg="white")

# Exit application
def exit_app():
    global recognition_running
    recognition_running = False
    cap.release()
    root.quit()

# Initialize the Tkinter window
root = tk.Tk()
root.title("Hand Gesture Recognition")
root.geometry("800x600")  # Set window size
root.configure(bg="#2196F3")  # Set background color

# Title label (above the live stream)
title_label = tk.Label(root, text="Hand Gesture Recognition", font=("Helvetica", 16), bg="#2196F3", fg="white")
title_label.pack(pady=10)

# Label to display available gestures
gesture_label = tk.Label(
    root,
    text="Gestures Names: Thumbs down, Victory, Thumbs up, Pointing",
    font=("Helvetica", 12),
    bg="#2196F3",
    fg="yellow"
)
gesture_label.pack(pady=5)

# Create a Label widget to display video frames (Live stream)
label_img = tk.Label(root, bg="black")
label_img.pack(pady=20)

# Create control buttons (Start, Stop, Exit)
frame_controls = tk.Frame(root, bg="#2196F3")
frame_controls.pack(pady=10)

start_button = tk.Button(frame_controls, text="Start Recognition", command=start_recognition, width=12, height=2, bg="#4CAF50", fg="white", font=("Helvetica", 12))
start_button.grid(row=0, column=0, padx=10)

stop_button = tk.Button(frame_controls, text="Stop Recognition", command=stop_recognition, width=18, height=2, bg="#F44336", fg="white", font=("Helvetica", 12))
stop_button.grid(row=0, column=1, padx=10)

exit_button = tk.Button(frame_controls, text="Exit", command=exit_app, width=12, height=2, bg="#9E9E9E", fg="black", font=("Helvetica", 12))
exit_button.grid(row=0, column=2, padx=10)

# Open the webcam using OpenCV
cap = cv2.VideoCapture(0)

# Check if camera is opened successfully
if not cap.isOpened():
    print("Error: Camera not accessible.")
    root.quit()

# Initialize gesture recognition state
recognition_running = False

# Start the update loop immediately
update_frame()

# Start the Tkinter event loop
root.mainloop()

# Release resources when done
cap.release()
cv2.destroyAllWindows()
