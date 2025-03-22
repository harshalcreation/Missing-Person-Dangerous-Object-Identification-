import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, simpledialog
import threading
from threading import Thread
import pygame
import speech_recognition as sr
import os
import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np
import csv
from datetime import datetime
from PIL import Image, ImageTk

stop_processing_flag = False

# Global variables
known_face_encodings = []
known_face_names = []
dangerous_objects = ["knife", "gun", "bomb"]
stop_processing_flag = False
video_processing_running = False
recognized_faces = {}
activity_log = []

# Initialize pygame for sound
pygame.mixer.init()

# Load known faces
def load_known_faces():
    global known_face_encodings, known_face_names
    print("Loading known faces...")
    face_folder = "faces_folder"
    for filename in os.listdir(face_folder):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(face_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(filename.split(".")[0])
                print(f"Loaded {filename.split('.')[0]} from {image_path}")
            else:
                print(f"No face found in {image_path}. Skipping.")

# Function to play alert sound
def play_alert_sound():
    pygame.mixer.music.load("alert.mp3")  # Path to the alert sound file
    pygame.mixer.music.play()

# Log recognized faces and objects to CSV
def log_to_csv(filename, data):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)

# Start surveillance
def start_surveillance():
    global stop_processing_flag
    stop_processing_flag = False
    print("Surveillance started!")

    video_capture = cv2.VideoCapture(0)
    model = YOLO('yolov5n.pt')  # Use the lightweight YOLO model
    class_names = model.names
    frame_skip = 2  # Process every 2nd frame

    frame_count = 0
    while not stop_processing_flag:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        if frame_count % frame_skip == 0:
            process_frame(frame, model, class_names)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    video_capture = None 

# Process a single frame for both face recognition and object detection
def process_frame(frame, model, class_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Recognition
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            name = known_face_names[best_match_index]
            log_to_csv('recognized_faces.csv', [name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            play_alert_sound()  # Alert sound for recognized face

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Object Detection
    results = model.predict(frame, conf=0.5)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(float(box.conf[0]), 2)
            class_id = int(box.cls[0])
            class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown ({class_id})"

            label = f"{class_name} ({confidence:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if class_name in dangerous_objects:
                cv2.putText(frame, "DANGER!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                play_alert_sound()
                log_to_csv('recognized_objects.csv', [class_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

    cv2.imshow("Processing", frame)

# Stop surveillance
def stop_processing():
    global stop_processing_flag
    stop_processing_flag = True
    print("Surveillance will stop shortly!")


# Add face to system
def add_face():
    name = simpledialog.askstring("Input", "Enter the name of the person:")
    if name:
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg")])
        if file_path:
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                save_path = os.path.join("faces_folder", f"{name}.jpg")
                cv2.imwrite(save_path, cv2.imread(file_path))
                print(f"Added face for {name}!")
                log_to_csv('activity_log.csv', [f"Added face: {name}", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
def start_surveillance_threaded():
    # Start surveillance in a separate thread to prevent GUI freezing
    surveillance_thread = Thread(target=start_surveillance)
    surveillance_thread.start()
# Manage objects
def manage_objects():
    action = simpledialog.askstring("Choose Action", "Do you want to 'add' or 'delete' a dangerous object?")
    if action:
        action = action.lower()
        if action == "add":
            dangerous_object = simpledialog.askstring("Input", "Enter a dangerous object to log (e.g., knife, gun, bomb):")
            if dangerous_object and dangerous_object not in dangerous_objects:
                dangerous_objects.append(dangerous_object)
                log_to_csv('activity_log.csv', [f"Added dangerous object: {dangerous_object}", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                print(f"Added dangerous object: {dangerous_object}")
            elif dangerous_object in dangerous_objects:
                print(f"{dangerous_object} is already in the list.")
        elif action == "delete":
            dangerous_object = simpledialog.askstring("Input", "Enter a dangerous object to delete:")
            if dangerous_object in dangerous_objects:
                dangerous_objects.remove(dangerous_object)
                log_to_csv('activity_log.csv', [f"Deleted dangerous object: {dangerous_object}", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                print(f"Deleted dangerous object: {dangerous_object}")
            else:
                print(f"{dangerous_object} not found in the list.")
        else:
            print("Invalid action! Please choose 'add' or 'delete'.")

# View activity log
def view_activity_log():
    log_window = tk.Toplevel()
    log_window.title("Activity Log")
    log_text = tk.Text(log_window, height=20, width=80)
    log_text.pack(padx=10, pady=10)

    # Read the log file and display its contents
    try:
        with open("activity_log.csv", "r") as log_file:
            log_text.insert(tk.END, log_file.read())
    except FileNotFoundError:
        log_text.insert(tk.END, "No activity log found.")
    
    log_text.config(state=tk.DISABLED)

# Process recorded video
# def process_recorded_video():
#     video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
#     if video_path:
#         print(f"Processing video: {video_path}")
#         video_capture = cv2.VideoCapture(video_path)
#         model = YOLO('yolov5n.pt')  # Use the lightweight YOLO model
#         class_names = model.names

#         while video_capture.isOpened():
#             ret, frame = video_capture.read()
#             if not ret:
#                 break

#             process_frame(frame, model, class_names)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         video_capture.release()
#         cv2.destroyAllWindows()
        
# def stop_recorded_video_processing():
#     global recorded_video_processing_flag
#     recorded_video_processing_flag = False
#     print("Stopping recorded video processing...")
# Process a recorded video
def process_recorded_video():
    global recorded_video_processing_flag
    recorded_video_processing_flag = True  # Start processing

    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if not video_path:
        print("No video file selected.")
        recorded_video_processing_flag = False  # Reset the flag if no file selected
        return

    def process_video():
        global recorded_video_processing_flag

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video file.")
            recorded_video_processing_flag = False
            return

        model = YOLO('yolov5n.pt')
        class_names = model.names

        while cap.isOpened() and recorded_video_processing_flag:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read frame from video.")
                break  # Exit loop if no frame is read

            if frame.size == 0:
                print("Empty frame received.")
                continue  # Skip this frame

            processed_frame = process_frame(frame, model, class_names)
            if processed_frame is not None and processed_frame.size > 0:
                cv2.imshow("Processed Video", processed_frame)
            else:
                print("Processed frame is invalid.")

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow manual exit via 'q'
                break

        cap.release()
        cv2.destroyAllWindows()
        recorded_video_processing_flag = False  # Reset the flag

    threading.Thread(target=process_video, daemon=True).start()


# Stop recorded video processing
def stop_recorded_video_processing():
    global recorded_video_processing_flag
    recorded_video_processing_flag = False  # Set the flag to False
    print("Stop signal sent for recorded video processing.")


# Start voice assistant automatically when the program is run
def start_voice_assistant():
    voice_thread = threading.Thread(target=listen_for_commands)
    voice_thread.daemon = True
    voice_thread.start()

# Listen for commands using voice recognition
def listen_for_commands():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Voice Assistant listening for commands...")

        while True:
            try:
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                print(f"Command received: {command}")

                if "start" in command:
                    print("Starting surveillance...")
                    start_surveillance()
                elif "stop" in command:
                    print("Stopping surveillance...")
                    stop_processing()
                elif "add" in command:
                    print("Adding face...")
                    add_face()
                elif "manage" in command:
                    print("Managing dangerous objects...")
                    manage_objects()
                elif "view" in command:
                    print("Viewing activity log...")
                    view_activity_log()
                elif "exit" in command:
                    print("Exiting...")
                    break
            except sr.UnknownValueError:
                print("Sorry, I didn't catch that. Please repeat.")
            except sr.RequestError:
                print("Sorry, the speech recognition service is down.")
                
                
def load_activity_log(text_widget):
    log_path = "activity_log.csv"
    if os.path.exists(log_path):
        with open(log_path, "r") as log_file:
            content = log_file.read()
            text_widget.config(state=tk.NORMAL)
            text_widget.delete("1.0", tk.END)
            text_widget.insert(tk.END, content)
            text_widget.config(state=tk.DISABLED)
    else:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, "No activity log found.")
        text_widget.config(state=tk.DISABLED)


def create_gui():
    root = tk.Tk()
    root.title("Surveillance System")
    root.geometry("800x600")

    # Header with Blinkers logo
    header = tk.Frame(root, bg="white", height=100)
    header.pack(fill=tk.X, side=tk.TOP)

    logo_path = "blinkers_logo.png"  # Path to the Blinkers logo
    logo_image = Image.open(logo_path)
    logo_image = logo_image.resize((80, 80), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = tk.Label(header, image=logo_photo, bg="white")
    logo_label.image = logo_photo
    logo_label.pack(side=tk.LEFT, padx=10, pady=10)

    header_title = tk.Label(header, text="Surveillance System", font=("Arial", 24, "bold"), bg="white")
    header_title.pack(side=tk.LEFT, padx=20)
    

    # Surveillance feed in the middle
    # feed_frame = tk.Frame(root, bg="black", height=400, width=600)
    # feed_frame.pack(expand=True)

    # feed_label = tk.Label(feed_frame, text="Camera Feed", font=("Arial", 16), fg="white", bg="black")
    # feed_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # # Buttons at the bottom
    # button_frame = tk.Frame(root)
    # button_frame.pack(side=tk.BOTTOM, pady=20)

    # buttons = [
    #     ("Start Surveillance", start_surveillance),
    #     ("Stop Surveillance", stop_processing),
    #     ("Process Recorded Video", process_recorded_video),
    #     ("Add Face", add_face),
    #     ("Manage Dangerous Objects", manage_objects),
    #     ("View Activity Log", view_activity_log),
    # ]

    # for text, command in buttons:
    #     tk.Button(button_frame, text=text, font=("Arial", 12), command=command).pack(side=tk.LEFT, padx=10)

    # # Start voice assistant automatically when the program runs
    # start_voice_assistant()

    # root.mainloop()
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill=tk.BOTH)

    # Tab 1: Real-Time Surveillance
    tab1 = tk.Frame(notebook)
    notebook.add(tab1, text="Real-Time Surveillance")

    feed_label = tk.Label(tab1, bg="black")
    feed_label.pack(expand=True, fill=tk.BOTH)

    tab1_buttons_frame = tk.Frame(tab1)
    tab1_buttons_frame.pack(side=tk.BOTTOM, pady=20)

    tk.Button(tab1_buttons_frame, text="Start Surveillance", command=start_surveillance_threaded).pack(side=tk.LEFT, padx=10)
    tk.Button(tab1_buttons_frame, text="Stop Surveillance", command=stop_processing).pack(side=tk.LEFT, padx=10)
    tk.Button(root, text="Add Face", font=("Arial", 14), command=add_face).pack(pady=10)


    # Tab 2: Recorded Video Processing
    tab2 = tk.Frame(notebook)
    notebook.add(tab2, text="Recorded Video Processing")

    tk.Button(tab2, text="Process Video", command=process_recorded_video).pack(pady=20)
    tk.Button(tab2, text="Stop Video Processing", command=stop_recorded_video_processing).pack(pady=10)

    # Tab 3: View Activity Log
    tab3 = tk.Frame(notebook)
    notebook.add(tab3, text="View Activity Log")

    activity_log_text = tk.Text(tab3, state=tk.DISABLED, wrap=tk.WORD)
    activity_log_text.pack(expand=True, fill=tk.BOTH)

    tk.Button(tab3, text="Load Activity Log", command=lambda: load_activity_log(activity_log_text)).pack(pady=10)

    root.mainloop()

# Start the program
if __name__ == "__main__":
    load_known_faces()
    create_gui()


