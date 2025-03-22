import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, simpledialog,messagebox
import threading
from threading import Thread
from playsound import playsound
import speech_recognition as sr
import os
import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np
import csv
from datetime import datetime
from PIL import Image, ImageTk
import mediapipe as mp
from geopy.geocoders import Nominatim
import requests
import cloudinary
import cloudinary.uploader
from pymongo import MongoClient
from SIH2 import estimate_height
from SIH2 import mp_draw
from twilio.rest import Client
from datetime import datetime, timedelta
import torch


# Dictionary to track the last detection time for each person
last_detected_time = {}
prev_gray = None
video_capture = None


# Twilio credentials
TWILIO_ACCOUNT_SID = "AC25d805c39b7334716de6e686585fe65e"
TWILIO_AUTH_TOKEN = "81f834f1471af05b5ef9d401511e4039"
TWILIO_PHONE = "+16467605071"
ALERT_PHONE = "+918766830188"
TWILIO_SMS = "+16467605071"
ALERT_SMS = "+918766830188"

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

cloudinary.config(
    cloud_name="dcvsavnx3",
    api_key="399294449426652",
    api_secret="7ZJuKbNHl0KsWPMAcKb4OWDH-j8"
)

# Mailgun API Credentials
MAILGUN_DOMAIN = "sandbox963745399947467e80086c44142a1697.mailgun.org"
MAILGUN_API_KEY = "98ed24216c0e33944a27d0946c813059-ac3d5f74-ad3a53e6"  # Your actual API key
SENDER_EMAIL = f"mailgun@{MAILGUN_DOMAIN}"
RECIPIENT_EMAIL = "harshalborkar501@gmail.com"

def send_email_with_image(image_path,subject="⚠️ Suspicious Activity Detected!", message="Suspicious activity detected. See the attached image."):
    with open(image_path, "rb") as image_file:
        response = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            files={"attachment": ("screenshot.jpg", image_file, "image/jpeg")},
            data={
                "from": f"Mailgun Alerts <{SENDER_EMAIL}>",
                "to": RECIPIENT_EMAIL,
                "subject": "⚠️ Suspicious Activity Detected!",
                "text": "Suspicious activity detected by the surveillance system. See attached image."
            }
        )
    if response.status_code == 200:
        print("✅ Email with image sent successfully!")
    else:
        print(f"❌ Email failed! Status Code: {response.status_code}")
        print(f"Response: {response.json()}")


# MongoDB credentials
MONGO_URI = "mongodb+srv://alhanmsiddique:jU0FQ5M89o3hlL1w@cluster0.o7nko.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "test"
COLLECTION_NAME = "listings"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
documents=collection.find()

for doc in documents:
    Title=doc.get("title")
    Height=doc.get("height")
    Weight=doc.get("weight")
    Occupations=doc.get("occupations")
    Description=doc.get("description")
    Contact=doc.get("price")
    Location="Priyadarshini College of Engineering,Nagpur"
    Country=doc.get("country")
    Reviews=doc.get("reviews")
 
stop_processing_flag = False

# Global variables
known_face_encodings = []
known_face_names = []
dangerous_objects = ["knife", "gun", "bomb"]
stop_processing_flag = False
video_processing_running = False
recognized_faces = {}
activity_log = []

# Global variable to keep track of the last screenshot ID
last_screenshot_id = 0
screenshot_folder = "screenshots"

def take_screenshot(label, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshots/{label}_{timestamp}.jpg"
    
    os.makedirs("screenshots", exist_ok=True)
    cv2.imwrite(filename, frame)
    
    print(f"Screenshot saved: {filename}")
    
    return filename

    # Save the screenshot with a sequential number
    screenshot_path = os.path.join(screenshot_folder, f"{screenshot_counter}.jpg")
    cv2.imwrite(screenshot_path, frame)
    print(f"Screenshot saved: {screenshot_path}")

    # Update the last screenshot time and increment the counter
    last_screenshot_time[name] = current_time
    screenshot_counter += 1    


# Create a directory for screenshots if it does not exist
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def load_known_faces():
    global known_face_encodings, known_face_names
    print("Connecting to MongoDB...")

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Fetch all documents containing images from MongoDB
    documents = collection.find({}, {"image": 1, "title": 1})
    print("Fetching known faces from MongoDB...")
    document_count = collection.count_documents({})
    print(f"Total documents found: {document_count}")

    # Create folder if not exists
    faces_folder = "faces_folder"
    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)

    # Load images from MongoDB
    for doc in documents:
        try:
            print(f"Processing document: {doc}")
            if "image" in doc and "url" in doc["image"]:
                image_url = doc["image"]["url"]
                title = doc.get("title", "Unknown")

                # Download the image
                image_path = os.path.join(faces_folder, f"{title}.jpg")
                response = requests.get(image_url, stream=True)

                if response.status_code == 200:
                    with open(image_path, "wb") as image_file:
                        for chunk in response.iter_content(1024):
                            image_file.write(chunk)

                    print(f"Image downloaded for {title}")

            else:
                print(f"Invalid document format: {doc}")

        except Exception as e:
            print(f"Error processing image for {doc.get('title', 'Unknown')}: {e}")

    # Load images from faces_folder
    print("Loading images from faces_folder...")
    for filename in os.listdir(faces_folder):
        image_path = os.path.join(faces_folder, filename)

        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(os.path.splitext(filename)[0])
                print(f"Successfully processed {filename}")

            else:
                print(f"No face found in {filename}. Skipping.")

        except Exception as e:
            print(f"Error loading image {filename}: {e}")

    # Close MongoDB connection
    client.close()

    print(f"Total faces loaded: {len(known_face_encodings)}")   
     
# Function to play alert sound
def play_alert_sound():
    playsound("alert.mp3", block=False) 
    
# def get_geolocation():
#     try:
#         # Get public IP address
#         ip = requests.get('https://api.ipify.org').text
#         # Use IP to get location
#         response = requests.get(f'https://ipinfo.io/{ip}/json').json()
#         location = response.get('loc', None)  # Get latitude and longitude
#         if location:
#             lat, lon = location.split(',')
#             geolocator = Nominatim(user_agent="geoapiExercises")
#             address = geolocator.reverse(f"{lat}, {lon}")
#             return address.address
#         else:
#             return "Location not available"
#     except Exception as e:
#         print(f"Geolocation error: {e}")
#         return "Location not available"
    
# Log recognized faces and objects to CSV
def log_to_csv(filename, data):
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    file_path = os.path.join(log_directory, filename)

    try:
        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)
        print(f"Logged data to {file_path}: {data}")
    except Exception as e:
        print(f"Error logging data to {file_path}: {e}")

        
# Start surveillance
def start_surveillance():
    global stop_processing_flag, video_capture
    stop_processing_flag = False
    print("Surveillance started!")

    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)  # Ensure only one capture object is active
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set capture width
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set capture height

    model = YOLO('yolov5n.pt')  # Use the lightweight YOLO model
    class_names = model.names
    frame_skip = 1  # Process every 2nd frame

    frame_count = 0
    while not stop_processing_flag:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        if frame_count % frame_skip == 0:
            process_frame(frame, model, class_names)  # Ensure this function doesn't open a new window

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    
def manage_dangerous_objects():
    def add_object():
        new_object = simpledialog.askstring("Add Object", "Enter the name of the new dangerous object:")
        if new_object and new_object not in dangerous_objects:
            dangerous_objects.append(new_object)
            messagebox.showinfo("Success", f"{new_object} added to dangerous objects.")

    def remove_object():
        object_to_remove = simpledialog.askstring("Remove Object", "Enter the name of the object to remove:")
        if object_to_remove in dangerous_objects:
            dangerous_objects.remove(object_to_remove)
            messagebox.showinfo("Success", f"{object_to_remove} removed from dangerous objects.")
        else:
            messagebox.showwarning("Not Found", f"{object_to_remove} not found in dangerous objects.")

    def view_objects():
        objects_list = "\n".join(dangerous_objects)
        messagebox.showinfo("Dangerous Objects", f"Current dangerous objects:\n{objects_list}")

    manage_window = tk.Toplevel()
    manage_window.title("Manage Dangerous Objects")
    manage_window.geometry("400x300")

    tk.Button(manage_window, text="Add Object", font=("Arial", 14), command=add_object).pack(pady=10)
    tk.Button(manage_window, text="Remove Object", font=("Arial", 14), command=remove_object).pack(pady=10)
    tk.Button(manage_window, text="View Objects", font=("Arial", 14), command=view_objects).pack(pady=10)


# Process a single frame for both face recognition and object detection
def process_frame(frame, model, class_names):
    global prev_gray
     
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = datetime.now()

        # Face Recognition
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        current_time = datetime.now()
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                name = known_face_names[best_match_index]
                
                if name in last_detected_time and (current_time - last_detected_time[name]).total_seconds() < 60:
                    continue  # Skip if the last recognition was less than one minute ago

                last_detected_time[name] = current_time
                log_to_csv('recognized_faces.csv', [name, current_time.strftime('%Y-%m-%d %H:%M:%S')])
                play_alert_sound()
                send_sms_alert(name)
                
            else:  # Loitering Detection for Unknown Persons
                if name in last_detected_time:
                    time_spent = (current_time - last_detected_time[name]).total_seconds()
                    if time_spent > 120:  # If loitering for more than 2 minutes
                        cv2.putText(frame, "Loitering Detected!", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        play_alert_sound()
                        log_to_csv('behavior_log.csv', ["Loitering detected", current_time.strftime('%Y-%m-%d %H:%M:%S')])

                last_detected_time[name] = current_time 
                
                
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # Object Detection using YOLO
        results = model.predict(rgb_frame)
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
                    alert_message = f"Dangerous object detected: {class_name}"
                    play_alert_sound()
                    image_path = take_screenshot(class_name, frame)
                    send_email_with_image(image_path)
                    # image_path = take_screenshot(name, frame)
                    send_sms_alert1(class_name)
                    send_email_with_image(image_path, f"Dangerous Object Detected: {class_name}!",
                                          f"Surveillance detected a {class_name} in the monitored area.")
                    log_to_csv('recognized_objects.csv', [class_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])


        # MediaPipe Pose Estimation
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            # mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Example behavior detection: Checking if hands are above the head
            left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            throat_y = nose.y + 0.05  # Approximate throat position

            # **Choking Attempt Detection**: Hands close to neck
            if (abs(left_wrist.y - throat_y) < 0.05 and abs(right_wrist.y - throat_y) < 0.05):
                choking_counter += 1
            else:
                choking_counter = 0  # Reset counter if hands move away

            # If hands stay near throat for 3 seconds -> Alert!
            if choking_counter > 30:  # Assuming ~10 FPS, 30 frames ≈ 3 seconds
                cv2.putText(frame, "⚠️ Choking Attempt Detected! ⚠️", (50, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                play_alert_sound()
                log_to_csv('behavior_log.csv', ["Choking Attempt Detected", current_time.strftime('%Y-%m-%d %H:%M:%S')])
                send_sms_alert("Possible strangulation attempt detected!")
                choking_counter = 0  # Reset after alert


            if left_wrist.y < nose.y and right_wrist.y < nose.y:
                cv2.putText(frame, "Hands above head!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                log_to_csv('behavior_log.csv', ["Hands above head detected", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                

        # Display the processed frame in the single window
        cv2.imshow("Surveillance", frame)
        
    except Exception as e:
        print(f"Error processing frame: {e}")

def send_sms_alert1(name):
    message_body =f"Alert: {name} : Dangerous object detected"
    twilio_client.messages.create(
        body=message_body,
        from_=TWILIO_SMS,
        to=ALERT_SMS
    )
def send_sms_alert(name):
    message_body =f"Alert: {name} has been identified by the surveillance system."
    twilio_client.messages.create(
        body=message_body,
        from_=TWILIO_SMS,
        to=ALERT_SMS
    )
    print(f"SMS alert sent about {name}")

def detect_height():
    def process_height():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to access the camera.")
            return

        messagebox.showinfo("Instructions", "Ensure your full body is visible in the frame.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)

            if pose_results.pose_landmarks:
                mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                height = estimate_height(frame, pose_results)
                if height:
                    cv2.putText(frame, f"Height: {height} cm", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Height Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Start height detection in a new thread
    height_thread = threading.Thread(target=process_height)
    height_thread.start()
    
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

                # Save the local file
                save_path = os.path.join("faces_folder", f"{name}.jpg")
                cv2.imwrite(save_path, cv2.imread(file_path))

                # Upload to Cloudinary
                try:
                    response = cloudinary.uploader.upload(file_path, folder="faces_folder")
                    cloudinary_url = response.get("secure_url")
                    print(f"Uploaded to Cloudinary: {cloudinary_url}")

                    # Log the upload to CSV
                    log_to_csv('activity_log.csv', [
                        f"Added face: {name} (Cloudinary URL: {cloudinary_url})",
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ])
                    messagebox.showinfo("Success", f"Face for {name} uploaded successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to upload image: {e}")
            else:
                messagebox.showwarning("Warning", "No face found in the selected image.")

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

        model = YOLO('yolov5n.pt')  # Initialize model
        model.to('cuda' if torch.cuda.is_available() else 'gpu')  # Set device

        class_names = model.names
        frame_skip = 2  # Process every 2nd frame for speed
        frame_count = 0

        while cap.isOpened() and recorded_video_processing_flag:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("End of video or failed to read frame.")
                break

            if frame_count % frame_skip == 0:
                resized_frame = cv2.resize(frame, (640, 640))  # Resize for faster inference
                processed_frame = process_frame(resized_frame, model, class_names)
                if processed_frame is not None:
                    cv2.imshow("Processed Video", processed_frame)

            frame_count += 1
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


                
def load_activity_log(text_widget):
    log_path = os.path.join("logs", "recognized_faces.csv")  # Ensure correct path
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

def create_control_bar(root):
    """Create a custom control bar with Minimize and Exit buttons."""
    control_bar = tk.Frame(root, bg="gray", height=30)
    control_bar.pack(side=tk.TOP, fill=tk.X)

    # Exit button
    exit_button = tk.Button(control_bar, text="X", bg="red", fg="white", font=("Arial", 12, "bold"), command=root.destroy)
    exit_button.pack(side=tk.RIGHT, padx=5, pady=2)

    # Minimize button
    minimize_button = tk.Button(control_bar, text="_", bg="blue", fg="white", font=("Arial", 12, "bold"), command=lambda: root.iconify())
    minimize_button.pack(side=tk.RIGHT, padx=5, pady=2)

    # Title label
    title_label = tk.Label(control_bar, text="Surveillance System", bg="gray", fg="white", font=("Arial", 14, "bold"))
    title_label.pack(side=tk.LEFT, padx=10)


def create_gui():
    root = tk.Tk()
    root.title("Surveillance System")
    # root.geometry("1420x1080")
    root.attributes('-fullscreen', True) 
    root.bind("<Escape>", lambda event: root.destroy())
    
    create_control_bar(root)
    
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

    header_title = tk.Label(header, text="Face First", font=("Arial", 24, "bold"), bg="white")
    header_title.pack(side=tk.LEFT, padx=20)
    
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
    tk.Button(tab1_buttons_frame, text="Add Face", command=add_face).pack(side=tk.LEFT, padx=10)
    tk.Button(tab1_buttons_frame, text="Manage Dangerous Objects", command=manage_dangerous_objects).pack(side=tk.LEFT, padx=10)
    tk.Button(tab1_buttons_frame, text="Detect Height", command=detect_height).pack(side=tk.LEFT, padx=10)  # New Button
    tab1_buttons_frame = tk.Frame(tab1)
    tab1_buttons_frame.pack(side=tk.BOTTOM, pady=20)

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