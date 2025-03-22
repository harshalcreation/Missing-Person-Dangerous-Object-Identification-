# Import required libraries
import cv2 as cv
import mediapipe as mp
import pyttsx3
import numpy as np

# Initialize MediaPipe pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Calibration parameters for a laptop webcam
KNOWN_HEIGHT_CM = 170  # Known height of reference person in cm
KNOWN_DISTANCE_CM = 60  # Distance from the camera in cm
REFERENCE_NOSE_TO_ANKLE_PIXELS = 400  # Update this value after calibration

# Dynamic scale factor (calculated based on calibration)
def calculate_scale_factor():
    return KNOWN_HEIGHT_CM / REFERENCE_NOSE_TO_ANKLE_PIXELS

scale_factor = calculate_scale_factor()

def speak(audio):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(audio)
    engine.runAndWait()

def estimate_height(frame, pose_results):
    if not pose_results.pose_landmarks:
        print("No landmarks detected.")
        return None

    # Get landmarks for nose and left/right ankle
    landmarks = pose_results.pose_landmarks.landmark
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Validate landmarks
    if nose.visibility < 0.5 or left_ankle.visibility < 0.5 or right_ankle.visibility < 0.5:
        print("Low visibility for key landmarks.")
        return None

    # Debug: Print landmark positions
    print(f"Nose: {nose.y}, Left Ankle: {left_ankle.y}, Right Ankle: {right_ankle.y}")

    # Calculate average ankle position
    avg_ankle_y = (left_ankle.y + right_ankle.y) / 2
    frame_height = frame.shape[0]

    # Calculate height in pixels
    height_in_pixels = abs(avg_ankle_y - nose.y) * frame_height

    # Debug: Print height in pixels
    print(f"Height in pixels: {height_in_pixels}")

    # Convert to centimeters using scale factor
    height_in_cm = height_in_pixels * scale_factor

    # Debug: Print estimated height
    print(f"Estimated Height: {height_in_cm} cm")

    return round(height_in_cm, 2)

def process_frame(frame):
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)

    if pose_results.pose_landmarks:
        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        height = estimate_height(frame, pose_results)

        if height:
            cv.putText(frame, f"Height: {height} cm", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            speak(f"Your estimated height is {height} centimeters")

    return frame

def main():
    cap = cv.VideoCapture(0)
    speak("Initializing height measurement system. Please ensure your full body is visible.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        processed_frame = process_frame(frame)
        cv.imshow("Height Detection", processed_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
