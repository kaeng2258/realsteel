import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from bluetooth import connect_bluetooth, send_robot_command
from pose_estimation import process_frame, draw_landmarks
from data_processing import load_data, save_data, preprocess_data
from model import build_and_train_model

# Initialize global variables
bt_socket = None
landmark_list = []
label_list = []

def initialize():
    global bt_socket, landmark_list, label_list
    bt_socket = connect_bluetooth()
    landmark_list, label_list = load_data()

def capture_and_process_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None, None
    image, results = process_frame(frame)
    if results.pose_landmarks:
        landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]
        image = draw_landmarks(image, results)
        return image, landmarks
    return image, None

def predict_pose(landmarks):
    min_distance = float('inf')
    min_label = "Ready"
    min_accuracy = 0
    for i, stored_landmarks in enumerate(landmark_list):
        dist = np.sum((np.array(landmarks) - np.array(stored_landmarks)) ** 2)
        if dist < min_distance:
            min_distance = dist
            min_label = label_list[i]
            min_accuracy = int(1 / (1 + dist) * 100)
    if min_accuracy < 80:
        min_label = "Ready"
    return min_label, min_accuracy

def handle_pose_prediction(image, min_label, min_accuracy):
    label_text = f"{min_label} ({min_accuracy}%)" if min_label != "Ready" else "Ready"
    cv2.putText(image, label_text, (image.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    command_dict = {
        "gurad": 23, "jap": 30, "straight": 33, "lturn": 28,
        "rturn": 29, "forward": 24, "back": 25, "lmove": 26, "rmove": 27
    }
    if min_label in command_dict:
        send_robot_command(bt_socket, command_dict[min_label])

def main_loop():
    global landmark_list, label_list
    input_label = True
    label = ""
    collecting = False
    exit_program = False

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        if exit_program:
            break

        image, landmarks = capture_and_process_frame(cap)
        if image is None:
            print("Error: Failed to capture image")
            break

        if landmarks:
            min_label, min_accuracy = predict_pose(landmarks)
            handle_pose_prediction(image, min_label, min_accuracy)

        if input_label:
            cv2.putText(image, "Enter label and press 'Enter':", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Press 'ESC' to quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('MediaPipe Pose', image)
            key = cv2.waitKey(10) & 0xFF
            if key == 13:  # 'Enter' key
                input_label = False
                collecting = False
            elif key == 8:  # 'Backspace' key
                label = label[:-1]
            elif key != 255 and key != 27:  # Other keys
                label += chr(key)
            elif key == 27:  # 'ESC' key
                exit_program = True
                break
        else:
            if collecting:
                cv2.putText(image, f"Collecting: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                if landmarks:
                    landmark_list.append(landmarks)
                    label_list.append(label)
            else:
                cv2.putText(image, "Press 's' to start, 'q' to stop", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Press 'ESC' to quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Pose', image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s') and not collecting:
                collecting = True
                print("Started collecting data...")
            elif key == ord('q') and collecting:
                collecting = False
                print("Stopped collecting data...")
                input_label = True
                label = ""
            elif key == 27:
                exit_program = True
                break

    cap.release()
    cv2.destroyAllWindows()

def train_and_predict():
    if len(landmark_list) > 0 and len(label_list) > 0:
        save_data(landmark_list, label_list)
        landmarks, labels, le = preprocess_data()
        model = build_and_train_model(landmarks, labels)

        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        while cap.isOpened():
            image, landmarks = capture_and_process_frame(cap)
            if image is None:
                break

            if landmarks:
                landmarks = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(landmarks)
                predicted_label = le.inverse_transform([np.argmax(prediction)])[0]

                h, w, _ = image.shape
                text_size, _ = cv2.getTextSize(predicted_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                text_x = w - text_size[0] - 10
                text_y = 30
                cv2.putText(image, f"Predicted Label: {predicted_label}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                print(f"Predicted label: {predicted_label}")

            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No data collected or saved.")

def display_saved_data():
    if os.path.exists('landmarks.npy') and os.path.exists('labels.npy'):
        landmarks = np.load('landmarks.npy')
        labels = np.load('labels.npy')
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        inverse_labels = le.inverse_transform(labels)

        for i, (landmark, label) in enumerate(zip(landmarks, inverse_labels)):
            print(f"Frame {i}: Label = {label}")
            print(landmark)
            print()
    else:
        print("No saved data to display.")

if __name__ == "__main__":
    initialize()
    main_loop()
    train_and_predict()
    display_saved_data()
