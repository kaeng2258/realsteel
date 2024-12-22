import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def process_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx in [0, 3, 8, 11, 12, 23, 24, 13, 14, 17, 18, 25, 26, 27, 28, 31, 32]:
                h, w, _ = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        selected_connections = [
            (11, 12), (23, 24), (11, 13), (12, 14), (13, 17), (14, 18),
            (11, 23), (12, 24), (11, 0), (12, 0), (23, 25), (24, 26),
            (25, 27), (26, 28), (27, 31), (28, 32), (0, 3), (0, 8), (3, 8)
        ]
        for connection in selected_connections:
            start_idx, end_idx = connection
            if results.pose_landmarks.landmark[start_idx].visibility > 0.5 and results.pose_landmarks.landmark[end_idx].visibility > 0.5:
                start = results.pose_landmarks.landmark[start_idx]
                end = results.pose_landmarks.landmark[end_idx]
                start_coords = (int(start.x * w), int(start.y * h))
                end_coords = (int(end.x * w), int(end.y * h))
                cv2.line(image, start_coords, end_coords, (255, 0, 0), 2)
    return image
