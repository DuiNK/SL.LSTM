import cv2
import numpy as np
import os
import mediapipe as mp
import shutil
from datetime import timedelta

# i.e if video of duration 30 seconds, saves 10 frame per second = 300 frames saved in total
SAVING_FRAMES_PER_SECOND = 10

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

# Set mediapipe model
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)

    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('data_raw/five')

# Actions that we try to detect
#actions = np.array(['A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Ô',
#                    'Ơ', 'P', 'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'W', 'X', 'Y', 'Z', 'dau sac', 'dau huyen', 'dau nga',
#                    'dau hoi', 'dau nang', 'space', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

#actions = np.aray(['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

#  videos worth of data
no_sequences = 1

# Videos are going to be 30 frames in length
sequence_length = 80

# Folder start
start_folder = 350

#creat folder

# for action in actions:
#     dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
#     for sequence in range(1,no_sequences+1):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, action, str(dirmax+sequence)))
#         except:
#             pass
root_folder_path = 'data_raw/five'

# Get a list of all items in the directory
items = os.listdir(root_folder_path)

# Initialize a counter for the number of directories


def count_max_folders(root_folder):
    # Get a list of all files and directories in the directory
    files_and_directories = os.listdir(root_folder)

    # Filter out only the directories
    directories = [f for f in files_and_directories if os.path.isdir(os.path.join(root_folder, f))]

    # Count the number of directories
    num_directories = len(directories)

    return num_directories
print(f'max_folder in path: {count_max_folders(DATA_PATH)}')

data_key_path = 'data_keypoint'


files = os.listdir(root_folder_path)
# Set the start and end times (in seconds)

# Create a dictionary to store files by their first 3 characters
files_by_prefix = {}
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    for file in files:
        start_time = 0
        end_time = 1
        # Set the frame rate
        frame_rate = 30
        # Set the frame index
        frame_index = 0

        prefix = file[:3]
        file_path = os.path.join(root_folder_path, file)

        if not os.path.exists(os.path.join(data_key_path, prefix)):
            os.makedirs(os.path.join(data_key_path, prefix))
            print(f"Created directory at {os.path.join(data_key_path, prefix)}")
            # else:
            #     print(f"Directory already exists")

        keypoint_path = os.path.join(data_key_path, prefix)
        max_dir = count_max_folders(keypoint_path)
        print(f'max_dỉ {max_dir}')

        # Open the video file
        cap = cv2.VideoCapture(file_path)
        while True:
            # Set the current time (in seconds)
            current_time = frame_index / frame_rate

            # Check if we've reached the end time
            if current_time > end_time:
                break

            # Read the next frame
            ret, frame = cap.read()

            # Check if the frame was successfully read
            if not ret:
                break

            # Check if we've reached the start time
            if current_time >= start_time:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                print("keypoint", keypoints)
                if not os.path.exists(os.path.join(data_key_path, prefix, str(max_dir))):
                    os.makedirs(os.path.join(data_key_path, prefix, str(max_dir)))
                    print(f"Created directory at {os.path.join(data_key_path, prefix, str(max_dir))}")
                else:
                    print(f"Directory already exists")

                npy_path = os.path.join(data_key_path, prefix, str(max_dir), str(frame_index))
                np.save(npy_path, keypoints)

            # Increment the frame index
            frame_index += 1

        # Release the video file
        cap.release()


# Create a folder for each prefix and move the files into the respective folder
# for prefix, files in files_by_prefix.items():
#     folder_path = os.path.join(root_folder_path, prefix)
#     os.makedirs(folder_path, exist_ok=True)
#     for file in files:
#         file_path = os.path.join(root_folder_path, file)
#         folder_file_path = os.path.join(folder_path, file)

