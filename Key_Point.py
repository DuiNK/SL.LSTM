import cv2
import numpy as np
import os
import mediapipe as mp
import shutil
from datetime import timedelta



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

#ROOT DATA PATH
root_folder_path = 'data_raw/five'


#count the folder in the dir
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

# Get the video dimensions and frame rate



with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    for file in files:

        #get prefix
        prefix = file[:3]
        file_path = os.path.join(root_folder_path, file)

        if not os.path.exists(os.path.join(data_key_path, prefix)):
            os.makedirs(os.path.join(data_key_path, prefix))
            print(f"Created directory at {os.path.join(data_key_path, prefix)}")
            # else:
            #     print(f"Directory already exists")

        keypoint_path = os.path.join(data_key_path, prefix)
        max_dir = count_max_folders(keypoint_path)
        # print(f'max_dỉ {max_dir}')

        # Open the video file
        cap = cv2.VideoCapture(file_path)
        # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'total frame of {total_frames}')

        # Calculate the frame interval to use for keyframe extraction
        interval = float(total_frames / 60)

        # Initialize the keyframe list
        keyframes = []
        frame_index = 0

        # Define the codec and output file name
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # output_filename = file + 'output_video.mp4'
        # out = cv2.VideoWriter(output_filename, fourcc, frame_rate, (frame_width, frame_height))
        for i in range(total_frames):
            # Set the current frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            # Read the next frame
            ret, frame = cap.read()


            # Check if the frame was successfully read
            if not ret:
                break
            # If this is a keyframe, add it to the list
            if i == round(interval * frame_index):
                # keyframes.append(frame)
                # cv2.imwrite(file_path + str(i) + '.jpg', frame)
                # out.write(frame)

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                # print("keypoint", keypoints)
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
        # out.release()


