import cv2
import numpy as np
import os

import mediapipe as mp
from scipy import stats
from keras.models import load_model



colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        #cv2.rectangle(output_frame, (0, 6 + num * 4), (int(prob * 10), 9 + num * 4), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 9 + num * 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame
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
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)

    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Data_v2')

# Actions that we try to detect
# actions = np.array(['A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Ô',
#                    'Ơ', 'P', 'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'W', 'X', 'Y', 'Z', 'dau sac', 'dau huyen', 'dau nga',
#                    'dau hoi', 'dau nang', 'space'])

actions = np.array(['A', 'B', 'C', 'D', 'E'])
# Thirty videos worth of data
#no_sequences = 80

# Videos are going to be 30 frames in length
sequence_length = 80

# Folder start
#start_folder = 30

label_map = {label:num for num, label in enumerate(actions)}



model = load_model('model_sign_ABCDE.hdf5')
# 1. New detection variables
sequence = []
sentences = []
predictions = []
threshold = 0.3

cap = cv2.VideoCapture(0)

# Set mediapipe model

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.3) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: failed to read video input")
            break
        #frame = cv2.flip(frame,1)
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)
        cv2.imshow('OpenCV Feed', image)

        # 2. Prediction logic

        keypoints = extract_keypoints(results)
        print("Keypoints shape:", keypoints.shape)
        # print("Keypoints:", keypoints)

        sequence.append(keypoints)
        if len(sequence) == sequence_length:
            input_data = np.expand_dims(sequence, axis=0)
            # print("sequence: ", sequence)
            # print("Input data shape:", input_data.shape)
            res = model.predict(input_data)[0]
            # print("Predicted probabilities:", res)
            predicted_label = actions[np.argmax(res)]
            # print("Predicted label:", predicted_label)
            predictions.append(predicted_label)
            if res.max() > threshold:
                sentences.append(predicted_label)
            if len(sentences) > 10:
                sentences = sentences[-10:]
            sequence = []

        # Check predictions and sentences
        print("Predictions:", predictions)
        print("Sentences:", sentences)
        # if len(sequence) == 80:
        #     res = model.predict(np.expand_dims(sequence, axis=0))[0] #bo [0]
        #     print(res[np.argmax(res)])
        #     print(len(res))
        #     print(actions[np.argmax(res)])
        #     sequence = []
        #     cv2.putText(image, 'STARTING COLLECTION', (120, 200),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
        #     fps = cap.get(cv2.CAP_PROP_FPS)
        #     print("Camera FPS:", fps)
        #     predictions.append(np.argmax(res))
        #
        #     # 3. Viz logic
        #     #if np.unique(predictions[-10:]) == np.argmax(res): #bo [0]
        #
        #     if res[np.argmax(res)] > threshold:
        #             sentences.append(actions[np.argmax(res)])
        #
        #     if len(sentences) > 10:
        #         sentences = sentences[-10:]

            # Viz probabilities
            # image = prob_viz(res, actions, image)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentences), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()