import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import os
import cv2 as cv
import numpy as np
import mediapipe as mp
import joblib
import streamlit as st

GREEN = (0, 255, 0)

#FUNCTIONS

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'hand_keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        finger_connections = [
            (2, 3, 4),  # Thumb
            (5, 6, 7, 8),  # Index finger
            (9, 10, 11, 12),  # Middle finger
            (13, 14, 15, 16),  # Ring finger
            (17, 18, 19, 20)  # Little finger
        ]

        for finger in finger_connections:
            for i in range(len(finger) - 1):
                cv.line(image, tuple(landmark_point[finger[i]]), tuple(landmark_point[finger[i + 1]]),
                        (0, 0, 0), 6)
                cv.line(image, tuple(landmark_point[finger[i]]), tuple(landmark_point[finger[i + 1]]),
                        (255, 255, 255), 2)

        # Connect palm to fingers
        palm_to_fingers = [(0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
        for connection in palm_to_fingers:
            cv.line(image, tuple(landmark_point[connection[0]]), tuple(landmark_point[connection[1]]),
                    (0, 0, 0), 2)
            cv.line(image, tuple(landmark_point[connection[0]]), tuple(landmark_point[connection[1]]),
                    (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        radius_inner = 1 if index % 4 == 0 else 3
        radius_outer = 1 if index % 4 == 0 else 1

        if index in [0, 1, 2, 3, 4]:
            color = (255, 255, 255)
        else:
            color = (0, 0, 0)

        cv.circle(image, (landmark[0], landmark[1]), radius_inner, (255, 255, 255), 2)
        cv.circle(image, (landmark[0], landmark[1]), radius_outer, (0,0,0), 2)

    return image

def draw_bounding_rect(use_brect, image, brect, color):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     color, 2)

    return image

def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        #info_text = info_text + ':' + hand_sign_text
        info_text = hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

#MAIN

from xgboost import XGBClassifier  # Updated for XGBoost model

# Argument parsing
args = get_args()

use_static_image_mode = args.use_static_image_mode
min_detection_confidence = args.min_detection_confidence
min_tracking_confidence = args.min_tracking_confidence

use_brect = True


# Model load
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=use_static_image_mode,
    max_num_hands=2,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)

@st.cache_resource
def load_model():
    return joblib.load(r'hand_xgboost_model.pkl')

# Load XGBoost model
xgboost_model = load_model()  # Replace with the path to your XGBoost model file

# Read labels
with open('hand_keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

def hand(vid, loading_bar_hand):

    # Camera preparation
    cap = cv.VideoCapture(vid)
    loading_bar_hand.progress(10)

    #mode = 0
    opencount = 0
    closecount = 0
    pointcount = 0
    count = 0

    output_frames = []
    loading_bar_hand.progress(20)

    while True:
        #key = cv.waitKey(10)
        #if key == 27:  # ESC
            #break
        #number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break

        debug_image = copy.deepcopy(image)
        debug_image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)

        height, width = debug_image.shape[:2]

        # Determine the scaling factor to make the longest edge 600 pixels
        scaling_factor = 800 / max(height, width)

        # Calculate the new dimensions
        new_height = int(height * scaling_factor)
        new_width = int(width * scaling_factor)

        # Resize the frame
        debug_image = cv.resize(debug_image, (new_width, new_height))

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                count =  count + 1
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                #logging_csv(number, mode, pre_processed_landmark_list)

                # Hand sign classification using XGBoost
                hand_sign_id = xgboost_model.predict([pre_processed_landmark_list])[0]

                if hand_sign_id == 0:
                    opencount = opencount+1
                    color = GREEN
                elif hand_sign_id == 1:
                    cosecount = closecount + 1
                    color = (0, 0, 255)
                elif hand_sign_id == 2:
                    pointcount = pointcount + 1
                    color = (0, 0, 255)

                cv.putText(debug_image, str(hand_sign_id), (100, 250), cv.FONT_HERSHEY_COMPLEX, 1.0, GREEN, 2)

                debug_image = draw_bounding_rect(use_brect, debug_image, brect, color)
                #debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    "",
                )

        #cv.imshow('Hand Gesture Recognition', debug_image)
        output_frames.append(debug_image)

        #if cv.waitKey(24) & 0xFF == ord('q'):
            #break

    cap.release()
    #cv.destroyAllWindows()

    loading_bar_hand.progress(80)

    try:

        open_score = (opencount/count)*100
        close_score = (closecount/count)*100
        point_score = (pointcount/count)*100

        #print(keypoint_classifier_labels[0], open_score, keypoint_classifier_labels[1], close_score, keypoint_classifier_labels[2], point_score)

        messagep = 'YOUR POSITIVE AREAS: '
        messagen = 'NEEDS IMPROVEMENT: '

        message = ''

        if open_score>=70:
            messagep = messagep + " Good job on using open hand gestures most of the time. Open hand gestures convey openness and enthusiasm."
        else:
            messagen = messagen + " Practice using open hand gestures to convey openness and enthusiasm."

        if close_score >=10:
            messagen = messagen + " Refrain from using closed hand gestures. Closed hands can be interpreted as nervousness and defensiveness."

        if point_score >=10:
            messagen = messagen + " Don't point your fingers too much. Pointing fingers can be seen as aggressive."
        
        if messagep == 'YOUR POSITIVE AREAS: ':
            messagep = ''
        if messagen == 'NEEDS IMPROVEMENT: ':
            messagen = ''
        
        message = messagep + '\n\n' + messagen
    
    except:
        open_score = 0
        message = 'No hand gestures detected.'
    
    loading_bar_hand.progress(90)
    
    return output_frames, message, open_score
