import csv
import copy
import itertools

import cv2 as cv
import numpy as np
import joblib
import mediapipe as mp
from xgboost import XGBClassifier
import streamlit as st

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

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


def draw_bounding_rect(use_brect, image, brect, rect_color):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), rect_color, 2)

    return image

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

def draw_info_text(image, brect, facial_text, add):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2]+add, brect[1] - 22),
                 (0, 0, 0), -1)

    if facial_text != "":
        info_text = 'Emotion :' + facial_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

@st.cache_resource
def load_model():
    return joblib.load(r'smile_xgboost_model.pkl')

def smile_detector(vid, loading_bar_smile):
    counter = 0
    smile_counter = 0

    output_frames = []

    cap_device = 0
    cap_width = 1920
    cap_height = 1080

    use_brect = True

    # Camera preparation
    cap = cv.VideoCapture(vid)

    loading_bar_smile.progress(10)

    # cap = cv.VideoCapture(r'E:\website files\The Art of Acting.mp4')
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    
    loading_bar_smile.progress(20)

    # Model load
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 

    # Load the XGBoost model
    xgb_model = load_model()

    # Read labels
    with open(r'smile_keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    
    loading_bar_smile.progress(30)

    mode = 0

    while True:
        # Process Key (ESC: end)
        #key = cv.waitKey(10)
        #if key == 27:  # ESC
            #break

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        #image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        debug_image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True

        # Find the dimensions of the frame
        height, width, _ = debug_image.shape

        # Determine the scaling factor to make the longest edge 600 pixels
        scaling_factor = 800 / max(height, width)

        # Calculate the new dimensions
        new_height = int(height * scaling_factor)
        new_width = int(width * scaling_factor)

        # Resize the frame
        debug_image = cv.resize(debug_image, (new_width, new_height))

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                counter = counter+1
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, face_landmarks)

                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, face_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Emotion classification using XGBoost model
                facial_emotion_id = xgb_model.predict([pre_processed_landmark_list])[0]

                # Determine the color of the bounding rectangle
                if facial_emotion_id in [0, 1]:
                    rect_color = (0, 255, 0)  # Green
                    smile_counter = smile_counter+1
                    add = 0
                else:
                    rect_color = (0, 0, 255)  # Red
                    add = 50

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect, rect_color)
                debug_image = draw_info_text(debug_image, brect, keypoint_classifier_labels[facial_emotion_id], add)

        # Screen reflection
        #cv.imshow('Facial Emotion Recognition', debug_image)
        output_frames.append(debug_image)
    
    loading_bar_smile.progress(80)

    cap.release()
    #cv.destroyAllWindows()

    try:

        smile_score = ((smile_counter/counter)*100)

        message = ''

        messagep = 'YOUR POSITIVE AREAS: '
        messagen = 'NEEDS IMPROVEMENT: '

        if smile_score<=25:
            messagen += "You look so serious. Smiles can help project a positive and confident image"
        elif 25<smile_score<=50:
            messagen += "Try to increase your smiling frequency. Smiles create a lasting positive impression."
        elif 50<smile_score<=75:
            messagen += "You've maintained a smile for most of the time. Consider maintaining your smiles a bit longer to convey even greater enthusiasm."
        elif 75<smile_score<=90:
            messagep += "Great job! Your smile throughout the video shows confidence and positivity. "
        elif 90<smile_score:
            messagep += "Impressive! Your prolonged smile reflects great confidence and enthusiasm throughout the interview."
        
        if messagep == 'YOUR POSITIVE AREAS: ':
            messagep = ''
        if messagen == 'NEEDS IMPROVEMENT: ':
            messagen = ''
        
        message = messagep + '\n\n' + messagen
    
    except:
        smile_score = 0
        message = 'No face detected.'
    
    loading_bar_smile.progress(90)

    return output_frames, message, smile_score
    
