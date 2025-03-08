import csv
import copy
import itertools

import cv2
import numpy as np
import mediapipe as mp
import joblib
import streamlit as st

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for landmark in landmarks.landmark[0:25]:  # Only take the first 23 landmarks
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

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark[11:25]):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_bounding_rect(use_brect, image, brect, rect_color):
    if use_brect:
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), rect_color, 2)

    return image

def draw_info_text(image, brect, facial_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if facial_text != "":
        info_text = 'Pose :' + facial_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image

@st.cache_resource
def load_model():
    return joblib.load(r'pose_XGB_model.pkl')

def body(vid, loading_bar_pose):
    # Rest of your code remains mostly unchanged, with adjustments for the Pose model
    pos = 0
    crossed = 0
    raised = 0
    explain = 0
    straight = 0 
    face = 0

    count = 0
    cap_device = 0
    cap_width = 1920
    cap_height = 1080
    output_frames = []

    mp_draw = mp.solutions.drawing_utils
    use_brect = True

    # Camera preparation
    cap = cv2.VideoCapture(vid)  # You may need to adjust the camera source

    loading_bar_pose.progress(10)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Load the XGBoost model
    xg_boost_model = load_model()

    # Read labels
    with open(r'pose_keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = f.read().splitlines()

    mode = 0

    loading_bar_pose.progress(20)

    progress = 20

    while True:
        # Process Key (ESC: end)
        #key = cv2.waitKey(10)
        #if key == 27:  # ESC
            #break

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        #image = cv2.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)

        # Find the dimensions of the frame
        height, width, _ = debug_image.shape

        # Determine the scaling factor to make the longest edge 600 pixels
        scaling_factor = 800 / max(height, width)

        # Calculate the new dimensions
        new_height = int(height * scaling_factor)
        new_width = int(width * scaling_factor)

        # Resize the frame
        debug_image = cv2.resize(debug_image, (new_width, new_height))

        # Detection implementation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks is not None:
            count = count + 1
            if progress<=80:
                progress + 0.01
                loading_bar_pose.progress((int(progress)))
                
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, results.pose_landmarks)

            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, results.pose_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # Emotion classification using SVM model
            facial_emotion_id = xg_boost_model.predict([pre_processed_landmark_list])[0]

            # Determine the color of the bounding rectangle
            if facial_emotion_id == 0:
                crossed = crossed + 1
            elif facial_emotion_id == 1:
                raised = raised + 1
            elif facial_emotion_id == 2:
                explain = explain +1
            elif facial_emotion_id == 3:
                straight = straight +1
            elif facial_emotion_id == 4:
                face = face +1

            if facial_emotion_id in [2, 3]:
                rect_color = (0, 255, 0)  # Green
                pos = pos + 1
            else:
                rect_color = (0, 0, 255)  # Red

            # Drawing part
            debug_image = draw_bounding_rect(use_brect, debug_image, brect, rect_color)
            debug_image = draw_info_text(
                debug_image,
                brect,
                keypoint_classifier_labels[facial_emotion_id])

        # Screen reflection
        #mp_draw.draw_landmarks(debug_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        #cv2.imshow('Pose Recognition', debug_image)
        output_frames.append(debug_image)

    cap.release()
    #cv2.destroyAllWindows()

    loading_bar_pose.progress(80)

    try:
        pos_score = (pos/count)*100
        crosed_score = (crossed/count)*100
        raised_score = (raised/count)*100
        face_score = (face/count)*100

        message = ''

        messagep = 'YOUR POSITIVE AREAS: '
        messagen = 'NEEDS IMPROVEMENT: '

        if pos_score>=70:
            messagep = messagep + " Good job on sitting straight and using hand gestures while explaining things. This emphasizes enthusiasm, and enhances approachability."
        else:
            messagen = messagen + " Sit straight and use hand gestures while explaining things. It emphasizes enthusiasm, and enhances approachability."

        if crosed_score >=10:
            messagen = messagen + " Don't cross your arms. Crossing arms signals disinterest or defensiveness."

        if raised_score >=10:
            messagen = messagen + " Don't raise your arms. Raised arms shows dominance and makes you look unprofessional."
        
        if face_score >=10:
            messagen = messagen + " Don't touch your face. Touching the face can convey insecurity."
        
        if messagep == 'YOUR POSITIVE AREAS: ':
            messagep = ''
        if messagen == 'NEEDS IMPROVEMENT: ':
            messagen = ''
        
        message = messagep + '\n\n' + messagen

    except Exception as e:
        print(e)
        pos_score=0
        message='No user detected.'
    
    loading_bar_pose.progress(90)
    
    return output_frames, message, pos_score
