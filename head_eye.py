import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import mediapipe as mp
import math
import pandas as pd
import joblib
import time
import pickle
import streamlit as st

# Load mediapipe face mesh solution
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define face mesh indices for different facial features
# Standard face mesh landmarks (compatible with all MediaPipe versions)
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384]

# Eye contours
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157]
RIGHT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384]

# Use standard eyes instead of iris
LEFT_IRIS = LEFT_EYE
RIGHT_IRIS = RIGHT_EYE

# Head pose points
HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]

# Load models
def load_models():
    head_model = pickle.load(open('head_pose.pkl', 'rb'))
    eye_model = pickle.load(open('eye_direction.pkl', 'rb'))
    return head_model, eye_model

def head_eye(file_path, loading_bar_placeholder=None):
    # Initialize video capture
    cap = cv2.VideoCapture(file_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    if loading_bar_placeholder:
        progress_bar = loading_bar_placeholder.progress(0)
    
    # To save the frames as a video
    output_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Load models
    try:
        head_model, eye_model = load_models()
    except Exception as e:
        return None, f"Error loading models: {str(e)}", 0, 0
    
    # Variables to track metrics
    head_data = []
    eye_data = []
    frame_count_processed = 0
    unsuccessful_frames = 0
    
    # Start time
    start_time = time.time()
    
    # Sample frames for analysis (every second)
    sample_interval = fps
    
    # Frame lists
    output_frames = []
    
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,  # Set to False if iris landmarks are causing issues
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        
        frame_idx = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            frame_idx += 1
            
            # Update progress bar
            if loading_bar_placeholder and frame_idx % 30 == 0:
                progress_bar.progress(min(frame_idx / frame_count, 1.0))
            
            # Sample frames (process 1 frame per second)
            if frame_idx % sample_interval != 0 and frame_idx != 1:
                continue
            
            # To improve performance
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                try:
                    # Extract mesh points
                    mesh_points = np.array([
                        [int(face_landmarks.landmark[idx].x * img_w), 
                         int(face_landmarks.landmark[idx].y * img_h)]
                        for idx in range(len(face_landmarks.landmark))
                    ])
                    
                    # Draw face mesh
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in HEAD_POSE_LANDMARKS:
                            cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), 2, (0, 255, 0), -1)
                    
                    # Head pose estimation
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            
                            # Get 3D coordinates
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z * img_h])
                    
                    # Convert to NumPy arrays
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)
                    
                    # The camera matrix
                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                          [0, focal_length, img_h / 2],
                                          [0, 0, 1]])
                    
                    # The distortion parameters
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    
                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    
                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    
                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                    
                    # Get the rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360
                    
                    # Calculate head direction
                    head_features = [x, y, z]
                    head_prediction = head_model.predict([head_features])[0]
                    head_data.append(head_prediction)
                    
                    # Display head direction
                    if head_prediction == 0:
                        head_text = "Head: Facing Camera"
                    elif head_prediction == 1:
                        head_text = "Head: Looking Left"
                    elif head_prediction == 2:
                        head_text = "Head: Looking Right"
                    else:
                        head_text = "Head: Looking Up/Down"
                    
                    # Eye direction estimation and iris detection
                    try:
                        # Left and right eye coordinates
                        l_eye_cont_coordinates = []
                        for i in LEFT_EYE_CONTOUR:
                            if i < len(mesh_points):
                                l_eye_cont_coordinates.append(tuple(mesh_points[i]))
                            else:
                                print(f"Warning: Left eye index {i} out of bounds")
                                if l_eye_cont_coordinates:
                                    l_eye_cont_coordinates.append(l_eye_cont_coordinates[-1])
                        
                        r_eye_cont_coordinates = []
                        for i in RIGHT_EYE_CONTOUR:
                            if i < len(mesh_points):
                                r_eye_cont_coordinates.append(tuple(mesh_points[i]))
                            else:
                                print(f"Warning: Right eye index {i} out of bounds")
                                if r_eye_cont_coordinates:
                                    r_eye_cont_coordinates.append(r_eye_cont_coordinates[-1])
                        
                        # Make sure we have enough points to continue
                        if len(l_eye_cont_coordinates) > 2 and len(r_eye_cont_coordinates) > 2:
                            l_eye_cont_array = np.array(l_eye_cont_coordinates, np.int32)
                            r_eye_cont_array = np.array(r_eye_cont_coordinates, np.int32)
                            
                            # Draw eye contours
                            cv2.polylines(image, [l_eye_cont_array], True, (0, 255, 0), 1)
                            cv2.polylines(image, [r_eye_cont_array], True, (0, 255, 0), 1)
                            
                            # Use eye landmarks to approximate iris
                            left_eye_points = [mesh_points[idx] for idx in LEFT_EYE if idx < len(mesh_points)]
                            right_eye_points = [mesh_points[idx] for idx in RIGHT_EYE if idx < len(mesh_points)]
                            
                            if left_eye_points and right_eye_points:
                                left_eye_points_array = np.array(left_eye_points)
                                right_eye_points_array = np.array(right_eye_points)
                                
                                # Calculate approximate center and radius of eyes
                                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(left_eye_points_array)
                                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(right_eye_points_array)
                                
                                # Draw iris circles
                                cv2.circle(image, (int(l_cx), int(l_cy)), int(l_radius), (0, 255, 0), 1)
                                cv2.circle(image, (int(r_cx), int(r_cy)), int(r_radius), (0, 255, 0), 1)
                                
                                # Calculate eye direction features
                                # Calculate eye landmarks relative positions
                                l_eye_center = np.mean(l_eye_cont_array, axis=0)
                                r_eye_center = np.mean(r_eye_cont_array, axis=0)
                                
                                l_iris_rel_x = (l_cx - l_eye_center[0]) / l_radius if l_radius > 0 else 0
                                l_iris_rel_y = (l_cy - l_eye_center[1]) / l_radius if l_radius > 0 else 0
                                r_iris_rel_x = (r_cx - r_eye_center[0]) / r_radius if r_radius > 0 else 0
                                r_iris_rel_y = (r_cy - r_eye_center[1]) / r_radius if r_radius > 0 else 0
                                
                                eye_features = [l_iris_rel_x, l_iris_rel_y, r_iris_rel_x, r_iris_rel_y, x, y, z]
                                eye_prediction = eye_model.predict([eye_features])[0]
                                eye_data.append(eye_prediction)
                                
                                # Display eye direction
                                if eye_prediction == 0:
                                    eye_text = "Eyes: Looking at Camera"
                                elif eye_prediction == 1:
                                    eye_text = "Eyes: Looking Left"
                                elif eye_prediction == 2:
                                    eye_text = "Eyes: Looking Right"
                                else:
                                    eye_text = "Eyes: Looking Up/Down"
                                
                                cv2.putText(image, head_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                cv2.putText(image, eye_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                frame_count_processed += 1
                            else:
                                cv2.putText(image, "Eye landmarks not detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                unsuccessful_frames += 1
                        else:
                            cv2.putText(image, "Eye contours incomplete", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            unsuccessful_frames += 1
                    
                    except Exception as e:
                        print(f"Error in eye processing: {e}")
                        cv2.putText(image, "Eye detection failed", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        unsuccessful_frames += 1
                        
                except Exception as e:
                    print(f"Error in face mesh processing: {e}")
                    cv2.putText(image, "Face mesh processing failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    unsuccessful_frames += 1
            else:
                cv2.putText(image, "Face not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                unsuccessful_frames += 1
            
            # Save frame to output video
            out.write(image)
            output_frames.append(image)
    
    cap.release()
    out.release()
    
    # Calculate the scores
    head_score = 0
    eye_score = 0
    
    if head_data:
        # Calculate percentage of time facing camera (head)
        head_facing_camera = head_data.count(0) / len(head_data) if head_data else 0
        head_score = int(head_facing_camera * 100)
    
    if eye_data:
        # Calculate percentage of time looking at camera (eyes)
        eye_looking_at_camera = eye_data.count(0) / len(eye_data) if eye_data else 0
        eye_score = int(eye_looking_at_camera * 100)
    
    # Generate message based on scores
    message = ""
    if head_score < 50:
        message += "Your head position could use improvement. Try to face the interviewer more directly. "
    else:
        message += "Good job maintaining a strong head position throughout the interview. "
    
    if eye_score < 50:
        message += "Your eye contact needs work. Try to look directly at the camera more consistently."
    else:
        message += "You maintained good eye contact during the interview."
    
    # Update progress bar to completion
    if loading_bar_placeholder:
        progress_bar.progress(1.0)
    
    return output_frames, message, head_score, eye_score