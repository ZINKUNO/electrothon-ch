import gc

import streamlit as st
import pandas as pd
import os
import cv2
import sqlite3
import datetime
import plotly.express as px
from displayDB import display_tables_and_contents
from head_eye import head_eye
from smile import smile_detector
from hand import hand
from pose_detector import body
from io import BytesIO
import numpy as np
import av

# Get the current date and time
current_datetime = datetime.datetime.now()

# Format it as a string including seconds and with the desired format
current_date = current_datetime.strftime("%dth %b %H:%M:%S")

# Create or connect to the SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create a table to store user data if it doesn't exist
c.execute('''
          CREATE TABLE IF NOT EXISTS users (
              username TEXT PRIMARY KEY,
              password TEXT,
              avg_scores FLOAT,
              n_videos INTEGER
          )
          ''')
conn.commit()

def create_user_scores_table(username):
    try:
        c.execute(f'''
            CREATE TABLE IF NOT EXISTS {username}_scores (
                username TEXT,  
                head_score INTEGER,
                eye_score INTEGER,
                smile_score INTEGER,
                hand_score INTEGER,
                pose_score INTEGER,
                date_added DATE,
                avg_score INTEGER,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        ''')
        conn.commit()
        #st.success(f"Table created successfully for user: {username}")
    except Exception as e:
        st.error(f"Error creating account for user {username}: {e}")

# Function to update user scores in the database
def update_user_scores(username, head_score, eye_score, smile_score, hand_score, pose_score, date, avg_score):
    c.execute(f'''
              INSERT INTO {username}_scores
              (username, head_score, eye_score, smile_score, hand_score, pose_score, date_added, avg_score)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)
              ''', (username, head_score, eye_score, smile_score, hand_score, pose_score, date, avg_score))
    conn.commit()

# Set up Streamlit session state
if 'state' not in st.session_state:
    st.session_state.state = 'login'

# Main Streamlit app
if st.session_state.state == 'login':
    st.title("Login")
    login_username = st.text_input("Username:")
    login_password = st.text_input("Password:", type="password")

    if st.button("Login"):
        # Check if username and password are not blank
        if not login_username or not login_password:
            st.error("Username and password cannot be blank.")
        else:
            # Check username and password (replace with your authentication logic)
            users = c.execute('''
                                 SELECT * FROM users
                                 WHERE username = ? AND password = ?
                                 ''', (login_username, login_password)).fetchone()

            if users:
                st.success("Login successful!")
                st.session_state.username = login_username  # Store username in session state
                st.session_state.state = 'main'
            else:
                st.error("Incorrect username or password. Please try again.")

    st.markdown("---")
    st.subheader("Don't have an account?")
    if st.button("Sign Up"):
        st.session_state.state = 'signup'

elif st.session_state.state == 'signup':
    st.title("Sign Up")
    signup_username = st.text_input("New Username:")
    signup_password = st.text_input("New Password:", type="password")
    confirm_password = st.text_input("Confirm Password:", type="password")

    if st.button("Sign Up"):
        # Check if username, password, and confirm password are not blank
        if not signup_username or not signup_password or not confirm_password:
            st.error("Username, password, and confirm password cannot be blank.")
        else:
            if signup_password == confirm_password:
                existing_user = c.execute('''
                                          SELECT * FROM users
                                          WHERE username = ?
                                          ''', (signup_username,)).fetchone()

                if existing_user:
                    st.error("Username already taken. Please choose a different one.")
                else:
                    c.execute('''
                              INSERT INTO users (username, password)
                              VALUES (?, ?)
                              ''', (signup_username, signup_password))
                    conn.commit()
                    
                    # Create a table for the new user's scores
                    create_user_scores_table(signup_username)
                    #create_user_scores_table(signup_username)

                    st.success("Account created successfully! You can now log in.")
                    st.session_state.state = 'login'
            else:
                st.error("Passwords do not match. Please try again.")

elif st.session_state.state == 'main':

    # Set the account icon image URL
    account_icon_url = "https://cdn.iconscout.com/icon/free/png-256/free-account-269-866236.png"  # Replace with the actual URL of your account icon
    # Calculate the center alignment
    center_alignment = "text-align: center;"

    # Display account icon and welcome message with center alignment
    st.sidebar.markdown(f'<div style="{center_alignment}"><img src="{account_icon_url}" width="50"></div>', unsafe_allow_html=True)
    st.sidebar.header(f"Welcome, {st.session_state.username}!")
        
    # Dropdown menu in the sidebar
    menu_option = st.sidebar.selectbox("Options", ["Profile", "Analyse", "Leaderboard", "Log out"])

    if menu_option == "Log out":
        st.session_state.state = 'login'
    elif menu_option == "Analyse":
        st.session_state.state = 'analyse'
    elif menu_option == "Leaderboard":
        st.session_state.state = 'leaderboard' 

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Century+Gothic:wght@400&display=swap');
            h1 {
                text-align: center;
                font-size: 35px;
                font-family: 'Century Gothic', sans-serif;
            }
            h2 {
                text-align: center;
                font-size: 24px;
                font-family: 'Century Gothic', sans-serif;
            }
            span.custom-font {
                font-family: 'Century Gothic', sans-serif;
            }
            span.red-text {
                color: red;
            }
        </style>
        <h1><span class="custom-font">Nonverbal </span><span class="custom-font red-text">AI vibe</span></h1>
        <h2>An <span class="red-text custom-font">AI powered</span> Non-verbal Communication Coach</h1>
    """, unsafe_allow_html=True)

    # Display progress graph
    username = st.session_state.username
    table_name = f"{username}_scores"

    # Query scores from the database
    scores_data = c.execute(f'''
                            SELECT head_score, eye_score, smile_score, hand_score, pose_score, date_added
                            FROM {table_name}
                            ''').fetchall()
    
    scores_= c.execute(f'''
                            SELECT head_score, eye_score, smile_score, hand_score, pose_score
                            FROM {table_name}
                            ''').fetchall()
    
    num_videos_analyzed = len(scores_data)

    if num_videos_analyzed>=2:
        st.title("Your Progress so far")

        # Query scores from the database, including 'avg_score'
        avgscores_data = c.execute(f'''
                                SELECT avg_score, date_added
                                FROM {table_name}
                                ''').fetchall()

        if avgscores_data:
            # Create a DataFrame
            scores_df = pd.DataFrame(avgscores_data, columns=['Avg_Score', 'Date'])
            
            # Set the 'Date' column as the index
            scores_df.set_index('Date', inplace=True)

            # Calculate the overall average of 'avg_scores'
            overall_avg_score = scores_df['Avg_Score'].mean()

            # Define color based on ranges
            if overall_avg_score < 40:
                avg_score_color = 'red'
            elif 40 <= overall_avg_score <= 70:
                avg_score_color = 'orange'
            else:
                avg_score_color = 'green'

            # Create a curvy line chart using Plotly
            fig = px.line(scores_df, x=scores_df.index, y='Avg_Score', labels={'Avg_Score': 'Overall Score'}, title = 'Overall score over time')
            fig.update_traces(line_shape='spline', line_smoothing=0.2, marker=dict(color='blue'))  # You can adjust the line_smoothing for curvature
            
            # Display the overall average score above the first graph with specified color
            st.markdown(f"Average Overall Score:<span style='color: {avg_score_color}; font-size: 40px;'> {overall_avg_score:.2f}</span>", unsafe_allow_html=True)

            # Display the number of videos analyzed
            st.markdown(f"Number of Videos Analyzed: {num_videos_analyzed}")

            # Display the Plotly figure
            st.plotly_chart(fig)

        # Create a DataFrame
        scores_df = pd.DataFrame(scores_data, columns=['Head', 'Eye', 'Smile', 'Hand', 'Pose', 'Date'])
        scores = pd.DataFrame(scores_, columns=['Head', 'Eye', 'Smile', 'Hand', 'Pose'])

        # Calculate the average scores
        avg_scores = scores.mean()
        
        # Display the average scores as text
        # Function to format scores with different colors based on ranges
        def format_score_with_color(score):
            if score < 40:
                return f'<span style="color: red;">{score:.2f}</span>'
            elif 40 <= score <= 70:
                return f'<span style="color: orange;">{score:.2f}</span>'
            elif 70 < score <= 100:
                return f'<span style="color: green;">{score:.2f}</span>'
            else:
                return f'{score:.2f}'
        
        # Apply the formatting function to each average score
        formatted_avg_scores = avg_scores.apply(format_score_with_color)
        
        # Display the average scores with different colors
        st.markdown(f"Average Scores: Head={formatted_avg_scores['Head']}, Eye={formatted_avg_scores['Eye']}, Smile={formatted_avg_scores['Smile']}, Hand={formatted_avg_scores['Hand']}, Pose={formatted_avg_scores['Pose']}", unsafe_allow_html=True)
        
        # Set the 'Date' column as the index
        scores_df.set_index('Date', inplace=True)

        # Create a curvy line chart using Plotly
        fig = px.line(scores_df, x=scores_df.index, y=scores_df.columns, labels={'value': 'Score'}, title='Scores Over Time')
        fig.update_traces(line_shape='spline', line_smoothing=0.2, marker=dict(size=8, color='blue'))  # Corrected attribute to 'marker'

        # Display the Plotly figure
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No progress data available. Analyse atleast 2 videos to view your progress")

    # Button to navigate to 'Welcome' page
    if st.button("Analyse a video"):
        st.session_state.state = 'analyse'

elif st.session_state.state == 'leaderboard':

    account_icon_url = "https://cdn.iconscout.com/icon/free/png-256/free-account-269-866236.png"  # Replace with the actual URL of your account icon
    # Calculate the center alignment
    center_alignment = "text-align: center;"

    # Display account icon and welcome message with center alignment
    st.sidebar.markdown(f'<div style="{center_alignment}"><img src="{account_icon_url}" width="50"></div>', unsafe_allow_html=True)
    st.sidebar.header(f"Welcome, {st.session_state.username}!")

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Century+Gothic:wght@400&display=swap');
            h1 {
                text-align: center;
                font-size: 35px;
                font-family: 'Century Gothic', sans-serif;
            }
            h2 {
                text-align: center;
                font-size: 24px;
                font-family: 'Century Gothic', sans-serif;
            }
            span.custom-font {
                font-family: 'Century Gothic', sans-serif;
            }
            span.red-text {
                color: red;
            }
        </style>
        <h1><span class="custom-font">Nonverbal </span><span class="custom-font red-text"> AI vibe</span></h1>
        <h2></span>Leaderboard</h1>
    """, unsafe_allow_html=True)
    
    # Dropdown menu in the sidebar
    menu_option = st.sidebar.selectbox("Options", ["Leaderboard", "Profile", "Analyse", "Log out"])

    if menu_option == "Log out":
        st.session_state.state = 'login'
    if menu_option == "Profile":
        st.session_state.state = 'main'
    elif menu_option == "Analyse":
        st.session_state.state = 'analyse' 

    # Sample SQL query to retrieve data from the users table, ordered by avg_scores in descending order
    leaderboard_data = c.execute('''
        SELECT username, avg_scores, n_videos
        FROM users
        ORDER BY avg_scores DESC
    ''').fetchall()

    # Create a DataFrame from the query results
    leaderboard_df = pd.DataFrame(leaderboard_data, columns=['Username', 'Average Overall Score', 'Number of Videos'])

    # Convert 'Number of Videos' column to integers, keeping NaN values unchanged
    leaderboard_df['Number of Videos'] = pd.to_numeric(leaderboard_df['Number of Videos'], errors='coerce').astype('Int64')

    # Display the leaderboard table
    st.table(leaderboard_df.style.highlight_max(axis=0, subset=['Average Overall Score'], color='#FFD700'))

elif st.session_state.state == 'analyse':

    # Display user information in the sidebar
    # Set the account icon image URL
    account_icon_url = "https://cdn.iconscout.com/icon/free/png-256/free-account-269-866236.png"  # Replace with the actual URL of your account icon
    # Calculate the center alignment
    center_alignment = "text-align: center;"

    # Display account icon and welcome message with center alignment
    st.sidebar.markdown(f'<div style="{center_alignment}"><img src="{account_icon_url}" width="50"></div>', unsafe_allow_html=True)
    st.sidebar.header(f"Welcome, {st.session_state.username}!")
    
    # Dropdown menu in the sidebar
    menu_option = st.sidebar.selectbox("Options", ["Analyse", "Profile", "Leaderboard", "Log out"])

    if menu_option == "Log out":
        st.session_state.state = 'login'
    if menu_option == "Profile":
        st.session_state.state = 'main'
    elif menu_option == "Leaderboard":
        st.session_state.state = 'leaderboard' 

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Century+Gothic:wght@400&display=swap');
            h1 {
                text-align: center;
                font-size: 35px;
                font-family: 'Century Gothic', sans-serif;
            }
            h2 {
                text-align: center;
                font-size: 24px;
                font-family: 'Century Gothic', sans-serif;
            }
            span.custom-font {
                font-family: 'Century Gothic', sans-serif;
            }
            span.red-text {
                color: red;
            }
        </style>
        <h1><span class="custom-font">Nonverbal </span><span class="custom-font red-text">AI vibe</span></h1>
        <h2>An <span class="red-text custom-font">AI powered</span> Non-verbal Communication Coach</h1>
    """, unsafe_allow_html=True)

    import pandas as pd

    from moviepy.video.io.VideoFileClip import VideoFileClip

    # Upload video file
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
    #record_button = st.button("Record Live Video")
    uploaded = 0

    if uploaded_file is not None:
        
        # Display video details
        video_details = {"Name": uploaded_file.name, "Type": uploaded_file.type, "Size": uploaded_file.size}

        # Create the "temp" directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)

        # Save the file temporarily and get the file path
        with st.spinner("Uploading..."):
            file_path = os.path.join("temp", uploaded_file.name)
            with open(file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
                
        # Check video duration
        video_clip = VideoFileClip(file_path)
        duration = video_clip.duration
        video_clip.close()
        
        if duration > 30:
            st.write("This is just a prototype. Currently, it doesn't have the power to process videos longer than 30 seconds.")
            st.write("Trimming video to the first 30 seconds...")
            
            # Trim the video
            trimmed_clip = VideoFileClip(file_path).subclip(0, 30)
            trimmed_file_path = os.path.join("temp", "trimmed_" + uploaded_file.name)
            trimmed_clip.write_videofile(trimmed_file_path, codec='libx264', audio_codec='aac', temp_audiofile='temp/temp-audio.m4a', remove_temp=True)
            trimmed_clip.close()
            
            # Delete original video
            os.remove(file_path)
            
            # Update file path
            file_path = trimmed_file_path
        
        uploaded = 1

    if uploaded ==1:
        
        # Process the video and save the output frames
        st.markdown("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Century+Gothic:wght@400&display=swap');
                h1 {
                    text-align: center;
                    font-size: 35px;
                    font-family: 'Century Gothic', sans-serif;
                }
                h2 {
                    text-align: center;
                    font-size: 24px;
                    font-family: 'Century Gothic', sans-serif;
                }
                span.custom-font {
                    font-family: 'Century Gothic', sans-serif;
                }
                span.red-text {
                    color: red;
                }
            </style>
            <h2></span> Head Pose and Eye Contact Analysis</h1>
        """, unsafe_allow_html=True)
        
        # Loading bar and message for head and eye analysis
loading_text = st.text("Please wait. Video is being processed for eye contact and head pose analysis...")
loading_bar_he = st.progress(0)

try:
    # Process the video
    output_frames, message, head_score, eye_score = head_eye(file_path, loading_bar_he)
    
    if output_frames is not None and len(output_frames) > 0:
        # Encode video frames using PyAV
        output_video_bytes = BytesIO()
        output_frames = np.array(output_frames)  # Convert frames to numpy array
        
        with av.open(output_video_bytes, 'w', format='mp4') as container:
            stream = container.add_stream('h264', rate=20)  # H264 codec with 20 fps
            for frame in output_frames:
                frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                packet = stream.encode(frame)
                if packet:
                    container.mux(packet)
            # Flush the stream to ensure all frames are encoded
            for packet in stream.encode():
                container.mux(packet)
        
        # Display the output video
        output_video_bytes.seek(0)  # Reset BytesIO object to start
        st.video(output_video_bytes, format='video/mp4')
    else:
        st.error("No video frames were generated for head and eye analysis.")
        head_score = 0
        eye_score = 0
    
    # Clear frames to free memory
    output_frames = []
    
except Exception as e:
    st.error(f"Error in head and eye analysis: {str(e)}")
    head_score = 0
    eye_score = 0

# Display message and table
st.markdown(str(message) if 'message' in locals() else "Analysis could not be completed.")

data = {'Metric': ['Head Score', 'Eye Score'],
        'Score': [int(head_score), int(eye_score)]}

df = pd.DataFrame(data)
st.table(df)

st.write("Head score is given for maintaining good head posture. Ensure that your camera is not tilted. The camera should be at your eye level for the system to properly understand the position of your head")
st.write("Eye score is given for maintaining good eye contact")

# Update loading bar
loading_bar_he.progress(100)
loading_text.text("Head pose and eye contact analysis complete!")

        #SMILE DET

st.markdown("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Century+Gothic:wght@400&display=swap');
                h1 {
                    text-align: center;
                    font-size: 35px;
                    font-family: 'Century Gothic', sans-serif;
                }
                h2 {
                    text-align: center;
                    font-size: 24px;
                    font-family: 'Century Gothic', sans-serif;
                }
                span.custom-font {
                    font-family: 'Century Gothic', sans-serif;
                }
                span.red-text {
                    color: red;
                }
            </style>
            <h2></span> Smile Analysis</h1>
        """, unsafe_allow_html=True)

        # Loading bar and message for smile detection
loading_text = st.text("Please wait. Video is being processed for smile analysis...")
loading_bar_smile = st.progress(0)

        # Process the video and save the output frames
output_frames, message, smile_score = smile_detector(file_path, loading_bar_smile)

        # Encode video frames using PyAV
output_video_bytes = BytesIO()
output_frames = np.array(output_frames)  # Convert frames to numpy array

with av.open(output_video_bytes, 'w', format='mp4') as container:
            stream = container.add_stream('h264', rate=20)  # H264 codec with 20 fps
            for frame in output_frames:
                frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                packet = stream.encode(frame)
                if packet:
                    container.mux(packet)

        # Display the output video
output_video_bytes.seek(0)  # Reset BytesIO object to start
st.video(output_video_bytes, format='video/mp4')
output_frames = []
              
st.markdown(str(message))

        # Display the health bars
data = {'Metric': ['Smile Score'],
            'Score': [int(smile_score)]}

df = pd.DataFrame(data)
st.table(df)

st.write(f"Smile score is given for maintaining smile")

        # Update loading bar and text
loading_bar_smile.progress(100)
loading_text.text("Smile analysis done!")

        #HANDS DET

st.markdown("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Century+Gothic:wght@400&display=swap');
                h1 {
                    text-align: center;
                    font-size: 35px;
                    font-family: 'Century Gothic', sans-serif;
                }
                h2 {
                    text-align: center;
                    font-size: 24px;
                    font-family: 'Century Gothic', sans-serif;
                }
                span.custom-font {
                    font-family: 'Century Gothic', sans-serif;
                }
                span.red-text {
                    color: red;
                }
            </style>
            <h2></span> Hand Analysis</h1>
        """, unsafe_allow_html=True)

        # Loading bar and message for smile detection
loading_text = st.text("Please wait. Video is being processed for hand analysis...")
loading_bar_hand = st.progress(0)

        # Process the video and save the output frames
output_frames, message, hand_score = hand(file_path, loading_bar_hand)

with av.open(output_video_bytes, 'w', format='mp4') as container:
            stream = container.add_stream('h264', rate=20)  # H264 codec with 20 fps
                  
            for frame in output_frames:
                frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                packet = stream.encode(frame)
                if packet:
                    container.mux(packet)

        # Display the output video
output_video_bytes.seek(0)  # Reset BytesIO object to start
st.video(output_video_bytes, format='video/mp4')
output_frames = []
              
st.markdown(str(message))

        # Display the health bars
data = {'Metric': ['Hand Score'],
            'Score': [int(hand_score)]}

df = pd.DataFrame(data)
st.table(df)

st.write(f"Hand score is given for maintaining open hand gestures")

loading_bar_hand.progress(100)
loading_text.text("Hand analysis done!")

        #POSE DET

st.markdown("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Century+Gothic:wght@400&display=swap');
                h1 {
                    text-align: center;
                    font-size: 35px;
                    font-family: 'Century Gothic', sans-serif;
                }
                h2 {
                    text-align: center;
                    font-size: 24px;
                    font-family: 'Century Gothic', sans-serif;
                }
                span.custom-font {
                    font-family: 'Century Gothic', sans-serif;
                }
                span.red-text {
                    color: red;
                }
            </style>
            <h2></span> Pose Analysis</h1>
        """, unsafe_allow_html=True)

        # Loading bar and message for smile detection
loading_text = st.text("Please wait. Video is being processed for pose analysis...")
loading_bar_pose = st.progress(0)

        # Process the video and save the output frames
output_frames, message, pose_score = body(file_path, loading_bar_pose)

with av.open(output_video_bytes, 'w', format='mp4') as container:
            stream = container.add_stream('h264', rate=20)  # H264 codec with 20 fps
                  
            for frame in output_frames:
                frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                packet = stream.encode(frame)
                if packet:
                    container.mux(packet)

        # Display the output video
output_video_bytes.seek(0)  # Reset BytesIO object to start
st.video(output_video_bytes, format='video/mp4')
output_frames = []
              
st.markdown(str(message))

        # Display the health bars
data = {'Metric': ['Pose Score'],
            'Score': [int(pose_score)]}

df = pd.DataFrame(data)
st.table(df)

st.write(f"Pose score is given for maintaining good body pose")

loading_bar_pose.progress(100)
loading_text.text("Pose analysis done!")

os.remove(file_path)

avg_score = (head_score + eye_score + smile_score + hand_score + pose_score)/5
    
        # Retrieve user data
username = st.session_state.username
update_user_scores(username, int(head_score), int(eye_score), int(smile_score), int(hand_score), int(pose_score), current_date, avg_score)
users = c.execute('''
                            SELECT * FROM users
                            WHERE username = ?
                            ''', (username,)).fetchone()

if users:
            st.title("Your Progress")

            table_name = f"{username}_scores"

            # Query scores from the database
            scores_data = c.execute(f'''
                                    SELECT head_score, eye_score, smile_score, hand_score, pose_score, date_added
                                    FROM {table_name}
                                    ''').fetchall()
            
            scores_= c.execute(f'''
                                    SELECT head_score, eye_score, smile_score, hand_score, pose_score
                                    FROM {table_name}
                                    ''').fetchall()

            if scores_data:

                # Query scores from the database, including 'avg_score'
                avgscores_data = c.execute(f'''
                                        SELECT avg_score, date_added
                                        FROM {table_name}
                                        ''').fetchall()

                if avgscores_data:
                    # Create a DataFrame
                    scores_df = pd.DataFrame(avgscores_data, columns=['Avg_Score', 'Date'])
                    
                    # Set the 'Date' column as the index
                    scores_df.set_index('Date', inplace=True)

                    # Calculate the overall average of 'avg_scores'
                    overall_avg_score = scores_df['Avg_Score'].mean()

                    # Define color based on ranges
                    if overall_avg_score < 40:
                        avg_score_color = 'red'
                    elif 40 <= overall_avg_score <= 70:
                        avg_score_color = 'orange'
                    else:
                        avg_score_color = 'green'

                    # Create a curvy line chart using Plotly
                    fig = px.line(scores_df, x=scores_df.index, y='Avg_Score', labels={'Avg_Score': 'Overall Score'}, title = 'Overall score over time')
                    fig.update_traces(line_shape='spline', line_smoothing=0.2, marker=dict(color='blue'))  # You can adjust the line_smoothing for curvature
                    
                    # Display the overall average score above the first graph with specified color
                    st.markdown(f"Average Overall Score:<span style='color: {avg_score_color}; font-size: 40px;'> {overall_avg_score:.2f}</span>", unsafe_allow_html=True)

                    # Display the Plotly figure
                    st.plotly_chart(fig)

                # Create a DataFrame
                scores_df = pd.DataFrame(scores_data, columns=['Head', 'Eye', 'Smile', 'Hand', 'Pose', 'Date'])
                scores = pd.DataFrame(scores_, columns=['Head', 'Eye', 'Smile', 'Hand', 'Pose'])

                # Calculate the average scores
                avg_scores = scores.mean()
                
                # Display the average scores as text
                # Function to format scores with different colors based on ranges
                def format_score_with_color(score):
                    if score < 40:
                        return f'<span style="color: red;">{score:.2f}</span>'
                    elif 40 <= score <= 70:
                        return f'<span style="color: orange;">{score:.2f}</span>'
                    elif 70 < score <= 100:
                        return f'<span style="color: green;">{score:.2f}</span>'
                    else:
                        return f'{score:.2f}'
                
                # Apply the formatting function to each average score
                formatted_avg_scores = avg_scores.apply(format_score_with_color)
                
                # Display the average scores with different colors
                st.markdown(f"Average Scores: Head={formatted_avg_scores['Head']}, Eye={formatted_avg_scores['Eye']}, Smile={formatted_avg_scores['Smile']}, Hand={formatted_avg_scores['Hand']}, Pose={formatted_avg_scores['Pose']}", unsafe_allow_html=True)
                
                # Set the 'Date' column as the index
                scores_df.set_index('Date', inplace=True)

                # Create a curvy line chart using Plotly
                fig = px.line(scores_df, x=scores_df.index, y=scores_df.columns, labels={'value': 'Score'}, title='Scores Over Time')
                fig.update_traces(line_shape='spline', line_smoothing=0.2, marker=dict(size=8, color='blue'))  # Corrected attribute to 'marker'

                n_videos = len(scores_df)

                c.execute('''
                    UPDATE users
                    SET avg_scores = ?,
                        n_videos = ?
                    WHERE username = ?;
                ''', (overall_avg_score, n_videos, username))

                conn.commit()

                #display_tables_and_contents('users.db')
                #print('avg score :', overall_avg_score, 'n videos: ', n_videos)
                
                # Display the Plotly figure
                st.plotly_chart(fig, use_container_width=True)

# Close the database connection
conn.close()
gc.collect()
