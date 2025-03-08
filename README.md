# 🎙️ SpeakPro AI
## 🌟 The Problem We Are Trying to Solve
Today many people struggle with improving their conversational skills due to lack of personalized feedback and practice. There is a need for an AI solution that provides real-time analysis and actionable feedback to help individuals enhance their communication abilities. To practice their conversational skills in a simulated environment so they can perform well in the real world.
## 💡 Our Solution
SpeakPro AI is an innovative tool designed to help individuals enhance their conversational skills by turning everyday interactions into valuable learning experiences. By recording, transcribing, and analyzing conversations, SpeakPro AI provides personalized feedback, performance insights, and simulated scenarios, enabling users to improve their communication abilities and gain confidence in any setting.
VerbalMate also provides the opportunity to practice your conversational skills with an Interviewer AI based on any situation you want to simulate by giving it a prompt.
Additionally, with Lucy 3D Conversational AI, users can engage in lifelike, personalized interactions with an AI avatar, adapting to contextual changes and recognizing emotions for a more immersive experience.
The Virtual Meeting Room feature allows users to join virtual meetings in a 3D environment, collaborate in real-time, and maintain anonymity, making it an ideal tool for practicing communication in group settings and professional scenarios.
## ✨ Key Features of SpeakPro AI
## 📊 Analyze Conversations with AI

   ### Performance Analyzer:

#### 🔍 Identifies areas for improvement

#### 🌟 Highlights strengths

##### 🧠 Offers self-reflection insights

#### 😊 Analyzes caller's hidden emotions

#### 📈 Provides scorecards to track progress


### Real-Time Feedback: 🕒 Offers feedback during or after calls
### Continuous Improvement: 🔄 Ensures ongoing skill enhancement

## 🤖 SpeakPro Ai Interviewer AI

Customizable Scenarios: 🎭 Simulate any interview by giving a prompt and setting difficulty level you want
Role-Specific Interviewers: 👔 Practice with tailored interviewers (e.g., HR, technical)
Dynamic Questioning: 🔄 Adapts questions based on your responses
Instant Feedback: ⚡ Get immediate performance insights

## 👩‍💼 Nonverbal AI Vibe

### Motivation

In the competitive job market, effective communication, both verbal and non-verbal, is crucial. Up to 50% of communication relies on body language. Job seekers often struggle with conveying the right non-verbal signals during interviews, impacting their chances of securing employment. This project aims to address this challenge and provide a solution that empowers job seekers to enhance their body language skills.

<img align="center" width="250" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Mehrabian.png/640px-Mehrabian.png" alt="Rule" />

📊 **Statistics** underscore the importance of body language in job interviews. Effective non-verbal communication sends positive messages to interviewers, revealing crucial aspects such as confidence, sincerity, and enthusiasm. Elements like posture, facial expressions, and gestures play a pivotal role in shaping interviewers’ perceptions of a candidate’s suitability for a job.

The frequency and scale of this issue are widespread, affecting job seekers across various demographics. The urgency of addressing this problem is underscored by the competitive nature of the job market and the impact body language has on interview outcomes. The call to action is to equip job seekers with the tools they need to master the art of non-verbal communication and enhance their prospects in the job market.

### Problem Statement

By leveraging MediaPipe, a deep learning framework, this project aims to develop a system that analyzes and provides feedback on various aspects of non-verbal communication. Through facial landmark detection, head pose estimation, eye contact analysis, smile detection, hand gestures, and body pose classification, this system seeks to empower job seekers with insights into their non-verbal behavior during interviews. Ultimately, the goal is to assist them in refining their body language, fostering confidence, and increasing their chances of success in the competitive job market.

## Proposed Solution

🤖 The proposed solution is an **AI-Powered Non-Verbal Communication Coach** that uses advanced computer vision techniques. For estimating the position and orientation of the head, a PnP (Perspective n Point) solver is used to obtain the roll, pitch, and yaw angles of the head. Eye contact detection involves calculating a gaze value. An Eye Aspect Ratio is calculated for blink detection. The smile detection, hand gesture classification, and body pose classification models are trained on labeled datasets of facial, hand, and pose landmark points, respectively.

## Methodology

### Overview

The proposed system integrates various components to aid job seekers in refining their non-verbal communication skills during interviews. The system includes:

1. **Facial Landmark Detection**: Using MediaPipe’s Face Mesh model to identify faces and detect 3D landmark points.
2. **Head Pose Estimation**: Utilizing a subset of facial landmarks to determine head orientation by solving the PnP problem.
3. **Eye Blink Detection**: Employing the Eye Aspect Ratio (EAR) to assess whether the eyes are open or closed.
4. **Eye Contact Detection**: Calculating gaze ratios to evaluate eye contact, considering head position.
5. **Smile Classification**: Training machine learning models on a custom dataset of facial landmarks.
6. **Hand Landmark Detection**: Utilizing MediaPipe’s Hands model to detect hand landmarks.
7. **Hand Gesture Classification**: Classifying hand gestures based on normalized and scaled landmark points.
8. **Pose Landmark Detection**: Using MediaPipe’s Pose model to detect and classify upper body poses.
9. **Feedback Generation**: Providing personalized feedback based on the analysis results.

### Components

1. **Face Landmark Detection**:
   - Utilizes MediaPipe’s Face Mesh model to detect 3D facial landmarks in each frame of the input video.
   - The model comprises the Face Detector (BlazeFace) and the 3D Face Landmark Model (based on ResNet architecture).

2. **Head Pose Estimation**:
   - Estimates head position using specific facial landmarks.
   - Employs a camera matrix to transform 3D coordinates into 2D representations.
   - Uses the Perspective n Point (PnP) algorithm to find rotation and translation vectors.

3. **Eye Blink Detection**:
   - Calculates the Eye Aspect Ratio (EAR) using six landmark points around the eye.
   - Determines the eye state (open or closed) based on the EAR threshold.

4. **Eye Contact Detection**:
   - Calculates gaze ratios for each eye.
   - Determines eye contact based on the gaze ratios and defined threshold values.
   - Uses a variable gaze value for improved accuracy.

5. **Smile Classification**:
   - Uses a custom CSV dataset of facial landmarks labeled with different types of smiles.
   - Normalizes and scales landmarks to train machine learning models for smile classification.

6. **Hand Landmark Detection**:
   - Utilizes MediaPipe’s Hands model to detect hand landmarks from input frames.
   - Includes palm detection (BlazePalm) and hand landmark prediction stages.

7. **Hand Gesture Classification**:
   - Uses a CSV file dataset containing hand landmark points labeled as different hand gestures.
   - Normalizes and scales landmarks to train machine learning models for gesture classification.

8. **Pose Landmark Detection**:
   - Utilizes MediaPipe’s Pose model to detect pose landmarks in each frame of the input video.
   - Predicts 33 landmarks on the human body using a CNN architecture.

9. **Pose Classification**:
   - Uses a CSV file dataset containing pose landmark points labeled as different poses.
   - Normalizes and scales landmarks to train machine learning models for pose classification.

10. **Feedback Generation**:
    - Provides personalized feedback based on observed results.
    - Considers various cues such as smiling, maintaining eye contact, head posture, body poses, and hand gestures.


## 🏢 Virtual Meeting Room

Immersive Interaction: 🌐 Join virtual meetings and interact in a 3D environment
Real-Time Collaboration: 👥 Share ideas, documents, and more with participants during the meeting
Anonymity Options: 🕶️ Participate without revealing your identity, enhancing privacy
Interactive Features: 🛠️ Includes tools for better engagement like whiteboards, group chat, and screen sharing

## 🛠️ Tech Stack
React.js, Node.js, Gemini, Python, Streamlit, ElevenLabs, Groq LLM, Codebuff
## 🚀 Run the project
1. Clone/Download this repo
```
 git clone https://github.com/Harshit-Raj-14/VerbalMate-AI
```
2. Run backend part
```
cd verbalmate--interview-ai
cd backend
```
3. Make a .env file
```
CopyOPENAI_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
ELEVEN_LAB_API_KEY=
node index.js
```
5. Run frontend part
```
cd frontend
npm install
npm start
```
7. Install and run Streamlit app
```
Install Streamlit
pip install streamlit
```
# Navigate to the Streamlit app directory
```
cd ../listener-ai-backend  # Adjust path as needed
```
# Install required packages
```
pip install -r requirements.txt
```
# Run the Streamlit app
```
streamlit run app.py
```
Go to localhost:3000 for the main application and localhost:8501 for the Streamlit interface
🔧 Troubleshooting

If you encounter any API key issues, make sure all the environment variables are properly set
Ensure you have Node.js (v14+) and npm installed on your system
For any port conflicts, check if any other applications are using port 3000 or 8501
If Streamlit installation fails, try using a virtual environment: python -m venv venv and source venv/bin/activate (or venv\Scripts\activate on Windows) before installing

## 📞 Support
Having trouble with SpeakPro AI? Open an issue on our GitHub repository or reach out to our development team.
## 🙏 THANK YOU
We appreciate your interest in SpeakPro AI! We're constantly working to improve conversational skills for everyone.
## 📱 Connect With Us
   ### email:adoranto737@gmail.com

GitHub
Website
