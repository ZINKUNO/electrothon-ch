# SpeakPro AI

SpeakPro AI helps you analyze your conversations and improve your communication skills using advanced AI technology.

## Features

- 🎙️ High-accuracy audio transcription
- 📊 Detailed performance analysis
- 📝 Personalized feedback
- 💡 Actionable improvement suggestions

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- Python (v3.8 or higher)
- Streamlit

### Installation

1. Clone this repository
2. Install backend dependencies:
   ```
   cd verbalmate-backend--main/backend
   npm install
   ```
3. Install frontend dependencies:
   ```
   cd ../frontend
   npm install
   ```
4. Install Streamlit app dependencies:
   ```
   cd ../../verbalmate-frontend--main/listener-ai-backend
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the backend server:
   ```
   cd verbalmate-backend--main/backend
   node index.js
   ```
2. Start the frontend application:
   ```
   cd ../frontend
   npm start
   ```
3. Start the Streamlit app:
   ```
   cd ../../verbalmate-frontend--main/listener-ai-backend
   streamlit run app.py
   ```

## Usage

1. Open your browser and navigate to http://localhost:3000
2. Click on "Analyze your conversations" to use the conversation analyzer
3. Click on "Start Conversation with AI" to practice your communication skills with AI

## Troubleshooting

- If the "Analyze your conversations" button doesn't work, make sure the Streamlit app is running on port 8501
- If you encounter any issues with the backend, check if port 5002 is already in use 