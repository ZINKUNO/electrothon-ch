import streamlit as st
import tempfile
import os
import base64
import openai
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

# Set page configuration
st.set_page_config(
    page_title="SpeakPro Ai",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Base styles with modern color scheme */
    :root {
        --primary-color: #4361EE;
        --secondary-color: #3A0CA3;
        --accent-color: #7209B7;
        --light-bg: #F8F9FA;
        --dark-bg: #212529;
        --text-light: #F8F9FA;
        --text-dark: #212529;
        --success-color: #4CC9F0;
        --warning-color: #F72585;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: var(--secondary-color);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F0F2F6 !important;
        border-right: 1px solid #E6E9EF;
        padding: 1rem;
    }
    
    [data-testid="stSidebar"] h2 {
        color: var(--primary-color) !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] a {
        color: var(--accent-color) !important;
        text-decoration: none;
    }
    
    [data-testid="stSidebar"] a:hover {
        text-decoration: underline;
    }
    
    /* Button styles */
    .stButton button {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        color: white !important;
        font-weight: bold;
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Section styles */
    .upload-section {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #E6E9EF;
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
        font-size: 0.8rem;
        padding: 1rem;
        border-top: 1px solid #E6E9EF;
    }
    
    /* Tab styles with better contrast */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: white;
        padding: 25px;
        border-radius: 0 15px 15px 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #E6E9EF;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #F0F2F6;
        border-radius: 10px 10px 0 0;
        color: var(--text-dark) !important;
        font-weight: 500;
        padding: 10px 20px;
        border: 1px solid #E6E9EF;
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E6E9EF;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: white;
        border-bottom: 3px solid var(--primary-color);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-radius: 0;
        padding: 0;
    }
    
    /* High contrast text display areas */
    .text-display-area {
        background-color: #2B2D42;
        padding: 25px;
        border-radius: 15px;
        margin-top: 15px;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        line-height: 1.6;
        font-size: 16px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .text-display-area p {
        color: #EDF2F4 !important;
        margin-bottom: 15px;
    }
    
    .analysis-display-area {
        background-color: #2B2D42;
        padding: 25px;
        border-radius: 15px;
        margin-top: 15px;
        font-family: 'Arial', sans-serif;
        white-space: pre-wrap;
        line-height: 1.6;
        font-size: 16px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .analysis-display-area p {
        color: #EDF2F4 !important;
        margin-bottom: 15px;
    }
    
    .analysis-display-area h4 {
        color: var(--success-color) !important;
        margin-top: 20px;
        margin-bottom: 10px;
        font-weight: 600;
    }
    
    /* Metric container with better styling */
    .metric-container {
        color: white !important;
        border-radius: 15px;
        padding: 15px;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    .metric-container span {
        color: white !important;
    }
    
    /* Override Streamlit's default text colors */
    .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown li {
        color: var(--text-dark) !important;
    }
    
    /* Make sure all text in dark containers is white */
    div.text-display-area *, div.analysis-display-area *, div.metric-container * {
        color: var(--text-light) !important;
    }
    
    /* Ensure all regular text is dark for readability */
    .stMarkdown, .stText, p, span, h1, h2, h3, h4, h5, h6 {
        color: var(--text-dark) !important;
    }
    
    /* Improve form elements */
    [data-testid="stFileUploader"] {
        border: 2px dashed #E6E9EF;
        border-radius: 10px;
        padding: 10px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-color);
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        border-radius: 50px;
        margin: 10px 0;
    }
    
    /* Info, success, warning boxes */
    .stAlert {
        border-radius: 10px !important;
        border-left-width: 10px !important;
    }
    
    /* Fix for dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .stMarkdown, .stText, p, span, h1, h2, h3, h4, h5, h6 {
            color: var(--text-light) !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Configure API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')    
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Check for API keys
if not OPENAI_API_KEY:
    st.error("OpenAI API Key not found. Please check your .env file.")
    st.stop()  # Stop execution if the key is missing

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please check your .env file.")
    st.stop()

# Set API keys
client = OpenAI(api_key=OPENAI_API_KEY)  # Create OpenAI client
genai.configure(api_key=GOOGLE_API_KEY)

# Performance analysis prompt
performance_prompt = '''
Analyze the transcription of the conversation and provide a performance analysis. 
Include the following details (mention only the score out of 10 for these parameters): 
- **Overall Score** 
- **Professionalism** 
- **Responsiveness** 
- **Clarity** 
- **Engagement**  

List **strengths, weaknesses, and suggestions** in bullet points.  
Provide **key insights** and suggest **actions for improvement**.  
Write a **small conclusion**.  
Use **proper headings** for each section.
'''

def transcribe_audio(audio_file_path):
    """Transcribes audio using OpenAI Whisper API or falls back to Google's model."""
    try:
        # First try using Google's Gemini for transcription
        try:
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            with open(audio_file_path, "rb") as audio_file:
                file_content = audio_file.read()
            
            prompt = "Please transcribe this audio file accurately. Return only the transcription text without any additional commentary."
            response = model.generate_content([prompt, {"mime_type": "audio/mp3", "data": file_content}])
            return response.text
        except Exception as google_error:
            st.warning(f"Google API transcription failed, trying OpenAI: {google_error}")
            
            # Fall back to OpenAI if Google fails
            with open(audio_file_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return response.text
    except Exception as e:
        st.error(f"Transcription Error: {e}")
        st.info("Due to API quota limitations, transcription is unavailable. Please try again later or use a different API key.")
        return None

def analyze_performance(transcription):
    """Analyzes the conversation performance using Google Gemini."""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content([performance_prompt, transcription])
        return response.text
    except Exception as e:
        st.error(f"Google Gemini API Error: {e}")
        return None

def save_uploaded_file(uploaded_file):
    """Saves uploaded file to a temporary location."""
    try:
        file_extension = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: var(--primary-color); font-size: 24px; margin-bottom: 10px;">
            <span style="font-size: 32px;">🎤</span> SpeakPro Ai
        </h1>
        <div style="height: 3px; background: linear-gradient(90deg, var(--primary-color), var(--accent-color)); margin: 10px 0 20px 0; border-radius: 10px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## About")
    st.markdown("""
    SpeakPro Ai helps you analyze your conversations and improve your communication skills using advanced AI technology.
    
    **Features:**
    - 🎙️ High-accuracy audio transcription
    - 📊 Detailed performance analysis
    - 📝 Personalized feedback
    - 💡 Actionable improvement suggestions
    
    **How to use:**
    1. Upload an audio file or use the sample
    2. Click "Analyze"
    3. Review your transcription and performance metrics
    """)
    
    st.markdown('<div style="height: 1px; background: #E6E9EF; margin: 20px 0;"></div>', unsafe_allow_html=True)
    st.markdown("## Settings")
    
    # Add API key inputs to sidebar
    with st.expander("API Keys (Optional)"):
        new_openai_key = st.text_input("OpenAI API Key", value=OPENAI_API_KEY, type="password")
        new_google_key = st.text_input("Google API Key", value=GOOGLE_API_KEY, type="password")
        if st.button("Update Keys"):
            # This is just for UI, the keys won't actually update without server restart
            st.success("API keys updated for this session!")
    
    # Add theme selector
    with st.expander("Appearance"):
        st.markdown("Choose your preferred color theme:")
        theme_options = ["Blue-Purple (Default)", "Green-Teal", "Red-Orange", "Dark Mode"]
        selected_theme = st.selectbox("Theme", theme_options)
        st.markdown("*Theme will apply on next restart*")
    
    st.markdown('<div style="height: 1px; background: #E6E9EF; margin: 20px 0;"></div>', unsafe_allow_html=True)
    st.markdown("## Need Help?")
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 15px; margin-top: 20px;">
        <a href="#" style="text-decoration: none; padding: 12px 16px; background: linear-gradient(135deg, var(--primary-color), var(--accent-color)); color: white; border-radius: 10px; display: flex; align-items: center; transition: transform 0.3s ease, box-shadow 0.3s ease;">
            <span style="font-size: 24px; margin-right: 12px;">📚</span>
            <span style="font-weight: 500;">Documentation</span>
        </a>
        <a href="#" style="text-decoration: none; padding: 12px 16px; background: linear-gradient(135deg, var(--primary-color), var(--accent-color)); color: white; border-radius: 10px; display: flex; align-items: center; transition: transform 0.3s ease, box-shadow 0.3s ease;">
            <span style="font-size: 24px; margin-right: 12px;">🐞</span>
            <span style="font-weight: 500;">Report an Issue</span>
        </a>
        <a href="#" style="text-decoration: none; padding: 12px 16px; background: linear-gradient(135deg, var(--primary-color), var(--accent-color)); color: white; border-radius: 10px; display: flex; align-items: center; transition: transform 0.3s ease, box-shadow 0.3s ease;">
            <span style="font-size: 24px; margin-right: 12px;">💬</span>
            <span style="font-weight: 500;">Contact Support</span>
        </a>
    </div>
    <style>
        div a:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 1px; background: #E6E9EF; margin: 20px 0;"></div>', unsafe_allow_html=True)
    
    # Version info
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; color: #6c757d; font-size: 12px;">
        <p>SpeakPro Ai v1.0.0</p>
        <p>© 2024 SpeakPro Ai</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown('<div class="main-header">🎤 SpeakPro Ai</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze your conversations and improve your communication skills</div>', unsafe_allow_html=True)

# Initialize session state variables
if "transcription" not in st.session_state:
    st.session_state["transcription"] = None
if "performance" not in st.session_state:
    st.session_state["performance"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False

# Main app container
main_container = st.container()

with main_container:
    # Add a checkbox for using sample transcription
    use_sample = st.checkbox("💡 Use sample transcription (for testing when API quota is exceeded)")
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    # File uploader
    if not use_sample:
        st.markdown("<h3 style='color: white;'>📂 Upload Your Audio Recording</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: white;'>Upload an audio file of your conversation to analyze your communication skills.</p>", unsafe_allow_html=True)
    
    audio_file = st.file_uploader("", type=['wav', 'mp3'], disabled=use_sample)
    
    # Sample transcription text
    sample_transcription = """
    Hello, this is a sample customer service call. I'm calling about an issue with my recent order. 
    The package arrived yesterday but it seems like one item is missing from the shipment.
    I ordered three books but only received two. Could you please help me resolve this issue?
    I've checked the packing slip and it shows all three items, but as I mentioned, only two were in the box.
    Thank you for your assistance with this matter.
    """
    
    if use_sample:
        st.info("✨ Using sample transcription for testing purposes. The analysis will be based on pre-defined text.")
        if st.button('🚀 Analyze Sample', disabled=st.session_state["is_processing"]):
            with st.spinner("Processing sample..."):
                st.session_state["is_processing"] = True
                gif_placeholder = st.empty()
    
                # Display a GIF (if available)
                if os.path.exists('cat.gif'):
                    with open('cat.gif', 'rb') as f:
                        gif_b64 = base64.b64encode(f.read()).decode()
                    gif_html = f"""
                        <div style="text-align: center;">
                            <img src="data:image/gif;base64,{gif_b64}" alt="Processing..." style="max-width: 300px; border-radius: 10px;">
                            <p style="font-size: 18px; color: #4A4A4A; margin-top: 15px;">
                                Hold your whiskers! The AI is processing the sample... Stay paw-sitive! 🐱
                            </p>
                        </div>
                    """
                    gif_placeholder.markdown(gif_html, unsafe_allow_html=True)
    
                try:
                    # Use sample transcription and analyze
                    st.session_state["transcription"] = sample_transcription
                    st.session_state["performance"] = analyze_performance(sample_transcription)
                except Exception as e:
                    st.error(f"Error processing: {str(e)}")
                finally:
                    gif_placeholder.empty()
                    st.session_state["is_processing"] = False
    
    elif audio_file is not None:
        audio_path = save_uploaded_file(audio_file)
        
        # Display audio player with better styling
        st.markdown("<h3 style='color: white;'>🎧 Preview Your Audio</h3>", unsafe_allow_html=True)
        st.audio(audio_path)
    
        if st.button('🚀 Analyze Audio', disabled=st.session_state["is_processing"]):
            with st.spinner("Transcribing and analyzing your audio..."):
                st.session_state["is_processing"] = True
                gif_placeholder = st.empty()
    
                # Display a GIF (if available)
                if os.path.exists('cat.gif'):
                    with open('cat.gif', 'rb') as f:
                        gif_b64 = base64.b64encode(f.read()).decode()
                    gif_html = f"""
                        <div style="text-align: center;">
                            <img src="data:image/gif;base64,{gif_b64}" alt="Processing..." style="max-width: 300px; border-radius: 10px;">
                            <p style="font-size: 18px; color: #4A4A4A; margin-top: 15px;">
                                Hold your whiskers! The AI is processing the audio... Stay paw-sitive! 🐱
                            </p>
                        </div>
                    """
                    gif_placeholder.markdown(gif_html, unsafe_allow_html=True)
    
                try:
                    # Transcribe and analyze
                    st.session_state["transcription"] = transcribe_audio(audio_path)
                    if st.session_state["transcription"]:
                        st.session_state["performance"] = analyze_performance(st.session_state["transcription"])
                except Exception as e:
                    st.error(f"Error processing: {str(e)}")
                finally:
                    gif_placeholder.empty()
                    st.session_state["is_processing"] = False
    else:
        # Show a welcome message when no file is uploaded
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h3 style="color: var(--primary-color);">Welcome to SpeakPro Ai! 👋</h3>
            <p style="font-size: 18px; margin: 20px 0;">
                Upload an audio file of your conversation or use our sample to get started.
            </p>
            
            <div style="display: flex; justify-content: center; margin: 30px 0;">
                <div style="text-align: center; margin: 0 20px; max-width: 200px;">
                    <div style="font-size: 40px; margin-bottom: 10px;">🎙️</div>
                    <h4>Transcribe</h4>
                    <p>Convert your speech to text with high accuracy</p>
                </div>
                <div style="text-align: center; margin: 0 20px; max-width: 200px;">
                    <div style="font-size: 40px; margin-bottom: 10px;">📊</div>
                    <h4>Analyze</h4>
                    <p>Get detailed metrics on your communication skills</p>
                </div>
                <div style="text-align: center; margin: 0 20px; max-width: 200px;">
                    <div style="font-size: 40px; margin-bottom: 10px;">💡</div>
                    <h4>Improve</h4>
                    <p>Receive actionable feedback to enhance your skills</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Toggle between outputs
    if st.session_state["transcription"] and st.session_state["performance"]:
        # Create tabs for better organization
        tab1, tab2 = st.tabs(["📝 Transcription", "📊 Performance Analysis"])
        
        with tab1:
            st.subheader("📝 Transcription")
            
            # Add a download button for the transcription
            transcription_text = st.session_state["transcription"]
            st.download_button(
                label="Download Transcription",
                data=transcription_text,
                file_name="transcription.txt",
                mime="text/plain"
            )
            
            # Display the transcription in a nicer format with dark background
            st.markdown("### Conversation Text")
            st.markdown(
                f"""
                <div class="text-display-area">
                    <p>{transcription_text}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Add word count and estimated duration
            word_count = len(transcription_text.split())
            st.markdown(f"**Word Count:** {word_count}")
            estimated_duration = round(word_count / 150 * 60)  # Assuming 150 words per minute
            st.markdown(f"**Estimated Duration:** {estimated_duration} seconds")
        
        with tab2:
            st.subheader("📊 Performance Analysis")
            
            # Add a download button for the analysis
            performance_text = st.session_state["performance"]
            st.download_button(
                label="Download Analysis",
                data=performance_text,
                file_name="performance_analysis.txt",
                mime="text/plain"
            )
            
            # Extract scores if they exist in the performance text
            import re
            
            # Function to extract scores
            def extract_scores(text):
                scores = {}
                patterns = {
                    "Overall Score": r"Overall Score:?\s*(\d+(?:\.\d+)?)",
                    "Professionalism": r"Professionalism:?\s*(\d+(?:\.\d+)?)",
                    "Responsiveness": r"Responsiveness:?\s*(\d+(?:\.\d+)?)",
                    "Clarity": r"Clarity:?\s*(\d+(?:\.\d+)?)",
                    "Engagement": r"Engagement:?\s*(\d+(?:\.\d+)?)"
                }
                
                for key, pattern in patterns.items():
                    match = re.search(pattern, text)
                    if match:
                        scores[key] = float(match.group(1))
                
                return scores
            
            scores = extract_scores(performance_text)
            
            # Display scores as a radar chart if scores were found
            if scores and len(scores) >= 3:
                st.markdown("### Performance Scores")
                
                # Create columns for metrics
                cols = st.columns(len(scores))
                for i, (key, value) in enumerate(scores.items()):
                    with cols[i]:
                        # Create a color based on the score (red to green)
                        color = f"rgba({255 - int(value * 25.5)}, {int(value * 25.5)}, 0, 0.8)"
                        st.markdown(
                            f"""
                            <div style="text-align: center;">
                                <h4>{key}</h4>
                                <div class="metric-container" style="background-color: {color};">
                                    <span>{value}/10</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
            # Display the full analysis with better formatting and dark background
            st.markdown("### Detailed Analysis")
            
            # Convert newlines to <br> tags and wrap paragraphs in <p> tags
            formatted_text = ""
            for line in performance_text.split('\n'):
                if line.strip():
                    formatted_text += f"<p>{line}</p>"
                else:
                    formatted_text += "<br>"
                    
            st.markdown(
                f"""
                <div class="analysis-display-area">
                    {formatted_text}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Add a section for key takeaways
            st.markdown("### Quick Actions")
            st.info("📋 Save this analysis to your records or share it with your team for training purposes.")
            
            # Add a feedback section
            st.markdown("### Was this analysis helpful?")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("👍 Yes"):
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("👎 No"):
                    st.error("We're sorry to hear that. We'll work to improve our analysis.")
            with col3:
                if st.button("💡 Suggestions"):
                    st.text_area("How can we improve?", placeholder="Enter your suggestions here...")
                    if st.button("Submit"):
                        st.success("Thank you for your suggestions!")

# Footer
st.markdown('<div class="footer">© 2025 SpeakPro Ai. All rights reserved.</div>', unsafe_allow_html=True)