import { useEffect, useRef, useState } from "react";
import interviewStyles from "../styles/interviewPage.module.css";
import interviewer from "../assets/interviewer.png";
import interviewee from "../assets/interviewee.png";
import axios from "axios";
import SpeechRecognition, {
  useSpeechRecognition,
} from "react-speech-recognition";
import { useNavigate } from "react-router-dom";
import { LOCAL_URL } from "../api";
import Loader from "../components/ui/Loader";

export default function InterviewPage() {
  const navigate = useNavigate();
  const [aiResponse, setAiResponse] = useState("");
  const [isInterviewerSpeaking, setIsInterviewerSpeaking] = useState(false);
  const [loading, setLoading] = useState(true);
  const [glowingEffect, setGlowingEffect] = useState({
    user: false,
    interviewer: false,
  });
  const isRun = useRef(false);
  const speechSynthesisRef = useRef(null);

  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition,
  } = useSpeechRecognition();

  // Cleanup function for speech synthesis
  useEffect(() => {
    return () => {
      if (speechSynthesisRef.current) {
        speechSynthesis.cancel();
      }
    };
  }, []);

  // starting loading before interview starts
  useEffect(() => {
    setTimeout(() => {
      setLoading(false);
    }, 3000);
  }, []);

  // send empty script on initial load so that ai starts responding
  useEffect(() => {
    const firstRequestToGemini = async () => {
      try {
        const response = await axios.post(`${LOCAL_URL}/api/chat`, {
          transcript: "",
        });
        setAiResponse(response.data.content);
        speak(response.data.content);
      } catch (error) {
        console.error("Error in first request:", error);
        setIsInterviewerSpeaking(false);
      }
    };

    if (isRun.current === true) return;
    isRun.current = true;

    setTimeout(() => {
      firstRequestToGemini();
    }, 2000);
  }, []);

  useEffect(() => {
    const interviewComplete = "Thank you for talking with SpeakPro Ai";

    if (
      aiResponse?.includes(interviewComplete) &&
      isInterviewerSpeaking === false
    ) {
      navigate("/feedback");
    }
  }, [aiResponse, isInterviewerSpeaking, navigate]);

  if (!browserSupportsSpeechRecognition) {
    return <span>Browser doesn't support speech recognition. This project best works on microsoft edge.</span>;
  }

  const handleStartListeningUser = () => {
    setGlowingEffect((prev) => ({ ...prev, user: true }));
    SpeechRecognition.startListening({
      continuous: true,
      language: "en-IN",
      interimResults: true,
    });
  };

  const handleStopListeningUser = async () => {
    setGlowingEffect((prev) => ({ ...prev, user: false }));
    SpeechRecognition.stopListening();

    try {
      const response = await axios.post(`${LOCAL_URL}/api/chat`, {
        transcript,
      });
      setAiResponse(response.data.content);
      speak(response.data.content);
      resetTranscript();
    } catch (error) {
      console.error("Error in chat request:", error);
      setIsInterviewerSpeaking(false);
    }
  };

  function speak(text) {
    // Cancel any ongoing speech
    speechSynthesis.cancel();
    
    setGlowingEffect((prev) => ({ ...prev, interviewer: true }));
    setIsInterviewerSpeaking(true);

    const utterance = new SpeechSynthesisUtterance(text);
    const voices = speechSynthesis.getVoices();
    utterance.voice = voices[0];
    utterance.lang = "en-IN";

    // Handle both successful completion and errors
    utterance.onend = () => {
      setIsInterviewerSpeaking(false);
      setGlowingEffect((prev) => ({ ...prev, interviewer: false }));
      speechSynthesisRef.current = null;
    };

    utterance.onerror = (event) => {
      console.error("Speech synthesis error:", event);
      setIsInterviewerSpeaking(false);
      setGlowingEffect((prev) => ({ ...prev, interviewer: false }));
      speechSynthesisRef.current = null;
    };

    // Store the utterance reference
    speechSynthesisRef.current = utterance;
    
    try {
      speechSynthesis.speak(utterance);
    } catch (error) {
      console.error("Speech synthesis error:", error);
      setIsInterviewerSpeaking(false);
      setGlowingEffect((prev) => ({ ...prev, interviewer: false }));
      speechSynthesisRef.current = null;
    }
  }

  return (
    <>
      {loading ? (
        <Loader text="VerbalMate is generating your conversation. Almost there...." />
      ) : (
        <div className={interviewStyles.interviewLayout}>
          <div className={interviewStyles.interviewLHS}>
            <div
              className={`${interviewStyles.sectionBackground} ${interviewStyles.interviewerSection}`}
            >
              <p className={interviewStyles.sectionTitle}>Interviewer</p>
              <img
                src={interviewer}
                alt="AI Interviewer"
                className={`${interviewStyles.interviewerProfile} ${
                  glowingEffect.interviewer ? interviewStyles.glowingEffect : ""
                }`}
              />
            </div>
            <div
              className={`${interviewStyles.sectionBackground} ${interviewStyles.intervieweeSection}`}
            >
              <p className={interviewStyles.sectionTitle}>You</p>
              <img
                src={interviewee}
                alt="AI Interviewee"
                className={`${interviewStyles.intervieweeProfile} ${
                  glowingEffect.user ? interviewStyles.glowingEffect : ""
                }`}
              />
              <div className={interviewStyles.intervieweeControlButtonGroup}>
                <button
                  onClick={handleStartListeningUser}
                  className={`${interviewStyles.intervieweeControlButton} ${
                    isInterviewerSpeaking || listening
                      ? interviewStyles.intervieweeControlButtonInactive
                      : ""
                  }`}
                  disabled={isInterviewerSpeaking || listening}
                >
                  Start answering
                </button>
                <button
                  onClick={handleStopListeningUser}
                  className={`${interviewStyles.intervieweeControlButton} ${
                    isInterviewerSpeaking || !listening
                      ? interviewStyles.intervieweeControlButtonInactive
                      : ""
                  }`}
                  disabled={isInterviewerSpeaking || !listening}
                >
                  Stop answering
                </button>
              </div>
            </div>
          </div>
          <div className={interviewStyles.interviewRHS}>
            <div
              className={`${interviewStyles.sectionBackground} ${interviewStyles.qASection}`}
            >
              <p className={interviewStyles.sectionTitle}>Question</p>
              <p className={interviewStyles.sectionContent}>{aiResponse}
                Please watch the video demo to view this feature in action as we have exhausted our api key.
              </p>
            </div>
            <div
              className={`${interviewStyles.sectionBackground} ${interviewStyles.qASection} ${interviewStyles.answerSection}`}
            >
              <p className={interviewStyles.sectionTitle}>Answer</p>
              <p className={interviewStyles.sectionContent}>{transcript}</p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}