import StarRating from "../components/ui/StarRating";
import feedbackPageStyles from "../styles/feedbackPage.module.css";
import { Link } from "react-router-dom";
import landingPageStyles from "../styles/landingPage.module.css";
import { useEffect, useState } from "react";
import { LOCAL_URL } from "../api";
import axios from "axios";
import { checkPattern, interviewFeedbackExpectedFormat } from "../util";
import Loader from "../components/ui/Loader";

export default function FeedbackPage() {
  const [interviewFeedback, setInterviewFeedback] = useState({});
  const [fallBackMsg, setFallBackMsg] = useState("");
  const [isFeedbackValid, setIsFeedbackValid] = useState(true);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    handleGetFeedback();
  }, []);

  const handleGetFeedback = async () => {
    try {
      const response = await axios.get(`${LOCAL_URL}/api/feedback`);
      if (response?.data?.message) {
        const parsedFeedback = JSON.parse(response?.data?.message);
        const validFeedback = checkPattern(
          parsedFeedback,
          interviewFeedbackExpectedFormat
        );

        setIsFeedbackValid(validFeedback);

        if (!validFeedback) {
          try {
            const response = await axios.get(`${LOCAL_URL}/api/feedback-error`);
            if (response?.data?.message) {
              const fallbackResponse = response?.data?.message;
              setIsLoading(false);
              setFallBackMsg(fallbackResponse);
            }
          } catch (err) {
            setIsLoading(false);
            setFallBackMsg("Apologies! Feedback cannot be generated.");
          }
        } else {
          setIsLoading(false);
          setInterviewFeedback(parsedFeedback);
        }
      }
    } catch (error) {
      setIsLoading(false);
      setFallBackMsg("Apologies! Feedback cannot be generated.");
    }
  };

  return (
    <>
      {isLoading ? (
        <Loader text="SpeakPro Ai is generating customized feedback for you" />
      ) : (
        <div className={feedbackPageStyles.feedbackPageLayout}>
          {isFeedbackValid ? (
            <>
              <div className={feedbackPageStyles.feedbackSection}>
                <h1 className={feedbackPageStyles.centeredTitle}>
                  Technical Feedback
                </h1>
                {interviewFeedback?.technicalFeedback?.map((item) => (
                  <div
                    key={item.id}
                    className={feedbackPageStyles.feedbackContent}
                  >
                    <p className={feedbackPageStyles.feedbackText}>
                      {item.feedback}
                    </p>
                    <StarRating rating={item.rating.toString().split("/")[0]} />
                  </div>
                ))}
              </div>
              <div className={feedbackPageStyles.feedbackSection}>
                <h1 className={feedbackPageStyles.centeredTitle}>Analysis</h1>

                {interviewFeedback?.analysis?.map((item) => {
                  return (
                    <div
                      key={item.id}
                      className={feedbackPageStyles.progressBarContainer}
                    >
                      <p>{item.paramName}</p>
                      <div className={feedbackPageStyles.progress}>
                        <div
                          style={{ width: `${item.percent}%` }}
                          className={feedbackPageStyles.progressValue}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className={feedbackPageStyles.feedbackSection}>
                <h1 className={feedbackPageStyles.centeredTitle}>
                  General Feedback
                </h1>
                <p>{interviewFeedback?.generalFeedback}</p>
              </div>
            </>
          ) : (
            <div className={feedbackPageStyles.feedbackSection}>
              <h1 className={feedbackPageStyles.centeredTitle}>
                Overall Feedback
              </h1>
              <p>{fallBackMsg}</p>
            </div>
          )}
          <Link to="/" className={landingPageStyles.startButton}>
            Go to home page
          </Link>
        </div>
      )}
    </>
  );
}
