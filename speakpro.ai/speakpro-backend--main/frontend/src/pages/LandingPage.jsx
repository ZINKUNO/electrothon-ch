import landingPageStyles from "../styles/landingPage.module.css";
import { Link } from "react-router-dom";

export default function LandingPage() {
  return (
    <main className={landingPageStyles.app}>
      <section className={landingPageStyles.container}>
        <div className={landingPageStyles.content}>
          <h1 className={landingPageStyles.heroText}>
            Improve your communication skills by talking to our friendly speak pro AI.
          </h1>
          <p className={landingPageStyles.description}>
            Still struggling to talk properly to a human? Let AI help you instead.
          </p>
          <Link to="https://app-va9ifsbbeykpefh5njajwj.streamlit.app/">
            <button className={landingPageStyles.startButton}>Analyze your conversations</button>
          </Link>
          <Link to="/category">
            <button className={landingPageStyles.startButton}>Start Conversation with AI</button>
          </Link>
          {<Link to="">
            <button className={landingPageStyles.startButton}>Lucy - 3D Conversational AI</button>
          </Link>}
          <Link to="https://framevr.io/tvj-kda-trl#office-hex-1">
            <button className={landingPageStyles.startButton}>Virtual Meeting Room</button>
          </Link>
        </div>
      </section>
    </main>
  );
}
