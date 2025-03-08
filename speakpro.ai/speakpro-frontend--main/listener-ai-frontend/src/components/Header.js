import React from "react";
import "./Header.css";

function Header({ onSectionChange }) {
  return (
    <div className="header">
      <h1>SpeakPro Ai</h1>
      <button className="button" onClick={() => window.location.href = "https://verbalmate-ai-listener.streamlit.app"}>
        Record / Upload Audio
      </button>
    </div>
  );
}

export default Header;
