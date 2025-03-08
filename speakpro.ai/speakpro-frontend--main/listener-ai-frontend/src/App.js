import React, { useState } from "react";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import AudioPlayer from "./components/AudioPlayer";
import Transcription from "./components/Transcription";
import PerformanceAnalysis from "./components/PerformanceAnalysis";
import "./App.css";

function App() {
  const [activeSection, setActiveSection] = useState("transcription");

  const handleSectionChange = (section) => {
    setActiveSection(section);
  };

  return (
    <div className="container">
      <Sidebar />
      <div className="content">
        <Header onSectionChange={handleSectionChange} />
        <AudioPlayer onSectionChange={handleSectionChange} />
        {activeSection === "transcription" && <Transcription />}
        {activeSection === "analysis" && <PerformanceAnalysis />}
      </div>
    </div>
  );
}

export default App;
