import React from "react";
import "./AudioPlayer.css";
import audioFile from '../assets/call-audio.mp3';

function AudioPlayer({ onSectionChange }) {
  return (
    <div className="audio-player">
      <audio controls>
        <source src={audioFile} type="audio/mp3" />
      </audio>
      <div className="buttons">
        <button
          className="button active"
          onClick={() => onSectionChange("transcription")}
        >
          Transcription
        </button>
        <button
          className="button"
          onClick={() => onSectionChange("analysis")}
        >
          Performance Analysis
        </button>
      </div>
    </div>
  );
}

export default AudioPlayer;
