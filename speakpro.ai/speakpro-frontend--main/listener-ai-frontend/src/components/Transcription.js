import React from "react";
import "./Transcription.css";

function Transcription() {
  return (
    <div className="section active">
      <h2>Transcript</h2>
      <div className="dialogue-box">
        <div className="dialogue-header">
          <span className="role dispatcher">You</span>
          <span className="timestamp">(0:02 - 0:06)</span>
        </div>
        <div className="dialogue-content">Hello! Good Morning. How are you doing?</div>
      </div>
      <div className="dialogue-box">
        <div className="dialogue-header">
          <span className="role caller">Person 1</span>
          <span className="timestamp">(0:07 - 0:08)</span>
        </div>
        <div className="dialogue-content">
          I am fine. Thanks for asking.
        </div>
      </div>
      <div className="dialogue-box">
        <div className="dialogue-header">
          <span className="role caller">Person 1</span>
          <span className="timestamp">(0:09 - 0:15)</span>
        </div>
        <div className="dialogue-content">
        Hey, did you hear about the new coffee shop that just opened downtown?
        </div>
      </div>
      <div className="dialogue-box">
        <div className="dialogue-header">
          <span className="role dispatcher">You</span>
          <span className="timestamp">(0:25 - 0:27)</span>
        </div>
        <div className="dialogue-content">
        Oh yeah, I saw something about it on Instagram. Have you tried it yet?
        </div>
      </div>
      <div className="dialogue-box">
        <div className="dialogue-header">
          <span className="role caller">Person 1</span>
          <span className="timestamp">(0:42 - 0:44)</span>
        </div>
        <div className="dialogue-content">
          Not yet, but I’m planning to this weekend. They say their caramel latte is amazing.
        </div>
      </div>
      <div className="dialogue-box">
        <div className="dialogue-header">
          <span className="role dispatcher">You</span>
          <span className="timestamp">(0:50 - 0:54)</span>
        </div>
        <div className="dialogue-content">Perfect! Maybe we can check it out together? I’ve been meaning to try something new too.</div>
      </div>
    </div>
  );
}

export default Transcription;
