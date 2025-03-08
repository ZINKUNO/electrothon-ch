import React from "react";
import "./PerformanceAnalysis.css";

function PerformanceAnalysis() {
  return (
    <div className="section">
      <h2>Performance Analysis</h2>
      <div className="scores">
        <div className="score-box">
          <h3>8/10</h3>
          <p>Total Score</p>
        </div>
        <div className="score-box">
          <h3>7/10</h3>
          <p>Professionalism</p>
        </div>
        <div className="score-box">
          <h3>8/10</h3>
          <p>Responsiveness</p>
        </div>
      </div>
      <div>
        <h3>Strengths</h3>
        <ul>
          <li>You initiated the conversations and kept them engaging by sharing interesting topics.</li>
        </ul>
        <h3>Weaknesses</h3>
        <ul>
          <li>You don’t explore alternative options in case Person 1 isn’t available or doesn’t like the suggestion.</li>
        </ul>
        <h3>Improvements</h3>
        <ul>
          <li>You could provide a few alternative options to ensure the plan is flexible and collaborative.</li>
        </ul>
      </div>
    </div>
  );
}

export default PerformanceAnalysis;
