import React from "react";
import "./Sidebar.css";

function Sidebar() {
  return (
    <div className="sidebar">
      <h2>Calls</h2>
      <div className="call active">
        Conversation [1]<br />
        <small>03:18</small>
      </div>
      <div className="pagination">
        <button>Previous</button>
        <span>1/1</span>
        <button>Next</button>
      </div>
    </div>
  );
}

export default Sidebar;
