import axios from "axios";
import { useState } from "react";
import { Link } from "react-router-dom";
import { LOCAL_URL } from "../api";
import { ALL_LEVELS } from "../constants/categories";
import { useAccess } from "../context/AccessContext";
import categoryPageStyles from "../styles/categoryPage.module.css";

function CategoryPage() {
  const { setAccessGranted } = useAccess();
  const [topic, setTopic] = useState(""); // Initialize as an empty string
  const [difficulty, setDifficulty] = useState(ALL_LEVELS[0]);

  function handleDifficulty(level) {
    setDifficulty(level);
  }

  function handleTopicInput(event) {
    setTopic(event.target.value); // Update topic based on input field
  }

  const sendCategoryToBackend = async () => {
    setAccessGranted(true);

    const data = {
      topic: topic.trim(), // Use the entered topic
      difficulty: difficulty.name.toLowerCase(),
    };

    try {
      const response = await axios.post(`${LOCAL_URL}/api/category`, { data });
      console.log("Prompt sent to backend:", response.data.message);
    } catch (error) {
      console.error("Error sending prompt to backend:", error);
    }
  };

  return (
    <div className={categoryPageStyles.mainContainer}>
      <h1 className={categoryPageStyles.mainHeading}>
        Write the situation prompt
      </h1>
      <div className={categoryPageStyles.categoryContainer}>
        <div>
          <h2 className={categoryPageStyles.categoryHeading}>Topic</h2>
          <br></br>
          <textarea
            className={`${categoryPageStyles.inputField} ${categoryPageStyles.largeInput}`}
            placeholder="Enter your situation prompt"
            value={topic}
            onChange={handleTopicInput} // Call handleTopicInput on input change
          />
          <h2 className={categoryPageStyles.categoryHeading}>Level</h2>
          <div className={categoryPageStyles.horizontalLevelContainer}>
            {ALL_LEVELS.map((level) => {
              return (
                <button
                  key={level.id}
                  className={`${categoryPageStyles.categoryBtn} ${
                    level.name.toLowerCase() === difficulty.name.toLowerCase()
                      ? categoryPageStyles.selectedOption
                      : ""
                  }`}
                  onClick={() => handleDifficulty(level)}
                >
                  {level.name}
                </button>
              );
            })}
          </div>
        </div>
      </div>
      <div className={categoryPageStyles.startInterviewBtnContainer}>
        <Link to="/interview">
          <button
            className={categoryPageStyles.startInterviewButton}
            onClick={sendCategoryToBackend}
          >
            Begin Conversation
          </button>
        </Link>
      </div>
    </div>
  );
}

export default CategoryPage;
