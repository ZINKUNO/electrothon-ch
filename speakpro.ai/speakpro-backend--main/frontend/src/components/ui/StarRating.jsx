import React from "react";
import starStyles from "../../styles/starRating.module.css";

const StarRating = ({ rating }) => {
  const starArray = [1, 2, 3, 4, 5];

  return (
    <div className={starStyles.starRating}>
      {starArray.map((starIndex) => (
        <span
          key={starIndex}
          className={`${starStyles.star} ${
            starIndex <= rating ? starStyles.filled : ""
          }`}
        >
          ★
        </span>
      ))}
    </div>
  );
};

export default StarRating;
