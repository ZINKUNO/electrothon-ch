export function checkPattern(response, pattern) {
  if (typeof response !== "object" || typeof pattern !== "object") return false;

  for (const key in pattern) {
    if (!(key in response)) return false;

    if (Array.isArray(pattern[key])) {
      if (!Array.isArray(response[key])) return false;

      for (let i = 0; i < pattern[key].length; i++) {
        if (!checkPattern(response[key][i], pattern[key][i])) return false;
      }
    } else if (typeof pattern[key] === "object") {
      if (!checkPattern(response[key], pattern[key])) return false;
    } else if (typeof pattern[key] === "string") {
      if (
        typeof response[key] !== "string" &&
        typeof response[key] !== "number"
      )
        return false;
    }
  }

  return true;
}

export const interviewFeedbackExpectedFormat = {
  technicalFeedback: [
    {
      id: 1,
      feedback: "[feedback_1]",
      rating: "[rating_1]",
    },
    {
      id: 2,
      feedback: "[feedback_2]",
      rating: "[rating_2]",
    },
    {
      id: 3,
      feedback: "[feedback_3]",
      rating: "[rating_2]",
    },
  ],
  analysis: [
    { id: 1, paramName: "Knowledge", percent: "70" },
    { id: 2, paramName: "Precision", percent: "20" },
    { id: 3, paramName: "Confidence", percent: "40" },
  ],
  generalFeedback: "[general_feedback]",
};
