const express = require("express");
const router = express.Router();
const Groq = require("groq-sdk");
require("dotenv").config();

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

let messages = [];

router.post("/category", async (req, res) => {
  try {
    const { topic, difficulty } = req.body.data;

    messages = [
      {
        role: "system",
        content: `You are SpeakPro Ai an interviewer who should ask only 3 ${difficulty} questions one by one related to ${topic}. Greet the user and directly start with the question. After the user responds, give a very short explanation and move to next question. Once all the questions are done, say "Thank you for interviewing with SpeakPro Ai."`,
      },
    ];

    res.status(200).json({ message: "Success: Prompt updated" });
  } catch (err) {}
});

router.post("/chat", async (req, res) => {
  try {
    const userResponse = req.body.transcript;
    messages.push({ role: "user", content: userResponse });

    const chatCompletion = await groq.chat.completions.create({
      messages: messages,
      model: "llama3-8b-8192",
      temperature: 1,
      max_tokens: 1024,
      top_p: 1,
      stream: true,
      stop: null,
    });

    let response = "";
    for await (const chunk of chatCompletion) {
      const delta = chunk.choices[0]?.delta?.content || "";
      response += delta;
    }

    messages.push({ role: "assistant", content: response });
    res.json({ role: "assistant", content: response });
  } catch (error) {
    res.status(500).json({ error: "Error during conversation" });
  }
});

router.get("/feedback", async (req, res) => {
  const interviewFeedback = {
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

  const prompt = `Analyse the conversation from ${JSON.stringify(
    messages
  )} to give detailed feedback and format it strictly according to ${JSON.stringify(
    interviewFeedback
  )}. Ensure the result you will provide contains only the object data without any surrounding text or additional strings. Dont write 
  "here is the response" or anything additional`;

  try {
    const feedbackMessage = [
      {
        role: "system",
        content: prompt,
      },
    ];

    const chatCompletion = await groq.chat.completions.create({
      messages: feedbackMessage,
      model: "llama3-8b-8192",
      temperature: 1,
      max_tokens: 1024,
      top_p: 1,
      stream: true,
      stop: null,
    });

    let response = "";
    for await (const chunk of chatCompletion) {
      const delta = chunk.choices[0]?.delta?.content || "";
      response += delta;
    }
    res.status(200).json({ message: response });
  } catch (err) {}
});

router.get("/feedback-error", async (req, res) => {
  const fallbackPrompt = `Analyse the conversation from ${JSON.stringify(
    messages
  )} to give detailed constructive feedback in first person. Ensure the result you will provide contains only the feedback without any surrounding text or additional strings. Don't write 
  "here is the response" or anything additional`;

  try {
    const fallbackMessage = [
      {
        role: "system",
        content: fallbackPrompt,
      },
    ];

    const chatCompletion = await groq.chat.completions.create({
      messages: fallbackMessage,
      model: "llama3-8b-8192",
      temperature: 1,
      max_tokens: 1024,
      top_p: 1,
      stream: true,
      stop: null,
    });

    let response = "";
    for await (const chunk of chatCompletion) {
      const delta = chunk.choices[0]?.delta?.content || "";
      response += delta;
    }

    res.status(200).json({ message: response });
  } catch (err) {}
});

module.exports = router;