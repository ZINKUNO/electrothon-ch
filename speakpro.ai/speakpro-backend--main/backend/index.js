const express = require("express");
const app = express();
const chatRoute = require("./server.js");
const cors = require("cors");

const corsOptions = {
  origin: ["http://localhost:3000"],
  credentials: true,
  optionSuccessStatus: 200,
};
app.use(cors(corsOptions));
app.use(express.json());

app.use("/api", chatRoute);

app.get("/", (req, res) => {
  res.send("Welcome to SpeakPro Ai backend!");
});
const PORT = process.env.PORT || 5003;
app.listen(PORT, () => console.log(`Server started on port ${PORT}`));