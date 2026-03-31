import React, { useState } from "react";
import axios from "axios";

function App() {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append("text", text);

    if (file) {
      formData.append("file", file);
    }

    try {
      const res = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData
      );

      setResult(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>Fake Content Detection</h2>

      <textarea
        rows={4}
        cols={50}
        placeholder="Enter text..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <br />
      <br />

      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <br />
      <br />

      <button onClick={handleSubmit}>Analyze</button>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h3>Result:</h3>

          <p>
            Text Score:{" "}
            {result.text_score !== null
              ? result.text_score.toFixed(3)
              : "N/A"}
          </p>

          <p>
            Image Score:{" "}
            {result.image_score !== null
              ? result.image_score.toFixed(3)
              : "N/A"}
          </p>

          <p>Final Score: {result.final_score}</p>

          <p>Decision: {result.decision}</p>

          <p>
            Confidence:{" "}
            {result.confidence !== undefined
              ? result.confidence
              : "N/A"}
          </p>

          <p>Reason: {result.reason}</p>
        </div>
      )}
    </div>
  );
}

export default App;