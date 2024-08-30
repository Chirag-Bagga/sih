// // src/Chatbot.js
// import React, { useState } from 'react';

// const Chatbot = () => {
//     const [input, setInput] = useState('');
//     const [messages, setMessages] = useState([]);

//     const handleSubmit = async (e) => {
//         e.preventDefault();

//         // Add user message to chat
//         setMessages([...messages, { type: 'user', text: input }]);
//         setInput('');

//         try {
//             // Send the question to the backend
//             const post_response = await fetch('http://localhost:8000/post-response', {
//                 method: 'POST',
//                 headers: { 'Content-Type': 'application/json' },
//                 body: JSON.stringify({ question: input })
//             });

//             if (!post_response.ok) throw new Error('Error in posting');

//             const get_response = await fetch('http://localhost:8000/get-response',
//                 {
//                     method: "GET",
//                     headers: { 'Content-Type': 'application/json' }
//                 }
//             )

//             if(!get_response.ok) throw new Error(' Error in getting the data');
            
//             const data = await get_response.json();

//             // Add chatbot response to chat
//             setMessages([...messages, { type: 'user', text: input }, { type: 'bot', text: data.answer }]);
//         } catch (error) {
//             console.error('Error fetching response:', error);
//         }
//     };

//     return (
//         <div>
//             <div className="chatbox">
//                 <div className="messages">
//                     {messages.map((msg, index) => (
//                         <div key={index} className={msg.type}>
//                             {msg.text}
//                         </div>
//                     ))}
//                 </div>
//                 <form onSubmit={handleSubmit}>
//                     <input
//                         type="text"
//                         value={input}
//                         onChange={(e) => setInput(e.target.value)}
//                         placeholder="Ask a question..."
//                         required
//                     />
//                     <button type="submit">Send</button>
//                 </form>
//             </div>
//             <style>
//                 {`
//                 .chatbox {
//                     width: 300px;
//                     margin: 0 auto;
//                     border: 1px solid #ccc;
//                     padding: 10px;
//                     border-radius: 5px;
//                 }
//                 .messages {
//                     max-height: 200px;
//                     overflow-y: auto;
//                     margin-bottom: 10px;
//                 }
//                 .user {
//                     text-align: right;
//                     background-color: #e1ffc7;
//                     padding: 5px;
//                     border-radius: 5px;
//                     margin: 5px 0;
//                 }
//                 .bot {
//                     text-align: left;
//                     background-color: #f1f1f1;
//                     padding: 5px;
//                     border-radius: 5px;
//                     margin: 5px 0;
//                 }
//                 form {
//                     display: flex;
//                     align-items: center;
//                 }
//                 input {
//                     flex: 1;
//                     padding: 5px;
//                     margin-right: 10px;
//                     border: 1px solid #ccc;
//                     border-radius: 5px;
//                 }
//                 button {
//                     padding: 5px 10px;
//                     border: none;
//                     background-color: #007bff;
//                     color: #fff;
//                     border-radius: 5px;
//                     cursor: pointer;
//                 }
//                 `}
//             </style>
//         </div>
//     );
// };

// export default Chatbot;
import React, { useState } from "react";

function QuestionForm() {
  const [question, setQuestion] = useState("");
  const [responseMessage, setResponseMessage] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Create the payload to send
    const questionData = {
      text: question,
      // Add any other fields that your backend expects here
    };

    try {
      const response = await fetch("http://localhost:8000/post-question", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(questionData),
      });

      // Parse the JSON response
      const data = await response.json();

      if (response.ok) {
        setResponseMessage(data.message || "Question saved successfully!");
      } else {
        setResponseMessage(data.error || "An error occurred.");
      }
    } catch (error) {
      console.error("Error:", error);
      setResponseMessage("An error occurred while sending the request.");
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <label>
          Question:
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            required
          />
        </label>
        <button type="submit">Submit</button>
      </form>
      {responseMessage && <p>{responseMessage}</p>}
    </div>
  );
}

export default QuestionForm;
