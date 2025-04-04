document.addEventListener("DOMContentLoaded", function () {
    const chatPopup = document.getElementById("chatPopup");
    const chatBody = document.getElementById("chatBody");
    const chatInput = document.getElementById("chatInput");

    chatInput.addEventListener("keypress", function (e) {
        if (e.key === "Enter") {
            const userMessage = chatInput.value;
            chatBody.innerHTML += `<div>User: ${userMessage}</div>`;
            chatInput.value = "";

            fetch("http://localhost:5000/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ question: userMessage })
            })
            .then(response => response.json())
            .then(data => {
                chatBody.innerHTML += `<div>Bot: ${data.response}</div>`;
                chatBody.scrollTop = chatBody.scrollHeight; // Scroll to the bottom
            })
            .catch(error => {
                console.error("Error:", error);
                chatBody.innerHTML += `<div>Bot: Sorry, I couldn't process your request.</div>`;
            });
        }
    });
});
