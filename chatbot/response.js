async function fetchResponse() {
    try {
        const response = await fetch("http://localhost:8000/get-response");
        if (!response.ok) {
            throw new Error("Network response was not ok.");
        }
        const data = await response.json();
        console.log("Response:", data);
        // Use the response data as needed
    } catch (error) {
        console.error("Error fetching response:", error);
    }
}

fetchResponse();
