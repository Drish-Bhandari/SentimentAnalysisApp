function analyzeReview() {
    let review = document.getElementById("reviewInput").value;
    fetch("/predict", {
        method: "POST",
        body: new URLSearchParams({ "review": review }),
        headers: { "Content-Type": "application/x-www-form-urlencoded" }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = "Sentiment: " + data.sentiment;
    });
}
