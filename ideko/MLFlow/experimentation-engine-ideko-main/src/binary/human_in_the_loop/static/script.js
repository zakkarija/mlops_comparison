//API_URL = "http://localhost:5000"
API_URL = "http://" + backend_ip_and_port + "/"

document.addEventListener("DOMContentLoaded", async () => {
    const fileContainer = document.getElementById("fileContainer");
    fileContainer.innerHTML = `
        <table border="1" cellspacing="0" cellpadding="5">
            <thead>
                <tr>
                    <th>Filename</th>
                    <th>Result</th>
                    <th>Feedback</th>
                </tr>
            </thead>
            <tbody id="fileTableBody"></tbody>
        </table>
    `;
    
    const fileTableBody = document.getElementById("fileTableBody");
    const modal = document.getElementById("modal");
    const modalMessage = document.getElementById("modalMessage");
    const confirmBtn = document.getElementById("confirmBtn");
    const cancelBtn = document.getElementById("cancelBtn");
    let selectedFile = null;
    let selectedValue = null;

    const response = await fetch(API_URL + "/files");
    const files = await response.json();
    const colors = ["red", "green"];
    filesList = files["files"]

    filesList.forEach((file, index) => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${file}</td>
            <td style="text-align: center;">
                <div class="circle" style="background-color: ${colors[index % colors.length]}; width: 20px; height: 20px; border-radius: 50%; border: 2px solid black; margin: auto;"></div>
            </td>
            <td style="text-align: center;">
                <button class="check" data-file="${file}" data-value="1">✔️</button>
                <button class="cross" data-file="${file}" data-value="0">❌</button>
            </td>
        `;
        fileTableBody.appendChild(row);
    });

    fileTableBody.addEventListener("click", (event) => {
        if (event.target.classList.contains("check") || event.target.classList.contains("cross")) {
            selectedFile = event.target.getAttribute("data-file");
            selectedValue = event.target.getAttribute("data-value");
            modalMessage.textContent = selectedValue === "1" 
                ? "You have selected that the model prediction has been correct. Are you sure you want to send this feedback to the system?" 
                : "You have selected that the model prediction has been incorrect. Are you sure you want to send this feedback to the system?";
            modal.style.display = "block";
        }
    });

    confirmBtn.addEventListener("click", async () => {
        modal.style.display = "none";
        if (selectedFile !== null)
            {
            const res = await fetch(API_URL + "/continue", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ file: selectedFile, value: selectedValue })
            });
            if (res.status === 200)
            {
                modalMessage.textContent = "Feedback sent successfully!";
                modal.style.display = "block";
                confirmBtn.style.display = "none";
                cancelBtn.style.display = "none";
            }
        }
    });

    cancelBtn.addEventListener("click", () => {
        modal.style.display = "none";
    });
});
