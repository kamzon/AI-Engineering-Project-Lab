function getCSRF() {
  const el = document.querySelector("#uploadForm input[name='csrfmiddlewaretoken']");
  return el ? el.value : "";
}

function resultItemHTML(data, csrfToken) {
  return `
    <li id="r-${data.id}" class="result">
      <div class="thumb">
        <img src="${data.image}" alt="${data.object_type}" />
      </div>
      <div class="info">
        <div><strong>Type:</strong> ${data.object_type}</div>
        <div><strong>Status:</strong> <span class="status badge status--${data.status}">${data.status}</span></div>
        <div><strong>Predicted:</strong> <span class="pred">${data.predicted_count}</span></div>
        ${data.corrected_count !== null ? `<div><strong>Corrected:</strong> <span class="corr">${data.corrected_count}</span></div>` : ""}
        <div class="small muted">${new Date(data.created_at).toLocaleString()}</div>
      </div>
      <form class="correctionForm" data-id="${data.id}">
        <input type="hidden" name="csrfmiddlewaretoken" value="${csrfToken}">
        <input type="number" name="corrected_count" min="0" placeholder="correct to…">
        <button type="submit" class="btn">Submit correction</button>
      </form>
    </li>
  `;
}

document.addEventListener("DOMContentLoaded", () => {
  const uploadForm = document.getElementById("uploadForm");
  const uploadBtn = document.getElementById("uploadBtn");
  const uploadBtnLabel = uploadBtn.querySelector(".btn-label");
  const uploadBtnSpinner = uploadBtn.querySelector(".spinner");
  const uploadStatus = document.getElementById("uploadStatus");
  const resultsList = document.getElementById("resultsList");

  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const image = document.getElementById("imageInput").files[0];
    const objectType = document.getElementById("objectType").value;
    if (!image || !objectType) return;

    const csrf = getCSRF();
    const fd = new FormData();
    fd.append("image", image);
    fd.append("object_type", objectType);

    uploadBtn.disabled = true;
    if (uploadBtnLabel && uploadBtnSpinner) {
      uploadBtnLabel.textContent = "Processing…";
      uploadBtnSpinner.style.display = "inline-block";
    }
    uploadStatus.textContent = "Processing…";

    try {
      const resp = await fetch("/api/count/", {
        method: "POST",
        body: fd,
        headers: { "X-CSRFToken": csrf }
      });
      const data = await resp.json();

      if (!resp.ok) {
        uploadStatus.textContent = `Error: ${data.detail || JSON.stringify(data)}`;
      } else {
        uploadStatus.textContent = "Done.";
        // Replace latest result with new one
        resultsList.innerHTML = resultItemHTML(data, csrf);
      }
    } catch (err) {
      uploadStatus.textContent = `Error: ${err}`;
    } finally {
      uploadBtn.disabled = false;
      if (uploadBtnLabel && uploadBtnSpinner) {
        uploadBtnLabel.textContent = "Count";
        uploadBtnSpinner.style.display = "none";
      }
      setTimeout(() => (uploadStatus.textContent = ""), 1500);
      uploadForm.reset();
    }
  });

  // Delegate correction submissions (latest + history)
  document.addEventListener("submit", async (e) => {
    if (!e.target.classList.contains("correctionForm")) return;
    e.preventDefault();

    const form = e.target;
    const id = form.dataset.id;
    const input = form.querySelector("input[name='corrected_count']");
    const csrf = form.querySelector("input[name='csrfmiddlewaretoken']")?.value || getCSRF();
    const value = input.value;

    if (value === "") return;

    const btn = form.querySelector("button[type='submit']");
    btn.disabled = true;

    try {
      const resp = await fetch("/api/correct/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": csrf
        },
        body: JSON.stringify({ result_id: id, corrected_count: value })
      });
      const data = await resp.json();

      if (!resp.ok) {
        alert(`Correction failed: ${data.detail || JSON.stringify(data)}`);
      } else {
        const li = document.getElementById(`r-${id}`);
        if (li) {
          const statusEl = li.querySelector(".status");
          statusEl.textContent = data.status;
          statusEl.classList.remove("status--processing", "status--predicted", "status--failed", "status--corrected");
          statusEl.classList.add(`status--${data.status}`);
          const corr = li.querySelector(".corr");
          if (corr) corr.textContent = data.corrected_count;
          else {
            const info = li.querySelector(".info");
            info.insertAdjacentHTML("beforeend",
              `<div><strong>Corrected:</strong> <span class="corr">${data.corrected_count}</span></div>`);
          }
        }
      }
    } catch (err) {
      alert(`Correction failed: ${err}`);
    } finally {
      btn.disabled = false;
      input.value = "";
    }
  });

  // Show history button
  const showHistoryBtn = document.getElementById("showHistoryBtn");
  const historySection = document.getElementById("historySection");
  const historyList = document.getElementById("historyList");

  showHistoryBtn.addEventListener("click", async () => {
    showHistoryBtn.disabled = true;
    try {
      const resp = await fetch("/history/");
      const data = await resp.json();
      historyList.innerHTML = "";
      const csrf = getCSRF();
      data.forEach(item => {
        historyList.insertAdjacentHTML("beforeend", resultItemHTML(item, csrf));
      });
      historySection.style.display = "block";
    } catch (err) {
      alert("Could not load history: " + err);
    } finally {
      showHistoryBtn.disabled = false;
    }
  });
});