  // --- Generate Image with AI ---
  const generateForm = document.getElementById("generateForm");
  const generateBtn = document.getElementById("generateBtn");
  const generateBtnLabel = generateBtn?.querySelector(".btn-label");
  const generateBtnSpinner = generateBtn?.querySelector(".loading-spinner");
  const generateStatus = document.getElementById("generateStatus");
  const genNumImages = document.getElementById("genNumImages");
  const genMaxObjects = document.getElementById("genMaxObjects");
  // Tag input for object types
  const genObjectTypeInput = document.getElementById("genObjectTypeInput");
  const addObjectTypeBtn = document.getElementById("addObjectTypeBtn");
  const genObjectTypesTags = document.getElementById("genObjectTypesTags");
  let objectTypes = [];
  function renderObjectTypes() {
    genObjectTypesTags.innerHTML = objectTypes.map((ot, i) =>
      `<span class="badge badge-info badge-lg flex items-center">${ot} <button type="button" class="ml-1 text-lg font-bold remove-ot" data-idx="${i}">&times;</button></span>`
    ).join("");
  }
  function addObjectType(val) {
    val = val.trim();
    if (val && !objectTypes.includes(val)) {
      objectTypes.push(val);
      renderObjectTypes();
    }
  }
  genObjectTypeInput?.addEventListener("keydown", e => {
    if (e.key === "Enter") {
      e.preventDefault();
      addObjectType(genObjectTypeInput.value);
      genObjectTypeInput.value = "";
    }
  });
  addObjectTypeBtn?.addEventListener("click", () => {
    addObjectType(genObjectTypeInput.value);
    genObjectTypeInput.value = "";
  });
  genObjectTypesTags?.addEventListener("click", e => {
    if (e.target.classList.contains("remove-ot")) {
      const idx = parseInt(e.target.dataset.idx, 10);
      objectTypes.splice(idx, 1);
      renderObjectTypes();
    }
  });

  // Tag input for backgrounds
  const genBackgroundInput = document.getElementById("genBackgroundInput");
  const addBackgroundBtn = document.getElementById("addBackgroundBtn");
  const genBackgroundsTags = document.getElementById("genBackgroundsTags");
  let backgrounds = [];
  function renderBackgrounds() {
    genBackgroundsTags.innerHTML = backgrounds.map((bg, i) =>
      `<span class="badge badge-secondary badge-lg flex items-center">${bg} <button type="button" class="ml-1 text-lg font-bold remove-bg" data-idx="${i}">&times;</button></span>`
    ).join("");
  }
  function addBackground(val) {
    val = val.trim();
    if (val && !backgrounds.includes(val)) {
      backgrounds.push(val);
      renderBackgrounds();
    }
  }
  genBackgroundInput?.addEventListener("keydown", e => {
    if (e.key === "Enter") {
      e.preventDefault();
      addBackground(genBackgroundInput.value);
      genBackgroundInput.value = "";
    }
  });
  addBackgroundBtn?.addEventListener("click", () => {
    addBackground(genBackgroundInput.value);
    genBackgroundInput.value = "";
  });
  genBackgroundsTags?.addEventListener("click", e => {
    if (e.target.classList.contains("remove-bg")) {
      const idx = parseInt(e.target.dataset.idx, 10);
      backgrounds.splice(idx, 1);
      renderBackgrounds();
    }
  });
  const genBlur = document.getElementById("genBlur");
  const genRotate = document.getElementById("genRotate");
  const genNoise = document.getElementById("genNoise");

  if (generateForm) {
    generateForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const numImages = genNumImages.value;
      const maxObjects = genMaxObjects.value;
  if (!objectTypes.length || !backgrounds.length) return;
      const blur = genBlur.value;
      const rotate = Array.from(genRotate.selectedOptions).map(o => parseInt(o.value, 10));
      const noise = genNoise.value;
      if (!numImages || !maxObjects || !objectTypes.length || !backgrounds.length || !rotate.length) return;
      const csrf = getCSRF();
      generateBtn.disabled = true;
      if (generateBtnLabel && generateBtnSpinner) {
        generateBtnLabel.textContent = "Generating…";
        generateBtnSpinner.classList.remove("hidden");
      }
      try {
        const resp = await fetch("/api/generate/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": csrf
          },
          body: JSON.stringify({
            num_images: parseInt(numImages, 10),
            max_objects_per_image: parseInt(maxObjects, 10),
            object_types: objectTypes,
            backgrounds: backgrounds,
            blur: parseFloat(blur),
            rotate: rotate,
            noise: parseFloat(noise)
          })
        });
        const data = await resp.json();
        if (!resp.ok) {
          generateStatus.textContent = `Error: ${data.error || JSON.stringify(data)}`;
        } else {
          generateStatus.textContent = "Images generated and uploaded.";
          // Show all results
          if (data.results) {
            resultsList.innerHTML = "";
            data.results.forEach(r => {
              if (r.result) resultsList.insertAdjacentHTML("beforeend", resultItemHTML(r.result, csrf));
            });
          }
        }
      } catch (err) {
        generateStatus.textContent = `Error: ${err}`;
      } finally {
        generateBtn.disabled = false;
        if (generateBtnLabel && generateBtnSpinner) {
          generateBtnLabel.textContent = "Generate";
          generateBtnSpinner.classList.add("hidden");
        }
        setTimeout(() => (generateStatus.textContent = ""), 2000);
        generateForm.reset();
      }
    });
  }
function getCSRF() {
  const el = document.querySelector("#uploadForm input[name='csrfmiddlewaretoken']");
  return el ? el.value : "";
}

function getStatusBadgeClass(status) {
  switch (status) {
    case "processing":
      return "badge badge-warning";
    case "predicted":
      return "badge badge-info";
    case "failed":
      return "badge badge-error";
    case "corrected":
      return "badge badge-success";
    default:
      return "badge";
  }
}

function resultItemHTML(data, csrfToken) {
  const panopticUrl = data.meta?.panoptic_url;
  return `
    <li id="r-${data.id}" class="grid gap-4 border border-base-300 rounded-xl p-3 bg-base-100">
      <div class="grid grid-cols-1 md:grid-cols-[200px_1fr_240px] gap-3 items-center">
        <div>
          <img class="w-48 h-36 object-cover rounded-lg border border-base-300" src="${data.image}" alt="${data.object_type}" />
        </div>
        <div class="info grid gap-1">
          <div><strong>Type:</strong> ${data.object_type}</div>
          <div><strong>Status:</strong> <span class="status ${getStatusBadgeClass(data.status)}">${data.status}</span></div>
          <div><strong>Predicted:</strong> <span class="pred">${data.predicted_count}</span></div>
          ${data.corrected_count !== null ? `<div><strong>Corrected:</strong> <span class="corr">${data.corrected_count}</span></div>` : ""}
          <div class="text-sm text-base-content/60">${new Date(data.created_at).toLocaleString()}</div>
        </div>
        <form class="correctionForm flex flex-col gap-2 mt-1" data-id="${data.id}">
          <input type="hidden" name="csrfmiddlewaretoken" value="${csrfToken}">
          <input class="input input-bordered w-32 mx-auto" type="number" name="corrected_count" min="0" placeholder="correct to…">
          <button type="submit" class="btn btn-sm">Submit correction</button>
        </form>
      </div>
      ${panopticUrl ? `
            <figure class="diff aspect-[3/2]" tabindex="0">
              <div class="diff-item-1" role="img" tabindex="0">
                <img alt="daisy" src="${data.image}" />
              </div>
              <div class="diff-item-2" role="img">
                <img alt="daisy" src="${panopticUrl}" />
              </div>
              <div class="diff-resizer"></div>
            </figure>
      ` : ""}
    </li>
  `;
}

document.addEventListener("DOMContentLoaded", () => {
  const uploadForm = document.getElementById("uploadForm");
  const uploadBtn = document.getElementById("uploadBtn");
  const uploadBtnLabel = uploadBtn.querySelector(".btn-label");
  const uploadBtnSpinner = uploadBtn.querySelector(".loading-spinner");
  const uploadStatus = document.getElementById("uploadStatus");
  const resultsList = document.getElementById("resultsList");

  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const files = Array.from(document.getElementById("imageInput").files || []);
    const objectType = document.getElementById("objectType").value;
    if (files.length === 0 || !objectType) return;

    const csrf = getCSRF();
    const fd = new FormData();
    // Send either one or multiple images using the 'images' field
    if (files.length === 1) {
      fd.append("image", files[0]);
    } else {
      files.forEach(file => fd.append("images", file));
    }
    fd.append("object_type", objectType);

    uploadBtn.disabled = true;
    if (uploadBtnLabel && uploadBtnSpinner) {
      uploadBtnLabel.textContent = "Processing…";
      uploadBtnSpinner.classList.remove("hidden");
    }

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
        // Replace latest result(s) with new one(s)
        resultsList.innerHTML = "";
        const items = Array.isArray(data) ? data : [data];
        items.forEach(item => {
          resultsList.insertAdjacentHTML("beforeend", resultItemHTML(item, csrf));
        });
      }
    } catch (err) {
      uploadStatus.textContent = `Error: ${err}`;
    } finally {
      uploadBtn.disabled = false;
      if (uploadBtnLabel && uploadBtnSpinner) {
        uploadBtnLabel.textContent = "Count";
        uploadBtnSpinner.classList.add("hidden");
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
          statusEl.className = `status ${getStatusBadgeClass(data.status)}`;
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
      historySection.classList.remove("hidden");
    } catch (err) {
      alert("Could not load history: " + err);
    } finally {
      showHistoryBtn.disabled = false;
    }
  });
});