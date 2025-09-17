  // --- Generate Image with AI ---
  const generateForm = document.getElementById("generateForm");
  const generateBtn = document.getElementById("generateBtn");
  const generateBtnLabel = generateBtn?.querySelector(".btn-label");
  const generateBtnSpinner = generateBtn?.querySelector(".loading-spinner");
  const generateMessage = document.getElementById("generateMessage");
  const genNumImages = document.getElementById("genNumImages");
  const genMaxObjects = document.getElementById("genMaxObjects");
  // Selected type comes from a single textbox now

  // Single background input (string)
  const genBackgroundInput = document.getElementById("genBackgroundInput");
  const genBlur = document.getElementById("genBlur");
  const genRotate = document.getElementById("genRotate");
  const genNoise = document.getElementById("genNoise");

  // Read static token (if provided via meta tag)
  const tokenMeta = document.querySelector('meta[name="api-static-token"]');
  const STATIC_API_TOKEN = tokenMeta ? tokenMeta.getAttribute('content') : '';

  function withApiHeaders(base = {}) {
    const headers = { ...(base || {}) };
    if (STATIC_API_TOKEN) headers['X-API-Token'] = STATIC_API_TOKEN;
    return headers;
  }

  if (generateForm) {
    generateForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const numImages = genNumImages.value;
      const maxObjects = genMaxObjects.value;
      const resultsList = document.getElementById("resultsList");
      const selectedTypeEl = document.getElementById("selectedType");
      const selectedType = (selectedTypeEl?.value || "").trim();
      const blur = genBlur.value;
      let rotate = [parseInt(genRotate.value || "0", 10)];
      const noise = genNoise.value;
      const backgroundVal = (genBackgroundInput?.value || "").trim();
      if (!numImages || !maxObjects || !selectedType || !backgroundVal) {
        setMessage(generateMessage, "Please fill all fields (type, background, rotate, counts).", 'error');
        return;
      }
      const csrf = getCSRF();
      generateBtn.disabled = true;
      const countBtn = document.getElementById("uploadBtn");
      if (countBtn) countBtn.disabled = true;
      if (generateBtnLabel && generateBtnSpinner) {
        generateBtnLabel.textContent = "Generating…";
        generateBtnSpinner.classList.remove("hidden");
      }
      try {
        const payload = {
          num_images: parseInt(numImages, 10),
          max_objects_per_image: parseInt(maxObjects, 10),
          object_types: [selectedType],
          backgrounds: [backgroundVal],
          blur: parseFloat(blur),
          rotate: rotate,
          noise: parseFloat(noise)
        };
        const resp = await fetch("/api/generate/", {
          method: "POST",
          headers: withApiHeaders({
            "Content-Type": "application/json",
            "X-CSRFToken": csrf
          }),
          body: JSON.stringify(payload)
        });
        const text = await resp.text();
        let data = null;
        try {
          data = text ? JSON.parse(text) : null;
        } catch (e) {
          // non-JSON response
        }
        if (!resp.ok) {
          const msg = formatApiError(data, text, resp.statusText);
          setMessage(generateMessage, `Error (${resp.status}): ${msg}`, 'error');
        } else {
          setMessage(generateMessage, "Images generated and uploaded.", 'success');
          // Show all results
          if (data.results) {
            resultsList.innerHTML = "";
            data.results.forEach(r => {
              if (r.result) resultsList.insertAdjacentHTML("beforeend", resultItemHTML(r.result, csrf));
            });
          }
          // Lock Upload section object type to the generated type and disable it
          const objectTypeSelect = document.getElementById("objectType");
          if (objectTypeSelect) {
            objectTypeSelect.innerHTML = "";
            const opt = document.createElement("option");
            opt.value = selectedType;
            // Keep text simple; backend expects raw value
            opt.textContent = selectedType;
            objectTypeSelect.appendChild(opt);
            objectTypeSelect.value = selectedType;
            objectTypeSelect.disabled = true;
          }
          // Enable finetuned toggle if model now exists (backend returns finetuned_model_dir)
          const toggle = document.getElementById("useFinetunedToggle");
          const label = document.getElementById("useFinetunedLabel");
          if (toggle && label && data && data.finetuned_model_dir) {
            toggle.disabled = false;
            label.textContent = "Use fine‑tuned (if available)";
          }
        }
      } catch (err) {
        setMessage(generateMessage, `Error: ${err}`, 'error');
      } finally {
        generateBtn.disabled = false;
        const countBtn = document.getElementById("uploadBtn");
        if (countBtn) countBtn.disabled = false;
        if (generateBtnLabel && generateBtnSpinner) {
          generateBtnLabel.textContent = "Generate";
          generateBtnSpinner.classList.add("hidden");
        }
        // persist message until next submit
        // Keep form values so the user can adjust parameters; do not auto-reset
      }
    });
  }
function getCSRF() {
  const el = document.querySelector("input[name='csrfmiddlewaretoken']");
  return el ? el.value : "";
}

function getStatusBadgeClass(status) {
  switch (status) {
    case "processing":
      return "badge badge-warning";
    case "predicted":
      return "badge badge-info";
    case "unsafe":
      return "badge badge-error";
    case "failed":
      return "badge badge-error";
    case "corrected":
      return "badge badge-success";
    case "rejected":
      return "badge badge-error";
    default:
      return "badge";
  }
}

// Heuristic detection of unsafe results across variants
function isUnsafeResult(data) {
  const status = (data?.status || '').toString().toLowerCase();
  const meta = data?.meta;
  const reason = data?.reason;
  const metaUnsafe = !!(meta && typeof meta === 'object' && (meta.unsafe === true || meta.pred_label === 'unsafe' || /unsafe/i.test(meta.error || '')));
  const reasonUnsafe = !!(reason && ((typeof reason === 'object' && (reason.pred_label === 'unsafe' || reason.unsafe === true)) || (typeof reason === 'string' && /unsafe/i.test(reason))));
  return status === 'unsafe' || status === 'rejected' || metaUnsafe || reasonUnsafe;
}

// Simple toast using DaisyUI classes; auto-removes after durationMs
function showToast(message, type = 'error', durationMs = 10000) {
  const container = document.createElement('div');
  container.className = 'toast toast-top toast-end z-50';
  const alert = document.createElement('div');
  alert.className = `alert ${type === 'error' ? 'alert-error' : type === 'success' ? 'alert-success' : 'alert-info'}`;
  const span = document.createElement('span');
  span.textContent = message;
  alert.appendChild(span);
  container.appendChild(alert);
  document.body.appendChild(container);
  setTimeout(() => {
    container.remove();
  }, Math.max(0, durationMs || 0));
}

function formatApiError(data, rawText, statusText) {
  if (!data) return rawText || statusText || "Request failed";
  if (typeof data === 'string') return data;
  if (data.error) return data.error;
  if (data.detail) return data.detail;
  // DRF validation errors come as { field: ["msg1", "msg2"], ... }
  try {
    const parts = [];
    Object.entries(data).forEach(([key, val]) => {
      const msgs = Array.isArray(val) ? val.join("; ") : (typeof val === 'string' ? val : JSON.stringify(val));
      parts.push(`${key}: ${msgs}`);
    });
    return parts.join(" | ") || (rawText || statusText || "Bad Request");
  } catch (_) {
    return rawText || statusText || "Bad Request";
  }
}

function resultItemHTML(data, csrfToken) {
  const panopticUrl = data.meta?.panoptic_url;
  const safetyInfo = data.meta?.safety || data.reason;
  const isUnsafe = (data.corrections_allowed === false) || isUnsafeResult(data);
  const unsafeProb = typeof (safetyInfo?.unsafe_prob ?? data.meta?.unsafe_prob) === 'number'
    ? (safetyInfo?.unsafe_prob ?? data.meta?.unsafe_prob).toFixed(2)
    : (safetyInfo?.unsafe_prob ?? data.meta?.unsafe_prob);
  return `
    <li id="r-${data.id}" class="grid gap-4 border border-base-300 rounded-xl p-3 bg-base-100" data-status="${(data.status||'').toString().toLowerCase()}" data-unsafe="${isUnsafe}">
      <div class="grid grid-cols-1 md:grid-cols-[200px_1fr_240px] gap-3 items-center">
        <div>
          <img class="w-48 h-36 object-cover rounded-lg border border-base-300" src="${data.image}" alt="${data.object_type}" />
        </div>
        <div class="info grid gap-1">
          <div><strong>Type:</strong> ${data.object_type}</div>
          <div><strong>Status:</strong> <span class="status ${getStatusBadgeClass(data.status)}">${data.status}</span></div>
          <div><strong>Predicted:</strong> <span class="pred">${data.predicted_count}</span></div>
          ${data.corrected_count !== null ? `<div><strong>Corrected:</strong> <span class="corr">${data.corrected_count}</span></div>` : ""}
          ${isUnsafe ? `
            <div class="alert alert-error mt-1">
              <span>
                This image won't be counted because it has unsafe content.
                ${unsafeProb !== undefined ? `(p_unsafe=${unsafeProb})` : ''}
              </span>
            </div>
          ` : ""}
          <div class="text-sm text-base-content/60">${new Date(data.created_at).toLocaleString()}</div>
        </div>
        ${isUnsafe ? `
          <div class="text-sm text-error mt-1">Corrections are disabled for unsafe images.</div>
        ` : `
          <form class="correctionForm flex flex-col gap-2 mt-1" data-id="${data.id}">
            <input type="hidden" name="csrfmiddlewaretoken" value="${csrfToken}">
            <input class="input input-bordered w-32 mx-auto" type="number" name="corrected_count" min="0" placeholder="correct to…">
            <button type="submit" class="btn btn-sm">Submit correction</button>
          </form>
        `}
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
  // Theme select handler
  const themeSelect = document.getElementById("themeSelect");
  if (themeSelect) {
    themeSelect.addEventListener("change", async () => {
      const theme = themeSelect.value;
      document.documentElement.setAttribute("data-theme", theme);
      const csrf = getCSRF();
      try {
        await fetch("/set-theme/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": csrf,
          },
          body: JSON.stringify({ theme }),
        });
      } catch (e) {
        // no-op; UX remains with the selected theme client-side
      }
    });
  }

  const uploadForm = document.getElementById("uploadForm");
  const uploadBtn = document.getElementById("uploadBtn");
  const uploadBtnLabel = uploadBtn.querySelector(".btn-label");
  const uploadBtnSpinner = uploadBtn.querySelector(".loading-spinner");
  const uploadMessage = document.getElementById("uploadMessage");
  const resultsList = document.getElementById("resultsList");
  const useFinetunedToggle = document.getElementById("useFinetunedToggle");
  const objectTypeSelect = document.getElementById("objectType");
  const defaultTpl = document.getElementById("defaultObjectTypesTpl");
  const finetunedTpl = document.getElementById("finetunedObjectTypesTpl");

  function setSelectOptionsFromTemplate(tpl) {
    if (!tpl || !objectTypeSelect) return;
    const html = tpl.innerHTML.trim();
    if (!html) return;
    const current = objectTypeSelect.value;
    objectTypeSelect.innerHTML = '<option value="" disabled selected>Select type…</option>' + html;
    const opt = Array.from(objectTypeSelect.options).find(o => o.value === current);
    if (opt) objectTypeSelect.value = current;
  }

  function syncObjectTypeToggle() {
    if (!useFinetunedToggle || !objectTypeSelect) return;
    if (!useFinetunedToggle.disabled) {
      if (useFinetunedToggle.checked) {
        if (finetunedTpl && finetunedTpl.innerHTML.trim()) setSelectOptionsFromTemplate(finetunedTpl);
      } else {
        if (defaultTpl && defaultTpl.innerHTML.trim()) setSelectOptionsFromTemplate(defaultTpl);
      }
    }
  }

  if (useFinetunedToggle) {
    useFinetunedToggle.addEventListener("change", syncObjectTypeToggle);
    setTimeout(syncObjectTypeToggle, 0);
  }

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
    if (useFinetunedToggle) {
      // Only send if enabled; server treats missing as auto
      if (!useFinetunedToggle.disabled) {
        fd.append("use_finetuned_classifier", useFinetunedToggle.checked ? "true" : "false");
      }
    }

    uploadBtn.disabled = true;
    if (generateBtn) generateBtn.disabled = true;
    if (uploadBtnLabel && uploadBtnSpinner) {
      uploadBtnLabel.textContent = "Processing…";
      uploadBtnSpinner.classList.remove("hidden");
    }

    let unsafeDetected = false;
    try {
      const resp = await fetch("/api/count/", {
        method: "POST",
        body: fd,
        headers: withApiHeaders({ "X-CSRFToken": csrf })
      });
      const text = await resp.text();
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch (_) { /* non-JSON */ }

      if (!resp.ok) {
        const unsafe = isUnsafeResult(data);
        if (unsafe) {
          const unsafeReason = data?.reason;
          const unsafeProb = typeof unsafeReason?.unsafe_prob === 'number' ? unsafeReason.unsafe_prob.toFixed(2) : (unsafeReason?.unsafe_prob ?? data?.meta?.unsafe_prob);
          setMessage(uploadMessage, `Unsafe image: This image won't be counted.${unsafeProb ? ` (p_unsafe=${unsafeProb})` : ''}`, 'error');
          unsafeDetected = true;
          showToast("Unsafe image: This image won't be counted because it has unsafe content.", 'error', 10000);
          if (data?.id) {
            resultsList.insertAdjacentHTML("beforeend", resultItemHTML(data, csrf));
          }
        } else {
          const msg = formatUploadError(data, text, resp.statusText);
          setMessage(uploadMessage, msg, 'error');
        }
      } else {
        setMessage(uploadMessage, "Done.", 'success');
        // Replace latest result(s) with new one(s)
        resultsList.innerHTML = "";
        const items = Array.isArray(data) ? data : [data];
        items.forEach(item => {
          resultsList.insertAdjacentHTML("beforeend", resultItemHTML(item, csrf));
        });
        // If any item is unsafe, surface a clear message and persistent toast
        if (items.some(isUnsafeResult)) {
          setMessage(uploadMessage, "Image rejected for safety – it was not counted.", 'error');
          unsafeDetected = true;
          showToast("Unsafe image: This image won't be counted because it has unsafe content.", 'error', 10000);
        }
      }
    } catch (err) {
      setMessage(uploadMessage, `Error: ${err}`, 'error');
    } finally {
      uploadBtn.disabled = false;
      if (generateBtn) generateBtn.disabled = false;
      if (uploadBtnLabel && uploadBtnSpinner) {
        uploadBtnLabel.textContent = "Count";
        uploadBtnSpinner.classList.add("hidden");
      }
      // Persist message until the next Count submit; no auto-clear
      uploadForm.reset();
    }
  });

  // Delegate correction submissions (latest + history)
  document.addEventListener("submit", async (e) => {
    if (!e.target.classList.contains("correctionForm")) return;
    e.preventDefault();

    const form = e.target;
    // Client-side guard: prevent corrections on unsafe items
    const li = form.closest("li[id^='r-']");
    if (li) {
      const status = (li.dataset.status || '').toLowerCase();
      const unsafeAttr = (li.dataset.unsafe || '').toString() === 'true';
      if (unsafeAttr || status === 'unsafe' || status === 'rejected') {
        showToast("Corrections are disabled for unsafe images.", 'error', 10000);
        return;
      }
    }
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
        headers: withApiHeaders({
          "Content-Type": "application/json",
          "X-CSRFToken": csrf
        }),
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

// Helper to show persistent message blocks under buttons
function setMessage(container, text, level = 'info') {
  if (!container) return;
  const cls = level === 'success' ? 'alert-success' : level === 'error' ? 'alert-error' : 'alert-info';
  container.innerHTML = `<div class="alert ${cls}"><span>${escapeHtml(text || '')}</span></div>`;
}

function escapeHtml(str) {
  return String(str)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

// Format upload-specific errors nicely (e.g., resolution limits)
function formatUploadError(data, rawText, statusText) {
  if (!data || typeof data !== 'object') {
    return formatApiError(data, rawText, statusText);
  }
  // Single file field error under "image"
  if (data.image !== undefined) {
    const err = Array.isArray(data.image) ? data.image[0] : data.image;
    if (err && typeof err === 'object') {
      const limits = err.limits || {};
      const actual = err.actual || {};
      const minW = parseInt(limits.min_width ?? limits.minWidth ?? 0, 10) || 0;
      const minH = parseInt(limits.min_height ?? limits.minHeight ?? 0, 10) || 0;
      const maxW = parseInt(limits.max_width ?? limits.maxWidth ?? 0, 10) || 0;
      const maxH = parseInt(limits.max_height ?? limits.maxHeight ?? 0, 10) || 0;
      const actW = actual.width !== undefined ? parseInt(actual.width, 10) : undefined;
      const actH = actual.height !== undefined ? parseInt(actual.height, 10) : undefined;
      const base = err.message || 'Please upload a valid image.';
      let msg = base;
      if (minW && minH && maxW && maxH) {
        msg = `Please upload an image between ${minW}×${minH} and ${maxW}×${maxH} pixels.`;
      }
      if (actW && actH) {
        msg += ` Your image is ${actW}×${actH}.`;
      }
      return msg;
    }
    if (typeof err === 'string') return err;
  }
  // Multiple files list error under "images": {index, error}
  if (data.images !== undefined) {
    const first = Array.isArray(data.images) ? data.images[0] : data.images;
    const idx = first?.index;
    const inner = first?.error;
    if (inner && typeof inner === 'object') {
      const limits = inner.limits || {};
      const actual = inner.actual || {};
      const minW = parseInt(limits.min_width ?? 0, 10) || 0;
      const minH = parseInt(limits.min_height ?? 0, 10) || 0;
      const maxW = parseInt(limits.max_width ?? 0, 10) || 0;
      const maxH = parseInt(limits.max_height ?? 0, 10) || 0;
      const actW = actual.width !== undefined ? parseInt(actual.width, 10) : undefined;
      const actH = actual.height !== undefined ? parseInt(actual.height, 10) : undefined;
      let msg = `Image ${typeof idx === 'number' ? `#${idx + 1} ` : ''}has an unsupported resolution.`;
      if (minW && minH && maxW && maxH) {
        msg += ` Allowed range is ${minW}×${minH} to ${maxW}×${maxH} pixels.`;
      }
      if (actW && actH) {
        msg += ` This one is ${actW}×${actH}.`;
      }
      return msg;
    }
  }
  return formatApiError(data, rawText, statusText);
}