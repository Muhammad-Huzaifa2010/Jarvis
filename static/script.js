
const API = "";

const fileInput = document.getElementById("fileInput");
const uploadBox = document.getElementById("uploadBox");
const uploadStatus = document.getElementById("uploadStatus");
const documentList = document.getElementById("documentList");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const chatArea = document.getElementById("chatArea");
const toast = document.getElementById("toast");
const sidebar = document.querySelector(".sidebar");

let isSending = false;
let isUploading = false;
let isLoadingDocuments = false;

const uploadedDocuments = new Map();

// --------------------------------------------------
// Notifications and chat helpers
// --------------------------------------------------

function showToast(message) {
  if (!toast) return;

  toast.textContent = message;
  toast.classList.add("show");

  clearTimeout(showToast.timer);

  showToast.timer = setTimeout(() => {
    toast.classList.remove("show");
  }, 3500);
}

function scrollToBottom() {
  if (chatArea) {
    chatArea.scrollTop = chatArea.scrollHeight;
  }
}

function hideWelcome() {
  if (welcome) {
    welcome.hidden = true;
  }
}

// --------------------------------------------------
// Display chat messages and citations
// --------------------------------------------------

function addMessage(role, text, sources = [], citations = []) {
  hideWelcome();

  const message = document.createElement("article");
  message.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Y" : "J";

  const content = document.createElement("div");
  content.className = "message-content";

  if (role === "assistant") {
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = "JARVIS";
    content.appendChild(label);
  }

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  content.appendChild(body);

  // Display citations when the API provides them.
  if (citations.length > 0) {
    const sourceBox = document.createElement("div");
    sourceBox.className = "sources";

    const title = document.createElement("strong");
    title.textContent = "Sources & Citations";
    sourceBox.appendChild(title);

    // Remove identical filename/excerpt entries.
    const uniqueCitations = [];
    const seen = new Set();

    citations.forEach((citation) => {
      if (
        !citation ||
        typeof citation.filename !== "string"
      ) {
        return;
      }

      const key = JSON.stringify([
        citation.filename,
        citation.excerpt || "",
      ]);

      if (!seen.has(key)) {
        seen.add(key);
        uniqueCitations.push(citation);
      }
    });

    uniqueCitations.forEach((citation) => {
      const item = document.createElement("details");
      item.className = "citation-item";

      const heading = document.createElement("summary");
      heading.className = "citation-heading";

      const filename = document.createElement("span");
      filename.className = "citation-filename";
      filename.textContent = citation.filename;

      const reference = document.createElement("span");
      reference.className = "source-chip";
      reference.textContent = `[${citation.reference || "Source"}]`;

      heading.append(filename, reference);

      // Page metadata is optional. Do not invent page numbers.
      if (
        citation.page !== null &&
        citation.page !== undefined &&
        Number.isFinite(Number(citation.page))
      ) {
        const page = document.createElement("span");
        page.className = "citation-page";

        // PDF loaders commonly use zero-based page indices.
        // DOCX citations in the current API return null.
        page.textContent = `Page ${Number(citation.page) + 1}`;

        heading.appendChild(page);
      }

      item.appendChild(heading);

      if (
        typeof citation.excerpt === "string" &&
        citation.excerpt.trim()
      ) {
        const excerpt = document.createElement("p");
        excerpt.className = "citation-excerpt";
        excerpt.textContent = citation.excerpt;
        item.appendChild(excerpt);
      }

      sourceBox.appendChild(item);
    });

    if (uniqueCitations.length > 0) {
      content.appendChild(sourceBox);
    }
  } else if (sources.length > 0) {
    // Backward-compatible display if citations are unavailable.
    const sourceBox = document.createElement("div");
    sourceBox.className = "sources";

    const title = document.createElement("strong");
    title.textContent = "Sources";
    sourceBox.appendChild(title);

    [...new Set(sources)].forEach((source) => {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.textContent = source;
      sourceBox.appendChild(chip);
    });

    content.appendChild(sourceBox);
  }

  message.append(avatar, content);
  messages.appendChild(message);

  scrollToBottom();

  return message;
}

function addTypingIndicator() {
  const message = addMessage("assistant", "Jarvis is thinking...");

  const body = message.querySelector(".message-body");

  if (body) {
    body.classList.add("typing");
  }

  return message;
}

// --------------------------------------------------
// Document list
// --------------------------------------------------

function renderEmptyDocuments(
  message = "No documents uploaded yet."
) {
  documentList.replaceChildren();

  const emptyMessage = document.createElement("p");
  emptyMessage.className = "file-meta";
  emptyMessage.textContent = message;

  documentList.appendChild(emptyMessage);
}

function renderDocuments() {
  documentList.replaceChildren();

  if (uploadedDocuments.size === 0) {
    renderEmptyDocuments();
    return;
  }

  for (const [name, count] of uploadedDocuments) {
    const item = document.createElement("div");
    item.className = "document-item";

    const icon = document.createElement("div");
    icon.className = "file-icon";

    icon.textContent = name.includes(".")
      ? name.split(".").pop().toUpperCase()
      : "FILE";

    const details = document.createElement("div");
    details.className = "file-details";

    const title = document.createElement("div");
    title.className = "file-name";
    title.textContent = name;
    title.title = name;

    const meta = document.createElement("div");
    meta.className = "file-meta";
    meta.textContent = `${count} chunks stored`;

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "delete-document-btn";
    deleteButton.textContent = "Delete";
    deleteButton.setAttribute(
      "aria-label",
      `Delete ${name}`
    );

    deleteButton.addEventListener("click", () => {
      deleteDocument(name, deleteButton);
    });

    details.append(title, meta);
    item.append(icon, details, deleteButton);

    documentList.appendChild(item);
  }
}

async function loadDocuments() {
  if (isLoadingDocuments) return;

  isLoadingDocuments = true;

  try {
    const response = await fetch(`${API}/documents`);
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(
        data.detail || "Could not load documents."
      );
    }

    if (!Array.isArray(data.documents)) {
      throw new Error(
        "The server returned an invalid document list."
      );
    }

    uploadedDocuments.clear();

    for (const document of data.documents) {
      if (
        typeof document.filename === "string" &&
        Number.isFinite(Number(document.chunks))
      ) {
        uploadedDocuments.set(
          document.filename,
          Number(document.chunks)
        );
      }
    }

    renderDocuments();
  } catch (error) {
    console.error("Document loading error:", error);

    showToast(
      error.message || "Could not load documents."
    );
  } finally {
    isLoadingDocuments = false;
  }
}

// --------------------------------------------------
// Delete document
// --------------------------------------------------

async function deleteDocument(filename, button) {
  const confirmed = window.confirm(
    `Delete "${filename}" and all its stored chunks? This cannot be undone.`
  );

  if (!confirmed) return;

  button.disabled = true;

  try {
    const response = await fetch(
      `${API}/documents/${encodeURIComponent(filename)}`,
      { method: "DELETE" }
    );

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(
        data.detail || "Could not delete the document."
      );
    }

    showToast(`Deleted ${filename}.`);

    await loadDocuments();
  } catch (error) {
    console.error("Document deletion error:", error);

    showToast(
      error.message || "Could not delete the document."
    );

    button.disabled = false;
  }
}

// --------------------------------------------------
// Upload document
// --------------------------------------------------

async function uploadFile(file) {
  if (isUploading) {
    showToast(
      "Please wait for the current upload to finish."
    );
    return;
  }

  const allowedExtensions = [
    ".pdf",
    ".docx",
    ".txt",
    ".md",
  ];

  const extension =
    "." + file.name.split(".").pop().toLowerCase();

  if (!allowedExtensions.includes(extension)) {
    uploadStatus.textContent = "Unsupported file type.";
    uploadStatus.className = "upload-status error";
    return;
  }

  if (file.size === 0) {
    uploadStatus.textContent = "The file is empty.";
    uploadStatus.className = "upload-status error";
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    uploadStatus.textContent =
      "Maximum file size is 10 MB.";

    uploadStatus.className = "upload-status error";
    return;
  }

  isUploading = true;
  fileInput.disabled = true;

  uploadStatus.className = "upload-status";
  uploadStatus.textContent = `Uploading ${file.name}...`;

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API}/documents`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(
        data.detail || `Upload failed (${response.status}).`
      );
    }

    uploadStatus.textContent =
      `Uploaded ${data.filename || file.name} — ` +
      `${data.chunks_added ?? 0} chunks stored.`;

    showToast("Document uploaded successfully.");

    // Refresh from the backend's authoritative document list.
    await loadDocuments();
  } catch (error) {
    uploadStatus.textContent = error.message;
    uploadStatus.className = "upload-status error";

    showToast(error.message);

    console.error("Document upload error:", error);
  } finally {
    isUploading = false;
    fileInput.disabled = false;
    fileInput.value = "";
  }
}

// --------------------------------------------------
// Send chat question
// --------------------------------------------------

async function sendQuestion(question) {
  const cleanedQuestion = question.trim();

  if (!cleanedQuestion || isSending) return;

  isSending = true;
  sendBtn.disabled = true;
  questionInput.disabled = true;

  addMessage("user", cleanedQuestion);

  questionInput.value = "";
  questionInput.style.height = "auto";

  const typingMessage = addTypingIndicator();

  try {
    const response = await fetch(`${API}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: cleanedQuestion,
      }),
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(
        data.detail || `Chat failed (${response.status}).`
      );
    }

    typingMessage.remove();
console.log("Jarvis citations:", data.citations);
console.log("Citation count:", data.citations?.length);
    // Pass both sources and citations to the message renderer.
    addMessage(
      "assistant",
      data.answer || "Jarvis returned an empty answer.",
      Array.isArray(data.sources) ? data.sources : [],
      Array.isArray(data.citations) ? data.citations : []
    );
  } catch (error) {
    typingMessage.remove();

    addMessage(
      "assistant",
      `Sorry, I couldn't get an answer. ${error.message}`
    );

    console.error("Chat error:", error);
  } finally {
    isSending = false;
    sendBtn.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
  }
}

// --------------------------------------------------
// File selection upload
// --------------------------------------------------

fileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];

  if (file) {
    uploadFile(file);
  }
});

// --------------------------------------------------
// Drag-and-drop upload
// --------------------------------------------------

uploadBox.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadBox.classList.add("dragging");
});

uploadBox.addEventListener("dragleave", () => {
  uploadBox.classList.remove("dragging");
});

uploadBox.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadBox.classList.remove("dragging");

  const file = event.dataTransfer.files?.[0];

  if (file) {
    uploadFile(file);
  }
});

// --------------------------------------------------
// Chat submission
// --------------------------------------------------

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion(questionInput.value);
});

// Enter sends; Shift+Enter adds a new line.
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

// Automatically resize the question input.
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";

  questionInput.style.height =
    `${Math.min(questionInput.scrollHeight, 150)}px`;
});

// --------------------------------------------------
// Suggested questions
// --------------------------------------------------

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    sendQuestion(button.dataset.question || "");
  });
});

// --------------------------------------------------
// Start a new chat without deleting documents
// --------------------------------------------------

document.getElementById("newChatBtn").addEventListener("click", () => {
  if (isSending) return;

  messages.replaceChildren();
  welcome.hidden = false;

  questionInput.value = "";
  questionInput.focus();
});

// --------------------------------------------------
// Mobile sidebar
// --------------------------------------------------

document.getElementById("mobileMenuBtn").addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

document.addEventListener("click", (event) => {
  if (
    window.innerWidth <= 760 &&
    sidebar.classList.contains("open") &&
    !sidebar.contains(event.target) &&
    !event.target.closest("#mobileMenuBtn")
  ) {
    sidebar.classList.remove("open");
  }
});

// --------------------------------------------------
// Load stored documents when the page opens
// --------------------------------------------------

loadDocuments();
