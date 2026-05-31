const uploadForm = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const uploadButton = document.getElementById("uploadButton");
const cameraButton = document.getElementById("cameraButton");
const pasteButton = document.getElementById("pasteButton");
const dropZone = document.getElementById("dropZone");
const cameraPanel = document.getElementById("cameraPanel");
const cameraView = document.getElementById("cameraView");
const captureButton = document.getElementById("captureButton");
const stopCameraButton = document.getElementById("stopCameraButton");
const cameraCanvas = document.getElementById("cameraCanvas");
const preview = document.getElementById("preview");
const previewFrame = document.querySelector(".preview-frame");
const fileName = document.getElementById("fileName");
const predictButton = document.getElementById("predictButton");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const message = document.getElementById("message");
const modelStatus = document.getElementById("modelStatus");

let selectedFile = null;
let cameraStream = null;
let previewUrl = null;

const classNames = ["metal", "paper", "plastic"];
const sourceButtons = {
    upload: uploadButton,
    camera: cameraButton,
    paste: pasteButton,
};

function formatPercent(value) {
    return `${Math.round(value * 100)}%`;
}

function titleCase(value) {
    return value.charAt(0).toUpperCase() + value.slice(1);
}

function setActiveSource(source) {
    Object.entries(sourceButtons).forEach(([name, button]) => {
        button.classList.toggle("active", name === source);
    });
}

function setMessage(text, isError = false) {
    message.textContent = text;
    message.classList.toggle("error", isError);
}

function resetResult() {
    predictionText.textContent = "Waiting";
    confidenceText.textContent = "0%";
    document.querySelectorAll(".prob-row").forEach((row) => {
        row.classList.remove("active");
        row.querySelector(".bar-fill").style.width = "0%";
        row.querySelector("strong").textContent = "0%";
    });
    document.querySelectorAll(".bin").forEach((bin) => {
        bin.classList.remove("active");
    });
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
    }

    cameraView.srcObject = null;
    cameraPanel.hidden = true;
    captureButton.disabled = true;
}

function setSelectedFile(file, source = "upload") {
    if (!file || !file.type.startsWith("image/")) {
        setMessage("Please choose an image file.", true);
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name || `${source}-image`;
    predictButton.disabled = false;
    setActiveSource(source);
    setMessage("");
    resetResult();

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    previewFrame.classList.add("has-image");
}

function renderResult(result) {
    predictionText.textContent = titleCase(result.class_name);
    confidenceText.textContent = formatPercent(result.confidence);

    classNames.forEach((className) => {
        const probability = result.probabilities[className] || 0;
        const row = document.querySelector(`.prob-row[data-class="${className}"]`);
        row.classList.toggle("active", className === result.class_name);
        row.querySelector(".bar-fill").style.width = formatPercent(probability);
        row.querySelector("strong").textContent = formatPercent(probability);
    });

    document.querySelectorAll(".bin").forEach((bin) => {
        bin.classList.toggle("active", bin.dataset.bin === result.class_name);
    });

    setMessage("Prediction complete.");
}

async function predictSelectedImage() {
    if (!selectedFile) {
        setMessage("Please choose an image file.", true);
        return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";
    setMessage("");

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        renderResult(data);
    } catch (error) {
        setMessage(error.message, true);
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Predict";
    }
}

async function startCamera() {
    setActiveSource("camera");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setMessage("Camera is not available in this browser.", true);
        return;
    }

    stopCamera();
    cameraPanel.hidden = false;
    captureButton.disabled = true;
    setMessage("");

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                facingMode: { ideal: "environment" },
            },
        });

        cameraView.srcObject = cameraStream;
        captureButton.disabled = false;
    } catch (_error) {
        stopCamera();
        setMessage("Camera access was blocked or unavailable.", true);
    }
}

function captureCameraImage() {
    if (!cameraStream || !cameraView.videoWidth || !cameraView.videoHeight) {
        setMessage("Camera frame is not ready.", true);
        return;
    }

    cameraCanvas.width = cameraView.videoWidth;
    cameraCanvas.height = cameraView.videoHeight;

    const context = cameraCanvas.getContext("2d");
    context.drawImage(cameraView, 0, 0, cameraCanvas.width, cameraCanvas.height);

    cameraCanvas.toBlob((blob) => {
        if (!blob) {
            setMessage("Camera capture failed.", true);
            return;
        }

        const file = new File([blob], `camera-${Date.now()}.jpg`, {
            type: "image/jpeg",
        });

        setSelectedFile(file, "camera");
        stopCamera();
    }, "image/jpeg", 0.92);
}

function imageFileFromClipboardData(clipboardData) {
    if (!clipboardData) {
        return null;
    }

    const items = Array.from(clipboardData.items || []);
    const imageItem = items.find((item) => item.type.startsWith("image/"));

    if (!imageItem) {
        return null;
    }

    const file = imageItem.getAsFile();

    if (!file) {
        return null;
    }

    if (file.name) {
        return file;
    }

    const extension = file.type.split("/")[1] || "png";
    return new File([file], `pasted-image.${extension}`, { type: file.type });
}

async function pasteImageFromClipboard() {
    setActiveSource("paste");
    stopCamera();

    if (!navigator.clipboard || !navigator.clipboard.read) {
        setMessage("Clipboard image reading is not available in this browser.", true);
        return;
    }

    try {
        const clipboardItems = await navigator.clipboard.read();

        for (const item of clipboardItems) {
            const imageType = item.types.find((type) => type.startsWith("image/"));

            if (imageType) {
                const blob = await item.getType(imageType);
                const extension = imageType.split("/")[1] || "png";
                const file = new File([blob], `pasted-image.${extension}`, {
                    type: imageType,
                });

                setSelectedFile(file, "paste");
                return;
            }
        }

        setMessage("No clipboard image detected.", true);
    } catch (_error) {
        setMessage("Clipboard image could not be read.", true);
    }
}

async function checkModel() {
    try {
        const response = await fetch("/health");
        const data = await response.json();

        if (response.ok && data.model_exists) {
            modelStatus.textContent = "Model ready";
            modelStatus.className = "status-pill ready";
            return;
        }

        modelStatus.textContent = "Model missing";
        modelStatus.className = "status-pill error";
    } catch (_error) {
        modelStatus.textContent = "Server offline";
        modelStatus.className = "status-pill error";
    }
}

captureButton.disabled = true;

uploadButton.addEventListener("click", () => {
    stopCamera();
    setActiveSource("upload");
});

cameraButton.addEventListener("click", () => {
    startCamera();
});

pasteButton.addEventListener("click", () => {
    pasteImageFromClipboard();
});

stopCameraButton.addEventListener("click", () => {
    stopCamera();
    setActiveSource("upload");
});

captureButton.addEventListener("click", () => {
    captureCameraImage();
});

imageInput.addEventListener("change", (event) => {
    stopCamera();
    setSelectedFile(event.target.files[0], "upload");
});

uploadForm.addEventListener("submit", (event) => {
    event.preventDefault();
    predictSelectedImage();
});

dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
    stopCamera();
    setSelectedFile(event.dataTransfer.files[0], "upload");
});

dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        imageInput.click();
    }
});

document.addEventListener("paste", (event) => {
    const file = imageFileFromClipboardData(event.clipboardData);

    if (!file) {
        return;
    }

    event.preventDefault();
    stopCamera();
    setSelectedFile(file, "paste");
});

window.addEventListener("beforeunload", () => {
    stopCamera();

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }
});

checkModel();
