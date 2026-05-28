# FeD — Face Detection Attendance System

> A full-stack face recognition attendance system built with a vanilla HTML/CSS/JS frontend, a Node.js/Express backend, custom-trained CNN models (TensorFlow.js), and a PostgreSQL database.

---

## 📁 Project Structure

```
FeD/
├── index.html          ← Main HTML entry point
├── css/
│   └── style.css       ← All styles (purple & black theme)
├── js/                 ← Frontend JavaScript Modules
│   ├── ui.js           ← UI utilities: tabs, toast, clock, helpers
│   ├── camera.js       ← Webcam control (registration & attendance)
│   ├── recognition.js  ← Backend API communication and client-side bounding boxes
│   └── app.js          ← State management, registration, attendance logic
├── server/             ← Node.js Backend & ML Pipeline
│   ├── db/             ← PostgreSQL schemas and initialization
│   ├── ml/             ← Custom CNN training and embedding logic
│   ├── server.js       ← Express API server
│   ├── .env            ← Environment variables (DB credentials)
│   └── package.json    ← Backend dependencies
└── README.md
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **Student Registration** | Name, Roll Number, Department + live face capture |
| **Face Recognition** | Backend-driven recognition via custom TensorFlow.js CNN models |
| **Robust Storage** | Powered by PostgreSQL for scalable and reliable data storage |
| **IST Timestamps** | All records stamped in Indian Standard Time (UTC+5:30) |
| **Attendance Log** | Full table with serial number, name, roll, dept, date-time, and confidence score |
| **Real-time Pipeline** | Low-latency API architecture for rapid face matching |
| **Responsive UI** | Works on desktop and mobile browsers |

---

## 🚀 Getting Started

### Prerequisites
- **Node.js** (v18+ recommended)
- **PostgreSQL** database (running locally or remotely)

### Step 1 — Setup the Backend

1. Navigate to the `server` directory:
   ```bash
   cd server
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file in the `server` directory and add your PostgreSQL credentials (see `.env` format from existing codebase if applicable):
   ```env
   DB_USER=your_user
   DB_HOST=localhost
   DB_NAME=fed_attendance
   DB_PASSWORD=your_password
   DB_PORT=5432
   ```
4. Initialize the database schema:
   ```bash
   npm run db:init
   ```
5. Start the backend server:
   ```bash
   npm start
   ```
   *(The server runs on port 3000. It will automatically load the ML models and connect to the DB).*

### Step 2 — Serve the Frontend

Open another terminal in the root directory (`FeD/`) and serve the static files:

```bash
# Using Python
python -m http.server 8080

# Using Node.js (npx)
npx serve .
```

Then open `http://localhost:8080` in your browser.

---

## 🧭 How to Use

### Step 1 — Register Students
1. Go to the **Register** tab
2. Fill in: Full Name, Roll Number, Department
3. Click **Open Camera** → position face in frame
4. Click **Capture Face** 
5. Click **Register Student** (The backend will extract facial embeddings and save them to the PostgreSQL database)

### Step 2 — Mark Attendance
1. Go to the **Attendance** tab
2. Click **Start Camera**
3. The system scans in real time by sending frames to the backend — recognized students get a **green box** and their attendance is logged automatically.
4. Unknown faces receive a **red box**.

### Step 3 — View Records
1. Go to the **Records** tab
2. View stats: total records, today's count, registered students
3. Full table with IST timestamps for every entry fetched dynamically from the database.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Structure | Semantic HTML5 |
| Styling | Pure CSS3 (custom properties, CSS Grid, Flexbox) |
| Frontend Logic | Vanilla JavaScript (ES6+ modules) |
| Backend | Node.js + Express |
| Face Detection / ML | Custom CNN embedding model via `@tensorflow/tfjs-node` |
| Database | PostgreSQL |
| Fonts | Syne (display) + JetBrains Mono (monospace) via Google Fonts |

---

## 📦 JavaScript Module Overview

### `js/ui.js` — UI Module
- `UI.getIST()` — returns current Indian Standard Time `Date` object
- `UI.formatIST(iso)` — formats an ISO string to a readable IST string
- `UI.startClock()` — starts the live clock in the header
- `UI.switchTab(id)` — switches the active panel/tab
- `UI.toast(msg, type)` — shows a floating notification (`info|success|error|warning`)
- `UI.setStatus(state, label)` — updates the system status badge

### `js/camera.js` — Camera Module
- `Camera.startRegCamera()` — opens webcam for registration
- `Camera.captureForReg()` — captures a frame and passes to App
- `Camera.startAttendanceCamera()` — opens webcam for live recognition loop

### `js/recognition.js` — Recognition Module
- `Recognition.loadModels()` — checks backend health and ML pipeline status
- `Recognition.modelsReady()` — returns `true` if backend is connected
- `Recognition.extractDescriptor(dataUrl)` — sends face image to backend for processing during registration
- `Recognition.runDetection(video, canvas, ctx)` — continually sends frames to server for real-time face matching

### `js/app.js` — App Module (State & Logic)
- `App.init()` — bootstraps the entire application
- `App.registerStudent()` — validates form, sends data to backend API
- `App.deleteStudent(roll)` — removes a student via backend API
- `App.markAttendance(student)` — delegates attendance logging to the server
- `App.updateRecords()` — fetches the latest attendance history from the database

---

## 🌐 Browser Compatibility

| Browser | Status |
|---|---|
| Chrome 90+ | ✅ Full support |
| Edge 90+ | ✅ Full support |
| Firefox 90+ | ✅ Full support |
| Safari 15+ | ⚠️ Camera may require HTTPS |

---

## 🔒 Privacy Note

All face data (embeddings) is processed and stored securely in the **PostgreSQL database**. No third-party APIs are used for facial recognition, meaning all computations are handled entirely by the designated backend server.

---

## 👨‍💻 Author

Built for **FeD — Face Detection Attendance System**  
Showcases real-time computer vision using modern JavaScript and a robust Node.js ML backend.

---

*Designed with a professional purple & black UI theme for portfolio use.*
