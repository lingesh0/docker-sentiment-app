# Sentiment Analysis Microservice

## 🧠 Overview
A full-stack sentiment analysis system using a FastAPI backend and a professional React (TypeScript) frontend. All components are Dockerized and orchestrated with Docker Compose.

---

## ✨ Features
- Responsive UI with dark mode, header, and footer
- Loading spinner, animated results, sentiment icons, and sample/clear buttons
- Accessible and mobile-friendly
- FastAPI backend with `/ping` (health) and `/predict` (sentiment) endpoints
- Dockerized for easy deployment

---

## 🚀 Step-by-Step Setup & Usage

### 1. Prerequisites
- **Docker Desktop** (with WSL2 enabled on Windows): [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)
- (Optional) **Python 3.10+** if you want to run or fine-tune outside Docker

### 2. Clone the Repository
```bash
# Replace with your repo URL if needed
git clone <repo-url>
cd assignment
```

### 3. Build & Run with Docker Compose
```bash
docker compose up --build
```
- This will build and start both the backend (port 8000) and frontend (port 3000).

### 4. Access the App
- **Frontend:** [http://localhost:3000](http://localhost:3000)
- **Backend health:** [http://localhost:8000/ping](http://localhost:8000/ping) (should return `{ "message": "pong" }`)
- **Backend API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Using the App
- Enter text in the textarea and click **Predict**
- Try the **Sample** button for a quick demo
- Use **Clear** to reset
- Toggle **Dark/Light mode** in the header
- View animated results and sentiment icons

---

## 🛠️ Troubleshooting

### See Real-Time Logs
```bash
docker compose logs -f
```

### Check Running Containers
```bash
docker ps
```

### Check Backend Health
- Visit [http://localhost:8000/ping](http://localhost:8000/ping) in your browser. You should see `{ "message": "pong" }`.

### Check Frontend
- Visit [http://localhost:3000](http://localhost:3000) in your browser. The app should load and connect to the backend.

### Common Issues
- **Docker not running:** Start Docker Desktop and try again.
- **Port conflicts:** Make sure nothing else is using ports 8000 or 3000.
- **Backend not responding:** Check logs with `docker compose logs backend`.
- **Frontend not connecting:** Ensure `REACT_APP_BACKEND_URL` is set to `http://backend:8000/predict` in Docker Compose.

---

## 📝 Design Decisions
- Loads fine-tuned model from `./model` if present, else uses Hugging Face default (if model code is added)
- All services run in Docker for easy deployment

---

## 🐳 Docker Images
- **Backend:** Python 3.10-slim, FastAPI
- **Frontend:** Node 18-alpine, built with React, served with `serve`

---

## 📹 Demo
- (Optional) Add a screen recording link here. 