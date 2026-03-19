# Mouna App

Full-stack sign language recognition demo.

## Structure
```
app/
├── backend/      FastAPI + WebSocket inference server
│   ├── main.py
│   └── signs.json
└── frontend/     Next.js 16 + MediaPipe JS browser app
    ├── app/
    │   ├── page.tsx          → / (Live Detection)
    │   ├── dictionary/       → /dictionary (Sign Browser)
    │   └── practice/         → /practice (Practice Mode)
    └── components/
        ├── Camera.tsx        MediaPipe keypoints + WebSocket
        ├── Results.tsx       Prediction display
        ├── SignCard.tsx      Sign dictionary card
        └── Nav.tsx           Navigation bar
```

## Running

### Backend (terminal 1)
```bash
# From project root
pip install fastapi uvicorn torch
uvicorn app.backend.main:app --reload --port 8000
```

### Frontend (terminal 2)
```bash
cd app/frontend
npm run dev   # → http://localhost:3000
```

## How it works
1. Camera.tsx loads MediaPipe PoseLandmarker + HandLandmarker in the browser
2. On "Record Sign", captures 2s of 225-dim keypoint vectors (~50 frames)
3. Sends `{"keypoints": [[225 floats] × N]}` over WebSocket to backend
4. Backend runs PoseTransformer inference, returns top-3 predictions
5. UI shows prediction, confidence bar, and matched sign description
