export const environment = {
  production: false,
  /** Local: normal sign-in. The deployed build turns guest mode on. */
  guestMode: false,
  apiBaseUrl: 'http://localhost:8002',
  wsBaseUrl: 'ws://localhost:8002',
  mediaPipe: {
    wasmPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm',
    handLandmarkerModel: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
    poseLandmarkerModel: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
    numHands: 2,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  },
  detection: {
    keypointBufferSize: 30,
    sendIntervalMs: 1000,
    confidenceThreshold: 0.3,
  },
};
