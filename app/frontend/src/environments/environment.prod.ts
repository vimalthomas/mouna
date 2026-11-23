export const environment = {
  production: true,
  /** Demo deployment: no sign-in or sign-up; the backend serves everyone as a
   * shared guest (needs ALLOW_GUEST=true on the backend). */
  guestMode: true,
  apiBaseUrl: '/api',
  wsBaseUrl: `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`,
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
