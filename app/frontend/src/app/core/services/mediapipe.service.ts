import { Injectable } from '@angular/core';
import { Subject, Observable } from 'rxjs';
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from '@mediapipe/tasks-vision';
import { environment } from '../../../environments/environment';
import { KeypointFrame } from '../models/sign.model';

@Injectable({ providedIn: 'root' })
export class MediaPipeService {
  private handLandmarker: HandLandmarker | null = null;
  private poseLandmarker: PoseLandmarker | null = null;
  private keypointSubject = new Subject<KeypointFrame>();
  private animFrameId: number | null = null;
  private initialized = false;
  initError: string | null = null;

  keypoints$: Observable<KeypointFrame> = this.keypointSubject.asObservable();

  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      const vision = await FilesetResolver.forVisionTasks(
        environment.mediaPipe.wasmPath
      );

      let delegate: 'GPU' | 'CPU' = 'GPU';
      try {
        this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: environment.mediaPipe.handLandmarkerModel,
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: environment.mediaPipe.numHands,
          minHandDetectionConfidence: environment.mediaPipe.minDetectionConfidence,
          minTrackingConfidence: environment.mediaPipe.minTrackingConfidence,
        });
      } catch {
        console.warn('GPU delegate failed for HandLandmarker, falling back to CPU');
        delegate = 'CPU';
        this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: environment.mediaPipe.handLandmarkerModel,
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          numHands: environment.mediaPipe.numHands,
          minHandDetectionConfidence: environment.mediaPipe.minDetectionConfidence,
          minTrackingConfidence: environment.mediaPipe.minTrackingConfidence,
        });
      }

      try {
        this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: environment.mediaPipe.poseLandmarkerModel,
            delegate,
          },
          runningMode: 'VIDEO',
          minPoseDetectionConfidence: environment.mediaPipe.minDetectionConfidence,
          minTrackingConfidence: environment.mediaPipe.minTrackingConfidence,
        });
      } catch {
        console.warn('Pose delegate failed, falling back to CPU');
        this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: environment.mediaPipe.poseLandmarkerModel,
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          minPoseDetectionConfidence: environment.mediaPipe.minDetectionConfidence,
          minTrackingConfidence: environment.mediaPipe.minTrackingConfidence,
        });
      }

      this.initialized = true;
      this.initError = null;
    } catch (err: any) {
      this.initError = err?.message || 'Failed to load MediaPipe models';
      throw err;
    }
  }

  startExtraction(videoElement: HTMLVideoElement): void {
    if (!this.initialized) return;

    const extract = () => {
      if (videoElement.readyState < 2) {
        this.animFrameId = requestAnimationFrame(extract);
        return;
      }

      const timestamp = performance.now();
      const keypoints = this.extractFrame(videoElement, timestamp);
      if (keypoints) {
        this.keypointSubject.next(keypoints);
      }

      this.animFrameId = requestAnimationFrame(extract);
    };

    this.animFrameId = requestAnimationFrame(extract);
  }

  stopExtraction(): void {
    if (this.animFrameId !== null) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
  }

  private extractFrame(
    video: HTMLVideoElement,
    timestamp: number
  ): KeypointFrame | null {
    if (!this.poseLandmarker || !this.handLandmarker) return null;

    try {
      const poseResult = this.poseLandmarker.detectForVideo(video, timestamp);
      const handResult = this.handLandmarker.detectForVideo(video, timestamp);

      const frame: number[] = [];

      if (poseResult.landmarks.length > 0) {
        for (const lm of poseResult.landmarks[0]) {
          frame.push(lm.x, lm.y, lm.z);
        }
      } else {
        frame.push(...new Array(99).fill(0));
      }

      const leftHand = handResult.handednesses.findIndex(
        h => h[0]?.categoryName === 'Left'
      );
      if (leftHand >= 0 && handResult.landmarks[leftHand]) {
        for (const lm of handResult.landmarks[leftHand]) {
          frame.push(lm.x, lm.y, lm.z);
        }
      } else {
        frame.push(...new Array(63).fill(0));
      }

      const rightHand = handResult.handednesses.findIndex(
        h => h[0]?.categoryName === 'Right'
      );
      if (rightHand >= 0 && handResult.landmarks[rightHand]) {
        for (const lm of handResult.landmarks[rightHand]) {
          frame.push(lm.x, lm.y, lm.z);
        }
      } else {
        frame.push(...new Array(63).fill(0));
      }

      return frame;
    } catch {
      return null;
    }
  }
}
