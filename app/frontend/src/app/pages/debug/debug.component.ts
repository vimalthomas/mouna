import { Component, inject, OnDestroy, ViewChild } from '@angular/core';
import { Subscription } from 'rxjs';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { CameraViewComponent } from '../../shared/components/camera-view/camera-view.component';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-debug',
  standalone: true,
  imports: [CameraViewComponent],
  template: `
    <div class="max-w-7xl mx-auto">
      <div class="mb-6">
        <h2 class="text-2xl font-bold text-white">Diagnostics</h2>
        <p class="text-sm text-slate-400 mt-1">Real-time pipeline — browser sends frames, backend runs MediaPipe + TGCN model</p>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <app-camera-view
            #cameraView
            [mirrorView]="true"
            (frameCaptured)="onFrame($event)"
          />
          <div class="flex gap-3 mt-4">
            @if (!isRecording) {
              <button (click)="startRecording()" class="flex-1 bg-indigo-500 hover:bg-indigo-400 text-white py-3 rounded-xl font-medium">
                Start Diagnostics
              </button>
            } @else {
              <button (click)="stopRecording()" class="flex-1 bg-red-500 hover:bg-red-400 text-white py-3 rounded-xl font-medium">
                Stop
              </button>
            }
          </div>

          @if (isRecording) {
            <div class="mt-3 bg-slate-800/80 rounded-xl p-4 border border-slate-700/50 font-mono text-xs space-y-1">
              <div class="flex justify-between">
                <span class="text-slate-400">Frames sent:</span>
                <span class="text-white">{{ frameCount }}</span>
              </div>
            </div>
          }
        </div>

        <div class="space-y-4" style="max-height: 700px; overflow-y: auto;">
          @if (diagnosis.length > 0) {
            <div class="bg-slate-800/80 rounded-xl p-4 border border-slate-700/50">
              <h3 class="text-sm font-semibold text-white mb-3">Diagnosis</h3>
              @for (d of diagnosis; track d) {
                <div class="flex items-start gap-2 mb-2 text-xs">
                  @if (d.startsWith('CRITICAL')) {
                    <span class="text-red-400 font-bold shrink-0">&#x26A0;</span>
                    <span class="text-red-300">{{ d }}</span>
                  } @else if (d.startsWith('WARNING')) {
                    <span class="text-amber-400 font-bold shrink-0">&#x26A0;</span>
                    <span class="text-amber-300">{{ d }}</span>
                  } @else {
                    <span class="text-emerald-400 font-bold shrink-0">&#x2714;</span>
                    <span class="text-emerald-300">{{ d }}</span>
                  }
                </div>
              }
            </div>
          }

          @if (lastAnalysis) {
            <div class="bg-slate-800/80 rounded-xl p-4 border border-slate-700/50">
              <h3 class="text-sm font-semibold text-white mb-3">MediaPipe Detection (Backend)</h3>
              <div class="font-mono text-xs space-y-2">
                <div class="flex justify-between">
                  <span class="text-slate-400">Pose:</span>
                  <span [class]="lastAnalysis.mediapipe_detection.pose_detected ? 'text-emerald-400' : 'text-red-400'">
                    {{ lastAnalysis.mediapipe_detection.pose_detected ? 'Detected' : 'NOT DETECTED' }}
                  </span>
                </div>
                <div class="flex justify-between">
                  <span class="text-slate-400">Left Hand:</span>
                  <span [class]="lastAnalysis.mediapipe_detection.left_hand_detected ? 'text-emerald-400' : 'text-red-400'">
                    {{ lastAnalysis.mediapipe_detection.left_hand_detected ? 'Detected' : 'NOT DETECTED' }}
                  </span>
                </div>
                <div class="flex justify-between">
                  <span class="text-slate-400">Right Hand:</span>
                  <span [class]="lastAnalysis.mediapipe_detection.right_hand_detected ? 'text-emerald-400' : 'text-red-400'">
                    {{ lastAnalysis.mediapipe_detection.right_hand_detected ? 'Detected' : 'NOT DETECTED' }}
                  </span>
                </div>
              </div>
            </div>

            <div class="bg-slate-800/80 rounded-xl p-4 border border-slate-700/50">
              <h3 class="text-sm font-semibold text-white mb-3">TGCN Model Input</h3>
              <div class="font-mono text-xs space-y-2">
                <div class="flex justify-between">
                  <span class="text-slate-400">Input Quality:</span>
                  <span [class]="lastAnalysis.tgcn_conversion.input_quality === 'good' ? 'text-emerald-400 font-bold' : 'text-red-400 font-bold'">
                    {{ lastAnalysis.tgcn_conversion.input_quality }}
                  </span>
                </div>
              </div>
            </div>

            <div class="bg-slate-800/80 rounded-xl p-4 border border-slate-700/50">
              <h3 class="text-sm font-semibold text-white mb-3">Model Prediction</h3>
              <div class="text-3xl font-bold text-indigo-400 capitalize mb-3">
                {{ lastPrediction?.sign || '—' }}
              </div>
              <div class="font-mono text-xs space-y-1">
                @for (p of lastPrediction?.top3 || []; track p.sign) {
                  <div class="flex justify-between">
                    <span class="text-slate-400 capitalize">{{ p.sign }}:</span>
                    <span class="text-white">{{ (p.confidence * 100).toFixed(1) }}%</span>
                  </div>
                }
              </div>
            </div>
          }

          @if (!lastAnalysis && !isRecording) {
            <div class="bg-slate-800/80 rounded-xl p-8 border border-slate-700/50 text-center">
              <p class="text-slate-400">Click "Start Diagnostics" to test the pipeline</p>
              <p class="text-xs text-slate-500 mt-2">Browser sends video frames → Backend MediaPipe extracts keypoints → TGCN model predicts</p>
            </div>
          }
        </div>
      </div>
    </div>
  `,
})
export class DebugComponent implements OnDestroy {
  @ViewChild('cameraView') cameraView!: CameraViewComponent;

  isRecording = false;
  frameCount = 0;
  lastAnalysis: any = null;
  lastPrediction: any = null;
  diagnosis: string[] = [];

  private ws: WebSocketSubject<any> | null = null;
  private subs: Subscription[] = [];
  private frameBuffer: string[] = [];

  startRecording(): void {
    this.cameraView.start();
    this.isRecording = true;
    this.frameCount = 0;
    this.frameBuffer = [];

    this.ws = webSocket(`${environment.wsBaseUrl}/ws/debug`);
    this.subs.push(
      this.ws.subscribe(msg => {
        if (msg.type === 'debug') {
          this.lastAnalysis = msg.analysis;
          this.lastPrediction = msg.prediction;
          this.diagnosis = msg.diagnosis;
        }
      })
    );
  }

  onFrame(base64Frame: string): void {
    if (!this.isRecording) return;
    this.frameCount++;
    this.frameBuffer.push(base64Frame);

    // Every 30 frames, extract keypoints on backend and analyze
    if (this.frameBuffer.length >= 30) {
      this.ws?.next({ type: 'frames', frames: this.frameBuffer });
      this.frameBuffer = [];
    }
  }

  stopRecording(): void {
    this.isRecording = false;
    this.cameraView.stop();
    this.ws?.complete();
    this.ws = null;
    this.frameBuffer = [];
  }

  ngOnDestroy(): void {
    this.stopRecording();
    this.subs.forEach(s => s.unsubscribe());
  }
}
