import {
  Component,
  ElementRef,
  EventEmitter,
  Input,
  OnDestroy,
  OnInit,
  Output,
  ViewChild,
  inject,
} from '@angular/core';
import { Subscription } from 'rxjs';
import { CameraService } from '../../../core/services/camera.service';

@Component({
  selector: 'app-camera-view',
  standalone: true,
  template: `
    <div
      class="relative rounded-2xl overflow-hidden bg-surface border border-border/50"
      [style.aspect-ratio]="maxHeight ? null : '4/3'"
      [style.height]="maxHeight"
      [style.max-height]="maxHeight"
    >
      <video
        #videoEl
        autoplay
        playsinline
        class="w-full h-full object-cover transition-transform duration-300"
        [style.transform]="'scale(' + zoomLevel + ')' + (mirrorView ? ' scaleX(-1)' : '')"
      ></video>
      <canvas #canvasEl class="hidden"></canvas>

      <!-- Zoom controls -->
      <div class="absolute bottom-3 left-3 z-20 flex items-center gap-1.5 bg-surface/80 backdrop-blur-sm rounded-lg px-2 py-1 border border-border/50">
        <button (click)="zoomOut()" class="text-ink-muted hover:text-ink transition-colors p-0.5" [class.opacity-30]="zoomLevel <= 1">
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607zM13.5 10.5h-6" />
          </svg>
        </button>
        <span class="text-xs text-ink-muted font-mono w-8 text-center">{{ (zoomLevel * 100).toFixed(0) }}%</span>
        <button (click)="zoomIn()" class="text-ink-muted hover:text-ink transition-colors p-0.5" [class.opacity-30]="zoomLevel >= 2">
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607zM10.5 7.5v6m3-3h-6" />
          </svg>
        </button>
      </div>

      @if (error) {
        <div class="absolute inset-0 flex items-center justify-center bg-surface">
          <div class="text-center p-6">
            <div class="w-14 h-14 rounded-full bg-red-500/15 flex items-center justify-center mx-auto mb-3">
              <svg class="w-7 h-7 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
              </svg>
            </div>
            <p class="text-red-500 text-sm font-medium mb-1">Camera Error</p>
            <p class="text-ink-muted text-xs">{{ error }}</p>
          </div>
        </div>
      }

      @if (loading) {
        <div class="absolute inset-0 flex items-center justify-center bg-surface/90 backdrop-blur-sm">
          <div class="text-center">
            <div class="w-10 h-10 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
            <p class="text-sm text-ink-muted">Initializing camera...</p>
          </div>
        </div>
      }
    </div>
  `,
})
export class CameraViewComponent implements OnInit, OnDestroy {
  @Input() mirrorView = true;
  @Input() autoStart = false;
  @Input() maxHeight: string | null = null;
  @Output() frameCaptured = new EventEmitter<string>();

  @ViewChild('videoEl', { static: true }) videoRef!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvasEl', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;

  private cameraService = inject(CameraService);
  private subs: Subscription[] = [];
  private captureInterval: ReturnType<typeof setInterval> | null = null;

  loading = false;
  error: string | null = null;
  zoomLevel = 1;
  isCapturing = false;

  zoomIn(): void {
    this.zoomLevel = Math.min(2, +(this.zoomLevel + 0.15).toFixed(2));
  }

  zoomOut(): void {
    this.zoomLevel = Math.max(1, +(this.zoomLevel - 0.15).toFixed(2));
  }

  ngOnInit(): void {
    this.subs.push(
      this.cameraService.error$.subscribe(err => (this.error = err))
    );
    if (this.autoStart) {
      this.start();
    }
  }

  start(): void {
    this.loading = true;
    this.error = null;

    this.cameraService.startCamera().subscribe(stream => {
      if (stream) {
        this.videoRef.nativeElement.srcObject = stream;
        this.videoRef.nativeElement.onloadeddata = () => {
          this.loading = false;
          this.startCapturing();
        };
      } else {
        this.loading = false;
      }
    });
  }

  private startCapturing(): void {
    if (this.isCapturing) return;
    this.isCapturing = true;

    const video = this.videoRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;
    const ctx = canvas.getContext('2d')!;

    // Capture at 10 FPS (enough for sign recognition, lower bandwidth).
    // Resolution matches the requested camera stream (640x480) — hand
    // landmarks need more detail than pose, and downscaling further
    // starves MediaPipe's hand detector before pose detection notices.
    this.captureInterval = setInterval(() => {
      if (video.readyState < 2) return;

      canvas.width = 640;
      canvas.height = 480;
      ctx.drawImage(video, 0, 0, 640, 480);

      const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
      const base64 = dataUrl.split(',')[1];
      this.frameCaptured.emit(base64);
    }, 100);
  }

  /** Exposes the live MediaStream so a parent can record it (e.g. via
   * MediaRecorder) alongside the existing frame-capture pipeline -- those
   * two consumers are independent, recording doesn't change what gets
   * sent for keypoint extraction. */
  getStream(): MediaStream | null {
    return this.videoRef.nativeElement.srcObject as MediaStream | null;
  }

  stop(): void {
    this.isCapturing = false;
    if (this.captureInterval) {
      clearInterval(this.captureInterval);
      this.captureInterval = null;
    }
    this.cameraService.stopCamera();
    this.videoRef.nativeElement.srcObject = null;
  }

  ngOnDestroy(): void {
    this.stop();
    this.subs.forEach(s => s.unsubscribe());
  }
}
