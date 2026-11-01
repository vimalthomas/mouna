import { Component, inject, OnDestroy, OnInit, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { CameraViewComponent } from '../../shared/components/camera-view/camera-view.component';
import { SkeletonCompareComponent } from '../../shared/components/skeleton-compare/skeleton-compare.component';
import { GeminiCoachService } from '../../core/services/gemini-coach.service';
import { SignsApiService } from '../../core/services/signs-api.service';
import { BBox, CoachingMessage, PrepStep, ReferenceBuckets, Sign, SkeletonFrame, StepResult } from '../../core/models/sign.model';
import { environment } from '../../../environments/environment';

type Screen = 'preview' | 'ready' | 'record';
type StepStatus = 'pending' | 'checking' | 'match' | 'mismatch' | 'na';

interface ChatMessage {
  role: 'user' | 'coach' | 'system';
  text: string;
  coachingData?: CoachingMessage;
}

interface StepState {
  step: PrepStep;
  label: string;
  status: StepStatus;
  result?: StepResult;
}

const STEP_LABELS: Record<PrepStep, string> = {
  handshape: 'Handshape',
  location: 'Location',
  orientation: 'Orientation',
  hand_count: 'One vs. two hands',
  facial: 'Facial expression',
};

const COUNTDOWN_SECONDS = 3;
const RECORDING_DURATION_MS = 5000;
const CHECK_HOLD_MS = 1200;

@Component({
  selector: 'app-coach',
  standalone: true,
  imports: [CameraViewComponent, SkeletonCompareComponent, FormsModule],
  template: `
    <div class="max-w-7xl mx-auto" [style.paddingRight]="chatOpen ? '380px' : '0'">

      <!-- Step indicator -->
      <div class="mb-5 grid grid-cols-3 gap-0 rounded-2xl border border-border/50 overflow-hidden bg-surface">
        @for (s of screenSteps; track s.id; let i = $index) {
          <button
            (click)="jumpToScreen(s.id)"
            class="px-4 py-2.5 text-left transition-colors"
            [class]="screen === s.id ? 'bg-accent/15' : 'hover:bg-surface/60'"
            [class.border-r]="i < screenSteps.length - 1"
            style="border-color: var(--border, #2b3140);"
          >
            <div class="flex items-center gap-2">
              <span
                class="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold font-mono shrink-0"
                [class]="screen === s.id ? 'bg-accent text-white' : 'bg-border/60 text-ink-muted'"
              >{{ i + 1 }}</span>
              <span class="text-xs font-semibold text-ink">{{ s.label }}</span>
            </div>
          </button>
        }
      </div>

      <!-- ============== SCREEN 1: PREVIEW ============== -->
      @if (screen === 'preview') {
        <div class="rounded-2xl bg-surface border border-border/50 p-5">
          <h2 class="text-lg font-bold text-ink mb-1">Watch before you try</h2>
          <p class="text-sm text-ink-muted mb-4">Understand what you're about to attempt — no camera yet.</p>

          <div class="flex gap-3 mb-4">
            <select
              [(ngModel)]="targetSignWord"
              (ngModelChange)="onTargetChange()"
              class="flex-1 h-11 bg-surface-sunk border border-border/50 rounded-xl px-4 text-ink text-sm capitalize focus:outline-none focus:border-accent/50"
            >
              @for (sign of signs; track sign.word) {
                <option [value]="sign.word">{{ sign.word }}</option>
              }
            </select>
          </div>

          <div class="flex items-center gap-2 mb-3">
            <span class="text-xs font-semibold text-ink-faint uppercase tracking-wide mr-1">Speed:</span>
            @for (r of playbackRates; track r.value) {
              <button
                (click)="setPlaybackRate(r.value)"
                class="text-xs font-semibold px-3 py-1 rounded-full border"
                [class]="playbackRate === r.value ? 'bg-accent text-white border-accent' : 'bg-surface-sunk text-ink-muted border-border/50'"
              >{{ r.label }}</button>
            }
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-3" #previewVideos>
            <div class="relative rounded-xl overflow-hidden bg-black border border-border/50" style="aspect-ratio: 16/10;">
              @if (referenceVideoUrl) {
                <video [src]="referenceVideoUrl" autoplay loop muted playsinline class="w-full h-full object-cover" (loadedmetadata)="applyRate($event)"></video>
              } @else {
                <div class="w-full h-full flex items-center justify-center text-ink-muted text-sm">No reference video for "{{ targetSignWord }}"</div>
              }
              <div class="absolute top-2 left-2 text-[10px] font-semibold uppercase tracking-wide text-white/80 bg-black/50 px-2 py-0.5 rounded">Full clip</div>
            </div>

            <!-- real hand, cropped from the SAME clip and kept in sync -- not a drawn icon -->
            <div class="relative rounded-xl overflow-hidden bg-black border border-border/50" style="aspect-ratio: 16/10;">
              @if (referenceVideoUrl && handZoomStyle) {
                <video [src]="referenceVideoUrl" autoplay loop muted playsinline class="absolute object-cover" [style]="handZoomStyle" (loadedmetadata)="applyRate($event)"></video>
              } @else if (referenceVideoUrl) {
                <video [src]="referenceVideoUrl" autoplay loop muted playsinline class="w-full h-full object-cover" (loadedmetadata)="applyRate($event)"></video>
              } @else {
                <div class="w-full h-full flex items-center justify-center text-ink-muted text-sm">No reference video</div>
              }
              <div class="absolute top-2 left-2 text-[10px] font-semibold uppercase tracking-wide text-white/80 bg-black/50 px-2 py-0.5 rounded">Hand, zoomed in</div>
            </div>
          </div>

          <div class="flex flex-wrap gap-2 mt-3">
            @if (referenceBuckets) {
              <span class="text-xs font-medium px-3 py-1 rounded-full bg-surface-sunk border border-border/50 text-ink-muted">{{ handshapeLabel }}</span>
              <span class="text-xs font-medium px-3 py-1 rounded-full bg-surface-sunk border border-border/50 text-ink-muted capitalize">signed near: {{ referenceBuckets.location.replace('_',' ') }}</span>
            }
            @if (targetSignInfo?.nms_required) {
              <span class="text-xs font-medium px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-600 dark:text-amber-400">{{ nmsShortLabel }}</span>
            }
          </div>

          @if (targetSignInfo?.nms_required) {
            <div class="mt-4 rounded-xl bg-amber-500/10 border border-amber-500/30 px-4 py-3">
              <p class="text-sm font-semibold text-amber-600 dark:text-amber-400">This sign needs a facial expression: {{ nmsShortLabel }}</p>
              <p class="text-xs text-ink-muted mt-1">{{ targetSignInfo?.nms_notes }}</p>
            </div>
          }

          @if (requiredSteps.length > 0) {
            <div class="flex flex-wrap gap-2 mt-4">
              @for (step of requiredSteps; track step) {
                <span class="text-xs font-medium px-3 py-1 rounded-full bg-surface-sunk border border-border/50 text-ink-muted">
                  {{ stepLabel(step) }}
                </span>
              }
            </div>
          }

          <div class="mt-5">
            <button (click)="goToReady()" class="px-6 h-11 bg-accent hover:bg-accent/90 text-white rounded-xl font-medium text-sm">
              I'm ready to practice →
            </button>
          </div>
        </div>
      }

      <!-- ============== SCREEN 2: GET READY ============== -->
      @if (screen === 'ready') {
        <div class="rounded-2xl bg-surface border border-border/50 p-5">
          <h2 class="text-lg font-bold text-ink mb-1">Get ready</h2>
          <p class="text-sm text-ink-muted mb-4">Analyze whenever you want — it's a hint, not a gate. Record any time.</p>

          <div class="relative rounded-xl overflow-hidden" style="max-height: 50vh;">
            <app-camera-view #cameraViewReady [mirrorView]="true" maxHeight="50vh" (frameCaptured)="onFrame($event)" />

            @if (faceBboxStyle) {
              <div class="absolute rounded-lg pointer-events-none border-2 border-indigo-400" [style]="faceBboxStyle">
                <span class="absolute -top-5 left-0 text-[10px] font-semibold text-white bg-black/60 px-1.5 py-0.5 rounded">Face</span>
              </div>
            }
            @if (handBboxStyle) {
              <div class="absolute rounded-lg pointer-events-none border-2 border-blue-500" [style]="handBboxStyle">
                <span class="absolute -top-5 left-0 text-[10px] font-semibold text-white bg-black/60 px-1.5 py-0.5 rounded">Hands</span>
              </div>
            }
          </div>

          <!-- live, auto-detected, no click required -->
          <div class="flex items-center gap-2 mt-3 flex-wrap">
            <span class="text-xs font-semibold text-ink-faint uppercase tracking-wide mr-1">Live:</span>
            <span class="text-xs font-medium px-2.5 py-1 rounded-full" [class]="liveStatus.pose ? 'bg-emerald-500/15 text-emerald-500' : 'bg-surface-sunk text-ink-faint border border-border/50'">
              {{ liveStatus.pose ? '✓' : '…' }} position
            </span>
            <span class="text-xs font-medium px-2.5 py-1 rounded-full" [class]="liveStatus.leftHand ? 'bg-emerald-500/15 text-emerald-500' : 'bg-surface-sunk text-ink-faint border border-border/50'">
              {{ liveStatus.leftHand ? '✓' : '…' }} left hand
            </span>
            <span class="text-xs font-medium px-2.5 py-1 rounded-full" [class]="liveStatus.rightHand ? 'bg-emerald-500/15 text-emerald-500' : 'bg-surface-sunk text-ink-faint border border-border/50'">
              {{ liveStatus.rightHand ? '✓' : '…' }} right hand
            </span>
            @if (requiredSteps.includes('facial')) {
              <span class="text-xs font-medium px-2.5 py-1 rounded-full" [class]="liveStatus.face ? 'bg-emerald-500/15 text-emerald-500' : 'bg-surface-sunk text-ink-faint border border-border/50'">
                {{ liveStatus.face ? '✓' : '…' }} face
              </span>
            }
          </div>
          <p class="text-xs text-ink-faint mt-1">These update automatically — no need to click anything for them.</p>

          @if (requiredSteps.includes('facial')) {
            <div class="mt-3 rounded-xl bg-amber-500/10 border border-amber-500/30 px-4 py-2.5">
              <p class="text-sm font-semibold text-amber-600 dark:text-amber-400">This sign needs a facial expression: {{ nmsShortLabel }}</p>
            </div>
          }

          <!-- one action, not a gated checklist -->
          <div class="mt-4">
            <button
              (click)="analyze()"
              [disabled]="analyzing"
              class="w-full h-12 rounded-xl font-medium text-sm text-white bg-accent hover:bg-accent/90 disabled:opacity-60 flex items-center justify-center gap-2"
            >
              @if (analyzing) {
                <span class="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin"></span>
                Analyzing…
              } @else {
                Analyze
              }
            </button>
          </div>

          @if (comments.length > 0) {
            <div class="flex flex-col gap-2 mt-4">
              @for (c of comments; track c.step) {
                <div
                  class="flex items-start gap-3 rounded-xl px-4 py-3 border"
                  [class]="c.tone === 'good' ? 'bg-emerald-500/10 border-emerald-500/30' : c.tone === 'warn' ? 'bg-amber-500/10 border-amber-500/30' : 'bg-surface-sunk border-border/50'"
                >
                  <span class="text-sm mt-0.5" [class]="c.tone === 'good' ? 'text-emerald-500' : c.tone === 'warn' ? 'text-amber-500' : 'text-ink-faint'">
                    {{ c.tone === 'good' ? '✓' : c.tone === 'warn' ? '!' : '·' }}
                  </span>
                  <div>
                    <p class="text-xs font-semibold uppercase tracking-wide text-ink-faint">{{ c.label }}</p>
                    <p class="text-sm text-ink mt-0.5">{{ c.text }}</p>
                  </div>
                </div>
              }
            </div>
            @if (facialCheckAttempted) {
              <p class="text-xs text-ink-faint mt-2">Facial comment is an early, unvalidated first pass — treat it as a hint, not a guarantee.</p>
            }
          }

          <div class="mt-5">
            <button (click)="goToRecord()" class="w-full h-11 rounded-xl font-medium text-sm text-white bg-red-500 hover:bg-red-400">
              Record →
            </button>
          </div>
        </div>
      }

      <!-- ============== SCREEN 3: RECORD -> FEEDBACK ============== -->
      @if (screen === 'record') {
        <div class="rounded-2xl bg-surface border border-border/50 p-5">
          @if (recordState !== 'done') {
            <h2 class="text-lg font-bold text-ink mb-1">Recording</h2>
            <p class="text-sm text-ink-muted mb-4">Hints are hidden now — this is the graded attempt.</p>

            <div class="relative rounded-xl overflow-hidden" style="max-height: 55vh;">
              <app-camera-view #cameraViewRecord [mirrorView]="true" maxHeight="55vh" (frameCaptured)="onFrame($event)" />

              @if (recordState === 'countdown') {
                <div class="absolute inset-0 flex items-center justify-center bg-black/50">
                  <span class="text-5xl font-bold text-amber-300 font-mono">{{ countdownValue }}</span>
                </div>
              }
              @if (recordState === 'recording') {
                <div class="absolute inset-0 border-4 border-red-500/60 rounded-xl pointer-events-none"></div>
                <div class="absolute bottom-3 left-4 right-4 flex items-center gap-2">
                  <span class="w-2.5 h-2.5 rounded-full bg-red-400 animate-pulse"></span>
                  <div class="flex-1 h-1.5 bg-black/40 rounded-full overflow-hidden">
                    <div class="h-full bg-red-400 rounded-full transition-all duration-100" [style.width.%]="recordingProgress"></div>
                  </div>
                </div>
              }
            </div>

            <div class="flex gap-3 mt-4">
              @if (recordState === 'ready') {
                <button (click)="onRecordClick()" class="px-6 h-11 bg-red-500 hover:bg-red-400 text-white rounded-xl font-medium text-sm">
                  ● Record
                </button>
              }
              <button (click)="jumpToScreen('ready')" class="px-4 h-11 bg-surface-sunk border border-border/50 text-ink rounded-xl text-sm">
                ← Back to prep
              </button>
            </div>
          } @else {
            <!-- feedback, in place -->
            <h2 class="text-lg font-bold text-ink mb-3">Your attempt</h2>

            <div class="flex items-center justify-between rounded-xl bg-surface-sunk border border-border/50 px-4 py-3 mb-2">
              <div>
                <div class="text-xs uppercase tracking-wide text-ink-faint font-semibold">open-set prediction</div>
                <div class="text-sm font-semibold text-ink mt-0.5">"{{ lastResult?.prediction }}" · {{ ((lastResult?.confidence||0)*100).toFixed(0) }}%</div>
              </div>
            </div>
            <div class="flex items-center justify-between rounded-xl bg-surface-sunk border border-border/50 px-4 py-3 mb-4">
              <div>
                <div class="text-xs uppercase tracking-wide text-ink-faint font-semibold">target-match verdict</div>
                <div class="text-sm font-semibold text-ink mt-0.5">vs. "{{ targetSignWord }}" reference</div>
              </div>
              <span
                class="text-xs font-bold uppercase px-3 py-1 rounded-full"
                [class]="lastResult?.verdict === 'good' ? 'bg-emerald-500/15 text-emerald-500' : 'bg-amber-500/15 text-amber-500'"
              >{{ (lastResult?.verdict || '').replace('_',' ') }}</span>
            </div>

            <div class="flex items-center gap-2 mb-3">
              <span class="text-xs font-semibold text-ink-faint uppercase tracking-wide mr-1">Playback:</span>
              @for (r of skeletonRates; track r.value) {
                <button
                  (click)="skeletonSpeed = r.value"
                  class="text-xs font-semibold px-3 py-1 rounded-full border"
                  [class]="skeletonSpeed === r.value ? 'bg-accent text-white border-accent' : 'bg-surface-sunk text-ink-muted border-border/50'"
                >{{ r.label }}</button>
              }
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
              <app-skeleton-compare
                [frames]="lastResult?.attempt_skeleton_raw ?? null"
                [parameters]="lastResult?.parameters ?? null"
                [speedMultiplier]="skeletonSpeed"
                color="#3f5fc4"
                label="Your motion"
              />
              <app-skeleton-compare
                [frames]="lastResult?.reference_skeleton ?? null"
                [speedMultiplier]="skeletonSpeed"
                color="#8a93a3"
                label="Target"
              />
            </div>
            <div class="grid grid-cols-1 gap-4 mb-4">
              <app-skeleton-compare
                mode="hands"
                [frames]="lastResult?.attempt_skeleton_raw ?? null"
                [parameters]="lastResult?.parameters ?? null"
                [speedMultiplier]="skeletonSpeed"
                color="#3f5fc4"
                label="Your hand, zoomed in"
              />
              <app-skeleton-compare
                mode="hands"
                [frames]="lastResult?.reference_skeleton ?? null"
                [speedMultiplier]="skeletonSpeed"
                color="#8a93a3"
                label="Target hand, zoomed in"
              />
            </div>

            @if (snapshotEntries.length > 0) {
              <div class="mb-4">
                <p class="text-xs font-semibold text-ink-faint uppercase tracking-wide mb-2">Where the difference was</p>
                <div class="grid grid-cols-2 gap-3">
                  @for (s of snapshotEntries; track s.param) {
                    <div>
                      <app-skeleton-compare [frames]="[s.frame]" [animate]="false" color="#ab7a14" [label]="s.param" />
                      <p class="text-xs text-ink-muted mt-1">Your {{ s.param }} diverged most from the target here.</p>
                    </div>
                  }
                </div>
              </div>
            }

            <div class="flex flex-col gap-2">
              @for (p of parameterCards; track p.name) {
                <div class="rounded-xl border px-4 py-3" [class]="p.match === true ? 'border-emerald-500/30' : p.match === false ? 'border-amber-500/30' : 'border-border/50 border-dashed opacity-70'">
                  <div class="flex items-center justify-between">
                    <span class="text-sm font-semibold text-ink">{{ p.name }}</span>
                    <span class="text-[10px] font-bold uppercase px-2 py-0.5 rounded"
                      [class]="p.match === true ? 'bg-emerald-500/15 text-emerald-500' : p.match === false ? 'bg-amber-500/15 text-amber-500' : 'bg-surface text-ink-faint border border-dashed border-border'"
                    >{{ p.match === true ? 'match' : p.match === false ? 'mismatch' : 'n/a' }}</span>
                  </div>
                  @if (p.detail) { <p class="text-xs font-mono text-ink-muted mt-1">{{ p.detail }}</p> }
                </div>
              }
            </div>

            <div class="flex gap-3 mt-5">
              <button (click)="retryRecording()" class="px-6 h-11 bg-accent hover:bg-accent/90 text-white rounded-xl font-medium text-sm">↺ Try again</button>
              <button (click)="jumpToScreen('preview')" class="px-4 h-11 bg-surface-sunk border border-border/50 text-ink rounded-xl text-sm">Choose another sign</button>
            </div>
          }
        </div>
      }
    </div>

    <!-- Fixed chat panel -->
    <div class="fixed top-0 right-0 h-full z-40 flex">
      <div class="bg-surface border-l border-border/50 flex flex-col" style="width: 360px; height: 100vh;">
        <div class="px-5 py-4 border-b border-border/50 flex items-center justify-between flex-shrink-0">
          <h3 class="text-sm font-semibold text-ink">AI Coach Chat</h3>
          @if (screen !== 'preview') {
            <div class="flex items-center gap-1.5"><div class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div><span class="text-xs text-emerald-500">Live</span></div>
          }
        </div>
        <div #chatScroll class="flex-1 overflow-y-auto p-4 space-y-3">
          @for (msg of chatMessages; track $index) {
            @if (msg.role === 'user') {
              <div class="flex justify-end"><div class="max-w-[85%] rounded-2xl rounded-br-sm px-4 py-2.5 bg-accent text-white"><p class="text-sm">{{ msg.text }}</p></div></div>
            }
            @if (msg.role === 'system') {
              <div class="flex justify-center"><div class="rounded-full px-4 py-1.5 bg-border/40 border border-border/60"><p class="text-xs text-ink-muted">{{ msg.text }}</p></div></div>
            }
            @if (msg.role === 'coach') {
              <div class="flex justify-start"><div class="max-w-[90%] rounded-2xl rounded-bl-sm px-4 py-3 bg-accent/10"><p class="text-sm text-ink leading-relaxed">{{ msg.text }}</p></div></div>
            }
          }
        </div>
        <div class="p-3 border-t border-border/50">
          <div class="flex gap-2">
            <input type="text" [(ngModel)]="chatInput" (keydown.enter)="sendChat()" [disabled]="screen==='preview'" placeholder="Ask the AI coach..."
              class="flex-1 bg-background border border-border/50 rounded-xl px-4 py-2.5 text-sm text-ink placeholder-ink-muted focus:outline-none disabled:opacity-40" />
            <button (click)="sendChat()" [disabled]="screen==='preview' || !chatInput.trim()" class="bg-accent text-white px-4 rounded-xl disabled:bg-border disabled:text-ink-muted">→</button>
          </div>
        </div>
      </div>
    </div>
  `,
})
export class CoachComponent implements OnInit, OnDestroy, AfterViewChecked {
  @ViewChild('previewVideos') previewVideosEl?: ElementRef<HTMLDivElement>;
  @ViewChild('cameraViewReady') cameraViewReady?: CameraViewComponent;
  @ViewChild('cameraViewRecord') cameraViewRecord?: CameraViewComponent;
  @ViewChild('chatScroll') chatScrollEl?: ElementRef<HTMLDivElement>;

  private coachService = inject(GeminiCoachService);
  private signsApi = inject(SignsApiService);
  private subs: Subscription[] = [];
  private shouldScroll = false;

  readonly screenSteps: { id: Screen; label: string }[] = [
    { id: 'preview', label: 'Preview' },
    { id: 'ready', label: 'Get ready' },
    { id: 'record', label: 'Record & feedback' },
  ];

  screen: Screen = 'preview';
  signs: Sign[] = [];
  targetSignWord = 'apple';
  requiredSteps: PrepStep[] = [];
  comments: { step: PrepStep; label: string; text: string; tone: 'good' | 'warn' | 'neutral' }[] = [];
  analyzing = false;
  private hasAnalyzedFacial = false;
  referenceBuckets: ReferenceBuckets | null = null;
  liveStatus = { pose: false, leftHand: false, rightHand: false, face: false };

  chatMessages: ChatMessage[] = [];
  chatInput = '';
  attemptCount = 0;
  correctCount = 0;
  sessionAccuracy = 0;
  chatOpen = true;

  recordState: 'ready' | 'countdown' | 'recording' | 'done' = 'ready';
  countdownValue = COUNTDOWN_SECONDS;
  recordingProgress = 0;
  private countdownTimer: ReturnType<typeof setInterval> | null = null;
  private recordingTimer: ReturnType<typeof setInterval> | null = null;
  private recordingStartTime = 0;

  referenceVideoUrl: string | null = null;
  readonly playbackRates = [
    { label: '0.5×', value: 0.5 },
    { label: '1×', value: 1 },
    { label: '1.5×', value: 1.5 },
  ];
  playbackRate = 1;
  readonly skeletonRates = [
    { label: '0.5×', value: 0.5 },
    { label: '1×', value: 1 },
    { label: '1.5×', value: 1.5 },
  ];
  skeletonSpeed = 1;
  faceBboxStyle: Record<string, string> | null = null;
  handBboxStyle: Record<string, string> | null = null;
  lastResult: CoachingMessage | null = null;

  get targetSignInfo(): Sign | undefined {
    return this.signs.find(s => s.word === this.targetSignWord);
  }

  get handZoomStyle(): Record<string, string> | null {
    const bbox = this.referenceBuckets?.hand_bbox;
    if (!bbox) return null;
    const bw = bbox.x_max - bbox.x_min;
    const bh = bbox.y_max - bbox.y_min;
    const scale = Math.min(8, 1 / Math.max(bw, bh, 0.05)); // uniform -- preserves aspect, no stretching
    const cx = (bbox.x_min + bbox.x_max) / 2;
    const cy = (bbox.y_min + bbox.y_max) / 2;
    const left = (0.5 - cx * scale) * 100;
    const top = (0.5 - cy * scale) * 100;
    return { width: `${scale * 100}%`, height: `${scale * 100}%`, left: `${left}%`, top: `${top}%` };
  }

  get handshapeLabel(): string {
    const HANDSHAPE_LABELS: Record<string, string> = {
      closed: 'Closed fist',
      half: 'Fingers half-curled',
      open: 'Open, fingers spread',
    };
    return HANDSHAPE_LABELS[this.referenceBuckets?.handshape ?? ''] ?? '';
  }

  get nmsShortLabel(): string {
    const NMS_LABELS: Record<string, string> = {
      wh_question_marker: 'furrowed brow',
      negation_marker: 'headshake',
    };
    return NMS_LABELS[this.targetSignInfo?.nms_type ?? ''] ?? 'facial marker';
  }

  get facialCheckAttempted(): boolean {
    return this.hasAnalyzedFacial;
  }

  get parameterCards(): { name: string; match: boolean | null; detail: string }[] {
    const params = this.lastResult?.parameters || {};
    const cards = ['handshape', 'location', 'orientation'].map(key => {
      const p = params[key];
      return {
        name: key.charAt(0).toUpperCase() + key.slice(1),
        match: p ? p.match : null,
        detail: p ? `target: ${p.target} · you: ${p.yours}` : '',
      };
    });
    cards.push({
      name: 'Hand count',
      match: this.lastResult?.hand_count_match ?? null,
      detail: '',
    });
    return cards;
  }

  get snapshotEntries(): { param: string; frame: SkeletonFrame }[] {
    const snaps = this.lastResult?.divergent_snapshots || {};
    return Object.entries(snaps).map(([param, frame]) => ({ param, frame }));
  }

  stepLabel(step: PrepStep): string {
    return STEP_LABELS[step] ?? step;
  }

  ngOnInit(): void {
    this.signsApi.getSigns().subscribe(signs => {
      this.signs = signs;
      if (signs.length > 0) this.targetSignWord = signs[0].word;
      this.onTargetChange();
    });

    this.subs.push(
      this.coachService.keypointStatus$.subscribe(msg => this.updateBboxes(msg)),
    );
  }

  ngAfterViewChecked(): void {
    if (this.shouldScroll) {
      const el = this.chatScrollEl?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
      this.shouldScroll = false;
    }
  }

  private pushChat(msg: ChatMessage): void {
    this.chatMessages.push(msg);
    this.shouldScroll = true;
  }

  // --- Screen 1: Preview ---

  onTargetChange(): void {
    const info = this.targetSignInfo;
    this.requiredSteps = info ? [...(info.manual_checks as PrepStep[]), ...(info.nms_required ? ['facial' as PrepStep] : [])] : [];
    this.loadReferenceVideo();
    this.referenceBuckets = null;
    this.signsApi.getReferenceBuckets(this.targetSignWord).subscribe({
      next: buckets => { this.referenceBuckets = buckets; },
      error: () => { this.referenceBuckets = null; },
    });
  }

  setPlaybackRate(rate: number): void {
    this.playbackRate = rate;
    this.previewVideosEl?.nativeElement.querySelectorAll('video').forEach(v => { v.playbackRate = rate; });
  }

  applyRate(event: Event): void {
    (event.target as HTMLVideoElement).playbackRate = this.playbackRate;
  }

  loadReferenceVideo(): void {
    this.signsApi.getVideo(this.targetSignWord).subscribe({
      next: res => { this.referenceVideoUrl = `${environment.apiBaseUrl}${res.video_url}`; },
      error: () => { this.referenceVideoUrl = null; },
    });
  }

  goToReady(): void {
    this.screen = 'ready';
    this.comments = [];
    this.ensureSession();
    setTimeout(() => this.cameraViewReady?.start(), 0);
  }

  // --- Screen 2: Get ready ---

  private sessionStarted = false;
  private ensureSession(): void {
    if (this.sessionStarted) {
      this.coachService.changeTarget(this.targetSignWord);
      return;
    }
    this.sessionStarted = true;
    this.coachService.startSession(this.targetSignWord);
    this.subs.push(
      this.coachService.coaching$.subscribe(msg => this.onCoachingMessage(msg)),
    );
  }

  onFrame(base64Frame: string): void {
    this.coachService.sendFrame(base64Frame);
  }

  /** One action, not a gated checklist: runs every required check in turn
   * (sequential, not parallel -- each backend check consumes and clears
   * the shared frame buffer, so concurrent calls would race on it) and
   * turns each result into a plain-language comment. Never blocks Record. */
  async analyze(): Promise<void> {
    if (this.analyzing) return;
    this.analyzing = true;
    this.comments = [];
    for (const step of this.requiredSteps) {
      await new Promise(r => setTimeout(r, CHECK_HOLD_MS));
      const result = await this.coachService.checkStep(this.targetSignWord, step);
      if (step === 'facial') this.hasAnalyzedFacial = true;
      this.comments.push(this.buildComment(step, result));
    }
    this.analyzing = false;
  }

  private buildComment(step: PrepStep, result: StepResult | null): { step: PrepStep; label: string; text: string; tone: 'good' | 'warn' | 'neutral' } {
    const label = this.stepLabel(step);
    if (!result || !result.applicable) {
      return { step, label, text: "Couldn't analyze this yet — try again with a clearer view.", tone: 'neutral' };
    }
    if (step === 'facial') {
      if (!result.present) return { step, label, text: "Couldn't see your face clearly — make sure it's in frame.", tone: 'neutral' };
      return result.detected
        ? { step, label, text: `Detected ${this.nmsShortLabel} — nice.`, tone: 'good' }
        : { step, label, text: `Didn't detect ${this.nmsShortLabel} yet — this sign usually needs it.`, tone: 'warn' };
    }
    if (step === 'hand_count') {
      return result.match
        ? { step, label, text: 'Hand count looks right.', tone: 'good' }
        : { step, label, text: "Hand count doesn't look right yet — this sign uses a different number of hands than what I'm seeing.", tone: 'warn' };
    }
    if (result.match === null || result.match === undefined) {
      return { step, label, text: "Couldn't get a clear read — try adjusting your position.", tone: 'neutral' };
    }
    return result.match
      ? { step, label, text: `Looks right — matches what '${this.targetSignWord}' needs.`, tone: 'good' }
      : { step, label, text: `Looks like '${result.yours}', but '${this.targetSignWord}' usually uses '${result.target}'.`, tone: 'warn' };
  }

  private updateBboxes(msg: CoachingMessage): void {
    this.faceBboxStyle = this.bboxToStyle(msg.face_bbox ?? null);
    this.handBboxStyle = this.bboxToStyle(msg.hand_bbox ?? null);
    if (msg.pose_detected !== undefined) {
      this.liveStatus = {
        pose: !!msg.pose_detected,
        leftHand: !!msg.left_hand_detected,
        rightHand: !!msg.right_hand_detected,
        face: !!msg.face_detected,
      };
    }
  }

  private bboxToStyle(bbox: BBox | null | undefined): Record<string, string> | null {
    if (!bbox) return null;
    // mirrored view (scaleX(-1)) -- flip x so the overlay lands on the visually mirrored position
    const left = (1 - bbox.x_max) * 100;
    const width = (bbox.x_max - bbox.x_min) * 100;
    const top = bbox.y_min * 100;
    const height = (bbox.y_max - bbox.y_min) * 100;
    return { left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` };
  }

  // --- Screen 3: Record -> feedback ---

  goToRecord(): void {
    this.screen = 'record';
    this.recordState = 'ready';
    setTimeout(() => this.cameraViewRecord?.start(), 0);
  }

  onRecordClick(): void {
    if (this.recordState !== 'ready') return;
    this.recordState = 'countdown';
    this.countdownValue = COUNTDOWN_SECONDS;
    this.countdownTimer = setInterval(() => {
      this.countdownValue--;
      if (this.countdownValue <= 0) {
        if (this.countdownTimer) clearInterval(this.countdownTimer);
        this.startRecording();
      }
    }, 1000);
  }

  private startRecording(): void {
    this.recordState = 'recording';
    this.recordingProgress = 0;
    this.recordingStartTime = Date.now();
    this.recordingTimer = setInterval(() => {
      const elapsed = Date.now() - this.recordingStartTime;
      this.recordingProgress = Math.min((elapsed / RECORDING_DURATION_MS) * 100, 100);
      if (elapsed >= RECORDING_DURATION_MS) {
        if (this.recordingTimer) clearInterval(this.recordingTimer);
        this.coachService.requestPrediction(this.targetSignWord);
      }
    }, 100);
  }

  retryRecording(): void {
    this.recordState = 'ready';
    this.lastResult = null;
  }

  private onCoachingMessage(msg: CoachingMessage): void {
    if (msg.type === 'session_start') {
      this.pushChat({ role: 'system', text: msg.message || 'Session started' });
    } else if (msg.type === 'coaching') {
      this.lastResult = msg;
      this.recordState = 'done';
      this.attemptCount = msg.attempt_number || this.attemptCount + 1;
      if (msg.correct) this.correctCount++;
      this.sessionAccuracy = Math.round((msg.session_accuracy || 0) * 100);
      this.pushChat({ role: 'coach', text: msg.feedback || '', coachingData: msg });
    } else if (msg.type === 'retry') {
      this.recordState = 'ready';
      this.pushChat({ role: 'system', text: msg.message || 'Please try the sign again.' });
    } else if (msg.type === 'chat_reply') {
      this.pushChat({ role: 'coach', text: msg.message || '' });
    } else if (msg.type === 'session_end') {
      this.pushChat({ role: 'system', text: msg.message || 'Session ended' });
    }
  }

  sendChat(): void {
    const text = this.chatInput.trim();
    if (!text) return;
    this.pushChat({ role: 'user', text });
    this.coachService.sendChat(text);
    this.chatInput = '';
  }

  jumpToScreen(id: Screen): void {
    if (id === 'ready' && !this.sessionStarted) { this.goToReady(); return; }
    if ((id === 'ready' || id === 'record') && !this.sessionStarted) this.ensureSession();
    this.screen = id;
    if (id === 'ready') setTimeout(() => this.cameraViewReady?.start(), 0);
    if (id === 'record') setTimeout(() => this.cameraViewRecord?.start(), 0);
  }

  ngOnDestroy(): void {
    if (this.countdownTimer) clearInterval(this.countdownTimer);
    if (this.recordingTimer) clearInterval(this.recordingTimer);
    if (this.sessionStarted) this.coachService.endSession();
    this.subs.forEach(s => s.unsubscribe());
  }
}
