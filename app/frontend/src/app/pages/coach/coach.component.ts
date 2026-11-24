import { Component, inject, OnDestroy, OnInit, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { ActivatedRoute } from '@angular/router';
import { Subscription } from 'rxjs';
import { CameraViewComponent } from '../../shared/components/camera-view/camera-view.component';
import { SkeletonCompareComponent } from '../../shared/components/skeleton-compare/skeleton-compare.component';
import { Hand3dComponent } from '../../shared/components/hand-3d/hand-3d.component';
import { SignSkinComponent } from '../../shared/components/sign-skin/sign-skin.component';
import { VideoBonesComponent } from '../../shared/components/video-bones/video-bones.component';
import { LookPickerComponent } from '../../shared/components/look-picker/look-picker.component';
import { AttemptChartsComponent } from '../../shared/components/attempt-charts/attempt-charts.component';
import { ScoreCardComponent } from '../../shared/components/score-card/score-card.component';
import { SignFrameComponent } from '../../shared/components/sign-frame/sign-frame.component';
import { CoachSessionService } from '../../core/services/coach-session.service';
import { SignsApiService } from '../../core/services/signs-api.service';
import { BBox, CoachingMessage, PrepStep, ReferenceBuckets, Sign, SkeletonFrame, StepResult, HandPath } from '../../core/models/sign.model';
import { environment } from '../../../environments/environment';

type Screen = 'intro' | 'preview' | 'ready' | 'record';
type StepStatus = 'pending' | 'checking' | 'match' | 'mismatch' | 'na';

interface ChatMessage {
  role: 'user' | 'coach' | 'system';
  text: string;
  coachingData?: CoachingMessage;
}

interface PrepStepView {
  id: 'shape' | 'place' | 'move';
  label: string;
  instruction: string;
  /** The CP checks this learner-facing step covers (empty = not auto-checked). */
  cps: PrepStep[];
  shots: { label: string; frac: number; zoom: boolean }[];
  clip?: boolean;
  hand3d?: boolean;
  tone: 'good' | 'warn' | 'neutral';
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
  imports: [CameraViewComponent, SkeletonCompareComponent, Hand3dComponent, SignSkinComponent, VideoBonesComponent, LookPickerComponent, AttemptChartsComponent, SignFrameComponent, ScoreCardComponent, FormsModule],
  template: `
    <div class="max-w-7xl mx-auto" [style.paddingRight]="showChat ? '380px' : '0'">

      <!-- Steps -->
      @if (screen !== 'intro') {
      <nav class="mb-6 flex items-center gap-3 text-sm" aria-label="Practice steps">
        @for (s of screenSteps; track s.id; let i = $index) {
          @if (i > 0) { <span class="h-px w-8 bg-border"></span> }
          <button (click)="jumpToScreen(s.id)" class="flex items-center gap-2 transition-colors"
            [class]="screen === s.id ? 'text-ink' : 'text-ink-faint hover:text-ink-muted'" [attr.aria-current]="screen === s.id ? 'step' : null">
            <span class="w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold border"
              [class]="screen === s.id ? 'bg-accent text-white border-accent' : stepIndex(s.id) < stepIndex(screen) ? 'border-teal text-teal' : 'border-border'">
              @if (stepIndex(s.id) < stepIndex(screen)) {
                <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.6"><path stroke-linecap="round" stroke-linejoin="round" d="m5 12.5 4.5 4.5L19 7.5" /></svg>
              } @else { {{ i + 1 }} }
            </span>
            <span [class.font-medium]="screen === s.id">{{ s.label }}</span>
          </button>
        }
        <span class="flex-1"></span>
        <button (click)="screen = 'intro'" class="text-xs text-ink-faint hover:text-ink underline underline-offset-4 decoration-border">How it works</button>
      </nav>
      }

      <!-- ============== INTRO: HOW PRACTICE WORKS ============== -->
      @if (screen === 'intro') {
        <div class="max-w-4xl mx-auto pt-4">
          <h1 class="font-display text-5xl text-ink">How practice works</h1>
          <p class="text-ink-muted mt-3 text-lg">Three short steps. About two minutes for each sign.</p>

          <ol class="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
            @for (step of introSteps; track step.title; let i = $index) {
              <li class="rounded-2xl bg-surface border border-border/70 p-5 flex flex-col">
                <div class="h-28 rounded-xl bg-surface-sunk flex items-center justify-center mb-4" [innerHTML]="step.art"></div>
                <div class="text-xs text-ink-faint">Step {{ i + 1 }}</div>
                <h2 class="font-display text-2xl text-ink mt-0.5">{{ step.title }}</h2>
                <p class="text-sm text-ink-muted leading-relaxed mt-2">{{ step.body }}</p>
              </li>
            }
          </ol>

          <div class="mt-8 flex items-center gap-5 flex-wrap">
            <button (click)="finishIntro()" class="h-12 px-7 rounded-xl bg-accent text-white font-medium hover:bg-accent/90 transition-colors">
              Start with <span class="capitalize">{{ targetSignWord }}</span> →
            </button>
            <label class="flex items-center gap-2 text-sm text-ink-muted cursor-pointer select-none">
              <input type="checkbox" [(ngModel)]="hideIntro" class="w-4 h-4 accent-[rgb(var(--color-accent))]" />
              Don't show this again
            </label>
          </div>
        </div>
      }

      <!-- ============== SCREEN 1: LEARN THE SIGN (no camera) ============== -->
      @if (screen === 'preview') {
        <div class="rounded-2xl bg-surface border border-border/50 p-5">
          @if (unavailableNotice) {
            <div class="mb-4 rounded-lg bg-surface-sunk px-4 py-2.5 text-sm text-ink-muted flex items-center justify-between gap-3">
              <span>{{ unavailableNotice }}</span>
              <button (click)="unavailableNotice = ''" class="text-ink-faint hover:text-ink" aria-label="Dismiss">✕</button>
            </div>
          }
          <div class="flex items-start justify-between gap-4 flex-wrap mb-4">
            <div class="min-w-0">
              <h1 class="font-display text-5xl text-ink capitalize">{{ targetSignWord }}</h1>
              @if (targetSignInfo?.description) { <p class="text-ink-muted mt-2">{{ targetSignInfo?.description }}</p> }
              <p class="text-sm text-ink mt-3"><span class="text-ink-faint">Now:</span> watch how the hand moves, then press <strong class="font-medium">Start prep</strong>.</p>
            </div>
            <select
              [(ngModel)]="targetSignWord"
              (ngModelChange)="onTargetChange()"
              class="h-10 w-48 bg-surface border border-border/50 rounded-xl px-3 text-ink text-sm capitalize focus:outline-none focus:border-accent/50"
              aria-label="Choose a sign"
            >
              @for (sign of practiceSigns; track sign.word) {
                <option [value]="sign.word">{{ sign.word }}</option>
              }
            </select>
          </div>

          <!-- the sign itself: real video + the hand in 3D -->
          <div class="grid grid-cols-1 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] gap-3">
            <div>
              <div #previewVideos>
                @if (referenceVideoUrl) {
                  <app-sign-frame [src]="referenceVideoUrl" [paths]="handPaths" phase="live" [playbackRate]="playbackRate" label="The dot follows the hand" />
                } @else {
                  <div class="rounded-xl bg-black aspect-video flex items-center justify-center text-white/60 text-sm">No example video for "{{ targetSignWord }}"</div>
                }
              </div>
              <div class="flex items-center gap-2 mt-2 flex-wrap">
                <span class="text-xs text-ink-faint mr-1">Speed</span>
                @for (r of playbackRates; track r.value) {
                  <button (click)="setPlaybackRate(r.value)" class="text-xs font-semibold px-3 py-1 rounded-full border"
                    [class]="!stepMode && playbackRate === r.value ? 'bg-accent text-white border-accent' : 'bg-surface-sunk text-ink-muted border-border/50'">{{ r.label }}</button>
                }
                <span class="w-px h-4 bg-border/60 mx-1"></span>
                <button (click)="toggleStepMode()" class="text-xs font-semibold px-3 py-1 rounded-full border"
                  [class]="stepMode ? 'bg-accent text-white border-accent' : 'bg-surface-sunk text-ink-muted border-border/50'">Step through</button>
                @if (stepMode) {
                  <button (click)="stepFrame(-1)" class="text-xs font-semibold w-8 h-7 rounded-full border bg-surface-sunk text-ink-muted border-border/50">◀</button>
                  <button (click)="stepFrame(1)" class="text-xs font-semibold w-8 h-7 rounded-full border bg-surface-sunk text-ink-muted border-border/50">▶</button>
                }
              </div>
            </div>
            <div>
              <!-- v3: hands and face as separate 3D close-ups, following the video.
                   Only a sign with no skin data falls back to the older 3D hand (hands panel only). -->
              <div class="flex flex-col gap-3">
                @if (skinAvailable !== false) {
                  <app-sign-skin [word]="targetSignWord" show="hands" label="Hands, close up" aspect="16/10"
                    [videoHost]="previewVideosEl?.nativeElement ?? null" (available)="skinAvailable = $event" />
                } @else if (referenceSkeleton?.length) {
                  <app-hand-3d [targetOnly]="true" [referenceFrames]="referenceSkeleton" [speedMultiplier]="playbackRate" [loopSeconds]="signSeconds" />
                }
                <app-sign-skin [word]="targetSignWord" show="face" label="Face, close up" aspect="16/10"
                  emptyText="No face in this example." [videoHost]="previewVideosEl?.nativeElement ?? null" />
              </div>
              <p class="mt-2 text-[11px] text-ink-muted">Both follow the video and its speed. Drag to turn, scroll to zoom.</p>
              <div class="mt-3"><app-look-picker /></div>
            </div>
          </div>

          <!-- the steps of the sign: start / middle / end positions -->
          @if (referenceVideoUrl) {
            <h2 class="font-display text-2xl text-ink mt-8 mb-3">How it moves</h2>
            <div class="grid grid-cols-3 gap-3">
              @for (ph of phaseCards; track ph.label; let i = $index) {
                <div>
                  <app-sign-frame [src]="referenceVideoUrl" [paths]="handPaths" [phase]="ph.phase" [label]="(i + 1) + '. ' + ph.label" />
                  <p class="text-sm text-ink-muted mt-2">{{ ph.caption }}</p>
                </div>
              }
            </div>
          }

          @if (targetSignInfo?.nms_required) {
            <div class="mt-4 rounded-xl bg-amber-500/10 border border-amber-500/30 px-4 py-3">
              <p class="text-sm font-semibold text-amber-600 dark:text-amber-400">This sign also needs a facial expression: {{ nmsShortLabel }}</p>
              @if (targetSignInfo?.nms_notes) { <p class="text-xs text-ink-muted mt-1">{{ targetSignInfo?.nms_notes }}</p> }
            </div>
          }

          <!-- other recordings of the same sign -->
          @if (variations.length) {
            <h2 class="font-display text-2xl text-ink mt-8 mb-1">Other people signing it</h2>
            @if (variantLabel) {
              <p class="text-[11px] text-ink-muted mb-2">This sign also has a {{ variantLabel.toLowerCase() }}, which is correct ASL too. Practice here uses the main example above, and the coach grades against it.</p>
            } @else {
              <p class="text-[11px] text-ink-muted mb-2">Signers vary, and some use a different version. The coach grades against the main example above.</p>
            }
            <div class="grid grid-cols-3 gap-2">
              @for (v of variations; track v.video_url) {
                <div class="relative rounded-lg overflow-hidden bg-black border border-border/50" style="aspect-ratio: 16/10;">
                  <video [src]="apiBase + v.video_url" autoplay loop [muted]="true" playsinline class="w-full h-full object-cover"></video>
                  @if (v.label) {
                    <span class="absolute left-1.5 top-1.5 px-1.5 py-0.5 rounded bg-surface/90 text-ink text-[10px] font-medium">{{ v.label }}</span>
                  }
                </div>
              }
            </div>
          }

          <div class="mt-6 flex items-center gap-3">
            <button (click)="goToReady()" class="px-6 h-11 bg-accent hover:bg-accent/90 text-white rounded-xl font-medium text-sm">Start prep →</button>
            <span class="text-xs text-ink-muted">Next, the camera turns on and I'll walk you through the position.</span>
          </div>
        </div>
      }

      <!-- ============== SCREEN 2: PREP STEPS ============== -->
      @if (screen === 'ready') {
        <div class="rounded-2xl bg-surface border border-border/50 p-4">
          <div class="flex items-baseline justify-between mb-3 flex-wrap gap-x-4 gap-y-1">
            <h1 class="font-display text-3xl text-ink">Get set for <span class="capitalize">{{ targetSignWord }}</span></h1>
            <span class="text-xs text-ink-muted">{{ hasReference ? 'Copy the picture and hold still. It moves on when it looks right.' : "Practice only: this sign can't be checked automatically yet." }}</span>
          </div>

          <div class="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_340px] gap-4">
            <!-- signing area: as big as the screen allows -->
            <div>
              <div class="relative rounded-xl overflow-hidden" style="max-height: 70vh;">
                <app-camera-view #cameraViewReady [mirrorView]="true" maxHeight="70vh" (frameCaptured)="onFrame($event)" />
                @if (faceBboxStyle) {
                  <div class="absolute rounded-lg pointer-events-none border-2 border-indigo-400" [style]="faceBboxStyle"></div>
                }
                @if (handBboxStyle) {
                  <div class="absolute rounded-lg pointer-events-none border-2 border-blue-500" [style]="handBboxStyle"></div>
                }
                <div class="absolute top-2 left-2 flex gap-1.5 flex-wrap">
                  <span class="text-[10px] font-semibold px-2 py-0.5 rounded-full" [class]="liveStatus.pose ? 'bg-emerald-500/80 text-white' : 'bg-black/50 text-white/70'">{{ liveStatus.pose ? '✓' : '…' }} you</span>
                  <span class="text-[10px] font-semibold px-2 py-0.5 rounded-full" [class]="liveStatus.leftHand || liveStatus.rightHand ? 'bg-emerald-500/80 text-white' : 'bg-black/50 text-white/70'">{{ liveStatus.leftHand || liveStatus.rightHand ? '✓' : '…' }} hands</span>
                  @if (requiredSteps.includes('facial')) {
                    <span class="text-[10px] font-semibold px-2 py-0.5 rounded-full" [class]="liveStatus.face ? 'bg-emerald-500/80 text-white' : 'bg-black/50 text-white/70'">{{ liveStatus.face ? '✓' : '…' }} face</span>
                  }
                </div>
              </div>
            </div>

            <!-- one prep step at a time -->
            <div class="flex flex-col min-w-0">
              <div class="flex items-center gap-1.5 mb-3 flex-wrap">
                @for (st of prepSteps; track st.id; let i = $index) {
                  <button (click)="prepIndex = i" [title]="st.label"
                    class="h-8 min-w-8 px-2 rounded-full text-xs font-bold flex items-center justify-center gap-1 border transition-colors"
                    [class]="i === prepIndex ? 'bg-accent text-white border-accent' : st.tone === 'good' ? 'bg-emerald-500/15 text-emerald-500 border-emerald-500/40' : st.tone === 'warn' ? 'bg-amber-500/15 text-amber-500 border-amber-500/40' : 'bg-surface-sunk text-ink-muted border-border/50'"
                  >{{ i + 1 }}</button>
                }
              </div>

              @if (currentPrep; as st) {
                <div class="text-xs text-ink-faint">{{ 'Step ' + (prepIndex + 1) + ' of ' + prepSteps.length }}</div>
                <h2 class="font-display text-2xl text-ink mb-3">{{ st.label }}</h2>

                @if (st.hand3d && referenceSkeleton?.length) {
                  <div class="mb-2"><app-hand-3d [targetOnly]="true" [referenceFrames]="referenceSkeleton" [speedMultiplier]="playbackRate" [loopSeconds]="signSeconds" /></div>
                }
                @if (referenceVideoUrl && st.clip) {
                  <div class="mb-3"><app-sign-frame [src]="referenceVideoUrl" [paths]="handPaths" phase="live" label="The motion" /></div>
                }
                @if (referenceVideoUrl) {
                  @for (sh of st.shots; track sh.label) {
                    @if (!sh.zoom && handPaths.length) {
                      <div class="mb-3"><app-sign-frame [src]="referenceVideoUrl" [paths]="handPaths" phase="full" [label]="sh.label" /></div>
                    } @else {
                    <div class="relative rounded-lg overflow-hidden bg-black border border-border/50 mb-2" style="aspect-ratio: 16/10;">
                      <video [src]="referenceVideoUrl" [muted]="true" playsinline preload="auto"
                        [class]="sh.zoom && handZoomStyle ? 'absolute max-w-none object-cover' : 'w-full h-full object-cover'"
                        [style]="sh.zoom ? handZoomStyle : null"
                        (loadedmetadata)="seekStill($event, sh.frac)"></video>
                      <span class="absolute top-2 left-2 text-[11px] font-medium text-white/90 bg-black/45 px-2 py-0.5 rounded-md">{{ sh.label }}</span>
                    </div>
                    }
                  }
                }

                <p class="text-sm text-ink leading-relaxed">{{ st.instruction }}</p>
                @if (isLiveCheckable(st)) {
                  <div class="mt-2 rounded-lg px-3 py-2 text-sm flex items-start gap-2"
                    [class]="liveFeedback?.tone === 'good' ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400' : liveFeedback?.tone === 'warn' ? 'bg-amber-500/10 text-amber-700 dark:text-amber-400' : 'bg-surface-sunk text-ink-muted'">
                    @if (!liveFeedback || liveFeedback.tone === 'neutral') { <span class="w-3.5 h-3.5 mt-0.5 shrink-0 border-2 border-current/30 border-t-current rounded-full animate-spin"></span> }
                    <span>{{ liveFeedback?.text || 'Hold this position… checking' }}</span>
                  </div>
                }

                @if (st.id === 'move') {
                  <button (click)="goToRecord()" class="mt-3 w-full h-11 rounded-xl font-medium text-sm text-white bg-red-500 hover:bg-red-400">● Record (3-2-1 starts right away)</button>
                  <div class="flex gap-2 mt-2">
                    <button (click)="prepIndex = prepIndex - 1" class="flex-1 h-9 rounded-xl text-xs border border-border/50 bg-surface-sunk text-ink">← Back</button>
                    @if (hasReference) {
                      <button (click)="analyze()" [disabled]="analyzing" class="flex-1 h-9 rounded-xl text-xs border border-border/50 bg-surface-sunk text-ink disabled:opacity-60 flex items-center justify-center gap-1.5">
                        @if (analyzing) { <span class="w-3 h-3 border-2 border-current/30 border-t-current rounded-full animate-spin"></span> Checking… hold still } @else { Check every part first }
                      </button>
                    }
                  </div>
                  @if (analyzeLines.length) {
                    <div class="mt-3 rounded-lg bg-surface-sunk border border-border/50 px-3 py-2.5">
                      <p class="text-xs font-semibold text-ink mb-1">{{ analyzeHead }}</p>
                      @for (l of analyzeLines; track $index) {
                        <p class="text-xs leading-relaxed" [class]="l.tone === 'warn' ? 'text-amber-600 dark:text-amber-400' : l.tone === 'good' ? 'text-emerald-600 dark:text-emerald-400' : 'text-ink-muted'">{{ l.text }}</p>
                      }
                    </div>
                  }
                } @else {
                  <div class="flex gap-2 mt-auto pt-3">
                    <button (click)="prepIndex = prepIndex - 1" [disabled]="prepIndex === 0" class="flex-1 h-10 rounded-xl text-sm border border-border/50 bg-surface-sunk text-ink disabled:opacity-40">← Back</button>
                    <button (click)="prepIndex = prepIndex + 1" class="flex-1 h-10 rounded-xl text-sm font-medium text-white bg-accent hover:bg-accent/90">Next →</button>
                  </div>
                }
              }
            </div>
          </div>
        </div>
      }

      <!-- ============== SCREEN 3: RECORD -> FEEDBACK ============== -->
      @if (screen === 'record') {
        <div class="rounded-2xl bg-surface border border-border/50 p-5">
          @if (recordState !== 'done') {
            <h1 class="font-display text-3xl text-ink mb-1">Your turn</h1>
            <p class="text-ink-muted mb-4"><span class="text-ink-faint">Now:</span> when the countdown ends, sign <span class="capitalize">{{ targetSignWord }}</span> once. You have 5 seconds.</p>

            <div class="relative rounded-xl overflow-hidden" style="max-height: 70vh;">
              <app-camera-view #cameraViewRecord [mirrorView]="true" maxHeight="70vh" (frameCaptured)="onFrame($event)" />

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
            <h1 class="font-display text-3xl text-ink mb-4">How it went</h1>

            <div class="flex items-center justify-between gap-3 rounded-xl border px-4 py-2.5 mb-4" [class]="lastResult?.verdict === 'good' ? 'border-emerald-500/40 bg-emerald-500/5' : 'border-amber-500/40 bg-amber-500/5'">
              <span class="text-sm text-ink">{{ lastResult?.graded === false ? 'Not graded: no reference recording for' : 'vs.' }} "<span class="capitalize">{{ targetSignWord }}</span>" — the coach's notes are in the chat →</span>
              <span class="text-xs font-medium px-3 py-1 rounded-full shrink-0"
                [class]="lastResult?.verdict === 'good' ? 'bg-emerald-500/15 text-emerald-500' : 'bg-amber-500/15 text-amber-500'"
              >{{ verdictLabel }}</span>
            </div>

            @if (lastResult?.looks_like) {
              <div class="rounded-xl bg-amber-500/10 border border-amber-500/30 px-4 py-3 mb-4">
                <p class="text-sm text-ink"><strong class="capitalize">Heads up:</strong> your handshape, location, and palm direction actually matched <strong class="capitalize">"{{ lastResult?.looks_like }}"</strong> more closely than "{{ targetSignWord }}". Here's that sign next to yours:</p>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                  <app-skeleton-compare
                    [frames]="lastResult?.attempt_skeleton_raw ?? null"
                    [speedMultiplier]="skeletonSpeed"
                    color="#3f5fc4"
                    label="Your motion"
                  />
                  <app-skeleton-compare
                    [frames]="lastResult?.looks_like_skeleton ?? null"
                    [speedMultiplier]="skeletonSpeed"
                    color="#ab7a14"
                    [label]="looksLikeLabel"
                  />
                </div>
              </div>
            }

            <!-- tabs: Video / Hands / Face / Motion / Score -->
            <div class="flex items-center justify-between mb-3 flex-wrap gap-2">
              <div class="flex items-center gap-1 rounded-full bg-surface-sunk border border-border/50 p-1" role="tablist">
                @for (t of feedbackTabs; track t.id) {
                  <button role="tab" [attr.aria-selected]="feedbackTab === t.id" (click)="feedbackTab = t.id"
                    class="text-xs font-semibold px-3.5 py-1.5 rounded-full transition-colors"
                    [class]="feedbackTab === t.id ? 'bg-accent text-white' : 'text-ink-muted hover:text-ink'">{{ t.label }}</button>
                }
              </div>
              @if (feedbackTab !== 'metrics') {
                <div class="flex items-center gap-2">
                  <span class="text-xs text-ink-faint mr-1">Speed</span>
                  @for (r of skeletonRates; track r.value) {
                    <button (click)="skeletonSpeed = r.value" class="text-xs font-semibold px-3 py-1 rounded-full border"
                      [class]="skeletonSpeed === r.value ? 'bg-accent text-white border-accent' : 'bg-surface-sunk text-ink-muted border-border/50'">{{ r.label }}</button>
                  }
                </div>
              }
            </div>

            @if (feedbackTab === 'hands' || feedbackTab === 'face' || feedbackTab === 'motion') {
              <div class="mb-3"><app-look-picker /></div>
            }
            @switch (feedbackTab) {
              @case ('video') {
                <div class="flex items-center gap-4 mb-2 text-xs text-ink-muted">
                  <label class="flex items-center gap-1.5 cursor-pointer"><input type="checkbox" [(ngModel)]="showMyBones"> My bones</label>
                  <label class="flex items-center gap-1.5 cursor-pointer"><input type="checkbox" [(ngModel)]="showTargetBones"> Target bones</label>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-2">
                  <app-video-bones [src]="recordedAttemptUrl" [skin]="lastResult?.attempt_skin ?? null" [mirror]="true"
                    [showBones]="showMyBones" [speed]="skeletonSpeed" label="You" emptyText="Your recording isn't available" />
                  <app-video-bones [src]="referenceVideoUrl" [word]="targetSignWord" [showBones]="showTargetBones" [speed]="skeletonSpeed" label="Target" />
                </div>
                <p class="text-xs text-ink-muted mb-4">Your video is mirrored, as you saw yourself while recording.</p>
              }
              @case ('hands') {
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                  <app-sign-skin [data]="lastResult?.attempt_skin ?? null" show="hands" label="Your hands" [speed]="skeletonSpeed"
                    emptyText="Your hands weren't seen clearly in this recording." />
                  <app-sign-skin [word]="targetSignWord" show="hands" label="Target hands" [speed]="skeletonSpeed" />
                </div>
              }
              @case ('face') {
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                  <app-sign-skin [data]="lastResult?.attempt_skin ?? null" show="face" label="Your face" [speed]="skeletonSpeed"
                    emptyText="Your face wasn't seen clearly in this recording." />
                  <app-sign-skin [word]="targetSignWord" show="face" label="Target face" [speed]="skeletonSpeed" />
                </div>
              }
              @case ('motion') {
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-2">
                  <app-sign-skin [data]="lastResult?.attempt_skin ?? null" show="all" label="You" [speed]="skeletonSpeed"
                    emptyText="Nothing was seen clearly in this recording." />
                  <app-sign-skin [word]="targetSignWord" show="all" label="Target" [speed]="skeletonSpeed" />
                </div>
                <p class="text-xs text-ink-muted mb-4">Face and hands together, so you can see where the hands are relative to the face.</p>
              }
              @case ('metrics') {
                <app-score-card [scores]="lastResult?.parameter_scores" />
                @if (lastResult?.parameter_scores) {
                  <h4 class="text-xs font-semibold uppercase tracking-wider text-ink-faint mb-2">More checks</h4>
                }
                @if (lastResult?.charts) {
                  <div class="mb-4"><app-attempt-charts [charts]="lastResult?.charts ?? null" /></div>
                }
                <div class="grid grid-cols-1 md:grid-cols-2 gap-2 mb-2">
                  @for (p of parameterCards; track p.name) {
                    <div class="rounded-xl border px-4 py-3" [class]="p.match === true ? 'border-emerald-500/30' : p.hint ? 'border-border/60' : p.match === false ? 'border-amber-500/30' : 'border-border/50 border-dashed opacity-70'">
                      <div class="flex items-center justify-between">
                        <span class="text-sm font-semibold text-ink">{{ p.name }}</span>
                        <span class="text-[11px] font-medium px-2 py-0.5 rounded-md"
                          [class]="p.match === true ? 'bg-emerald-500/15 text-emerald-500' : p.hint ? 'bg-surface-sunk text-ink-muted' : p.match === false ? 'bg-amber-500/15 text-amber-500' : 'bg-surface text-ink-faint border border-dashed border-border'"
                        >{{ p.match === true ? 'match' : p.hint ? 'worth a look' : p.match === false ? 'mismatch' : 'not graded' }}</span>
                      </div>
                      @if (p.why) { <p class="text-sm text-ink mt-1.5">{{ p.why }}</p> }
                      @if (p.hint) { <p class="text-[11px] text-ink-faint mt-1">This check isn't reliable yet, so take it as a hint, not a mistake.</p> }
                      @if (p.detail) { <p class="text-[11px] text-ink-faint mt-1">{{ p.detail }}</p> }
                    </div>
                  }
                </div>
              }
            }

            <div class="flex gap-3 mt-5">
              <button (click)="retryRecording()" class="px-6 h-11 bg-accent hover:bg-accent/90 text-white rounded-xl font-medium text-sm">↺ Try again</button>
              <button (click)="jumpToScreen('preview')" class="px-4 h-11 bg-surface-sunk border border-border/50 text-ink rounded-xl text-sm">Choose another sign</button>
            </div>
          }
        </div>
      }
    </div>

    <!-- Chat panel: only on Record & feedback, so prep gets the full width -->
    @if (showChat) {
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
                <div class="flex justify-start"><div class="max-w-[90%] rounded-2xl rounded-bl-sm px-4 py-3 bg-accent/10"><p class="text-sm text-ink leading-relaxed whitespace-pre-line">{{ msg.text }}</p></div></div>
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
    }
  `,
})
export class CoachComponent implements OnInit, OnDestroy, AfterViewChecked {
  @ViewChild('previewVideos') previewVideosEl?: ElementRef<HTMLDivElement>;
  @ViewChild('cameraViewReady') cameraViewReady?: CameraViewComponent;
  @ViewChild('cameraViewRecord') cameraViewRecord?: CameraViewComponent;
  @ViewChild('chatScroll') chatScrollEl?: ElementRef<HTMLDivElement>;

  private coachService = inject(CoachSessionService);
  private signsApi = inject(SignsApiService);
  private route = inject(ActivatedRoute);
  private subs: Subscription[] = [];
  private shouldScroll = false;

  readonly screenSteps: { id: Screen; label: string }[] = [
    { id: 'preview', label: 'Learn the sign' },
    { id: 'ready', label: 'Prep' },
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
  private mediaRecorder: MediaRecorder | null = null;
  private recordedChunks: Blob[] = [];
  /** The user's own actual attempt video, not a skeleton derived from it --
   * see startAttemptRecording(). Null until the first attempt finishes, or
   * permanently null if MediaRecorder isn't supported in this browser. */
  recordedAttemptUrl: string | null = null;
  private recordingStartTime = 0;

  referenceVideoUrl: string | null = null;
  /** Seconds, derived from frame_start/frame_end/fps (see loadReferenceVideo)
   * -- null end means "no known trim window, play/loop the whole file"
   * (WLASL's frame_end === -1 convention: this file already IS the clean clip). */
  referenceTrimStart = 0;
  referenceTrimEnd: number | null = null;
  readonly playbackRates = [
    { label: '0.5×', value: 0.5 },
    { label: '1×', value: 1 },
    { label: '1.5×', value: 1.5 },
  ];
  playbackRate = 1;
  /** "Spell it out" mode -- P1 in the late-signers paper asked to slow a
   * sign down enough to step through it, which is a different need than
   * just playing it slower (speed is continuous, this is discrete and
   * learner-controlled). Pauses both preview videos and steps them a fixed
   * time delta at a time instead of playing. */
  stepMode = false;
  private readonly STEP_SECONDS = 0.1;
  readonly skeletonRates = [
    { label: '0.5×', value: 0.5 },
    { label: '1×', value: 1 },
    { label: '1.5×', value: 1.5 },
  ];
  skeletonSpeed = 1;
  readonly recordingSeconds = RECORDING_DURATION_MS / 1000;
  /** Feedback screen used to show all 4 skeleton panels (body x2, hands x2)
   * stacked at once -- split into tabs per explicit request so only one
   * comparison is on screen at a time. */
  readonly feedbackTabs: { id: 'video' | 'hands' | 'face' | 'motion' | 'metrics'; label: string }[] = [
    { id: 'video', label: 'Video' },
    { id: 'hands', label: 'Hands' },
    { id: 'face', label: 'Face' },
    { id: 'motion', label: 'Motion' },
    { id: 'metrics', label: 'Score' },
  ];
  /** Video tab: draw each person's tracked skeleton on their own video. */
  showMyBones = true;
  showTargetBones = false;
  /** Leads with the real-video comparison, not the skeleton -- an actual
   * human next to an actual human is the primary view now; the skeleton
   * tabs are the diagnostic/secondary ones, not the other way around. */
  feedbackTab: 'video' | 'hands' | 'face' | 'motion' | 'metrics' = 'video';

  /** Only what/who/how/no have a regenerated reference face mesh so far
   * (see scripts/auto_screen_references.py) -- used to show a plain-
   * language note instead of two silently-empty-looking face panels. */
  get looksLikeLabel(): string {
    return `"${this.lastResult?.looks_like ?? ''}" looks like this`;
  }

  get hasFaceData(): boolean {
    const seq = this.lastResult?.reference_skeleton;
    return !!seq?.some(f => !!f.face);
  }
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

  get parameterCards(): { name: string; match: boolean | null; why: string; detail: string; hint: boolean }[] {
    const params = this.lastResult?.parameters || {};
    const cards = ['handshape', 'location', 'orientation'].map(key => {
      const p = params[key];
      return {
        name: key === 'orientation' ? 'Palm direction' : key.charAt(0).toUpperCase() + key.slice(1),
        match: p ? p.match : null,
        // Plain-language explanation (see parameters.py's _describe_mismatch)
        // leads; the raw bucket values stay as a small secondary line for
        // anyone who wants the exact reading, not the headline anymore.
        why: (p && !p.match && p.why) || '',
        detail: p?.target_plain && p?.yours_plain ? `target: ${p.target_plain} · you: ${p.yours_plain}` : '',
        // a mismatch from a check that isn't reliable yet is a hint, not a mistake
        hint: !!p && p.match === false && p.reliable === false,
      };
    });
    cards.push({
      name: 'Hand count',
      match: this.lastResult?.hand_count_match ?? null,
      why: '',
      detail: '',
      hint: this.lastResult?.hand_count_match === false && this.lastResult?.check_reliability?.['hand_count']?.reliable === false,
    });
    return cards;
  }

  /** A short looping window of the attempt's OWN raw motion around the exact
   * frame that diverged most from the reference, per mismatched parameter --
   * replaces a single frozen snapshot with something you can actually watch
   * repeat (P1's "replay it" request). Raw space, not canonical, so it looks
   * like the signer's own hand, same as the "Your motion" panel above. */
  get replayWindows(): { param: string; frames: SkeletonFrame[] }[] {
    const seq = this.lastResult?.attempt_skeleton_raw;
    const divergent = this.lastResult?.divergent_frames;
    if (!seq || seq.length === 0 || !divergent) return [];
    const WINDOW = 8; // frames either side of the divergent moment
    return Object.entries(divergent).map(([param, idx]) => {
      const start = Math.max(0, idx - WINDOW);
      const end = Math.min(seq.length, idx + WINDOW + 1);
      return { param, frames: seq.slice(start, end) };
    }).filter(w => w.frames.length > 0);
  }

  /** Shows one frame of the reference clip as a still, by seeking a paused
   * <video> -- no canvas capture, so no cross-origin issues. */
  seekStill(event: Event, frac: number): void {
    const v = event.target as HTMLVideoElement;
    // Prefer the span where the hands are actually up in this video.
    const up = this.referenceBuckets?.hands_up_seconds;
    const start = up ? up.start : this.hasTrimWindow ? this.referenceTrimStart : 0;
    const end = up ? up.end : this.hasTrimWindow ? this.referenceTrimEnd! : (v.duration || 0);
    v.currentTime = start + frac * Math.max(0, end - start);
  }

  get verdictLabel(): string {
    if (this.lastResult?.graded === false) return 'not graded';
    const v = this.lastResult?.verdict;
    return v === 'good' ? 'looks right' : v === 'needs_work' ? 'close' : v ? 'not yet' : '';
  }

  prepIndex = 0;
  /** Signs with a reference recording (gradable); empty until loaded. */
  gradable = new Set<string>();
  hasReference = true;
  /** Start / middle / end stills, within the span the hands are up. */
  readonly phases: { label: string; phase: 'start' | 'middle' | 'end'; caption: string }[] = [
    { label: 'Start', phase: 'start', caption: 'The ring shows where the hand starts.' },
    { label: 'Middle', phase: 'middle', caption: 'The arrow shows which way it moves.' },
    { label: 'End', phase: 'end', caption: 'The arrowhead shows where it finishes.' },
  ];
  variations: { video_url: string; label?: string }[] = [];
  /** v3: null while the 3D skin loads, false if this sign has none (show the older 3D hand). */
  skinAvailable: boolean | null = null;
  get variantLabel(): string | null { return this.variations.find(v => v.label)?.label ?? null; }
  unavailableNotice = '';

  /** Each hand's path through the example video (for the direction marks). */
  get handPaths(): HandPath[] { return this.referenceBuckets?.hand_paths ?? []; }

  /** How long the signing itself lasts in the example video (seconds), so
   * the 3D hand loops at the clip's pace instead of a fixed, slower cycle. */
  get signSeconds(): number {
    const up = this.referenceBuckets?.hands_up_seconds;
    if (up && up.end > up.start) return up.end - up.start;
    const ts = this.handPaths.flatMap(p => [p.t[0], p.t[p.t.length - 1]]);
    return ts.length ? Math.max(...ts) - Math.min(...ts) : 2;
  }

  /** Every hand is made in one spot (e.g. who, no, color): describe the
   * place and the small movement instead of a path. */
  get signStaysInPlace(): boolean { return this.handPaths.length > 0 && this.handPaths.every(p => p.still); }

  get phaseCards(): { label: string; phase: 'start' | 'middle' | 'end'; caption: string }[] {
    if (!this.signStaysInPlace) return this.phases;
    return [
      { label: 'Start', phase: 'start', caption: 'The dashed ring shows where the hand stays.' },
      { label: 'Middle', phase: 'middle', caption: 'The movement happens in that spot: watch the clip above.' },
      { label: 'End', phase: 'end', caption: 'The hand finishes in the same place.' },
    ];
  }

  // ---- intro ("How practice works") ----
  private static readonly INTRO_KEY = 'mouna_hide_practice_intro';
  hideIntro = false;
  readonly introSteps: { title: string; body: string; art: SafeHtml }[] = [];

  finishIntro(): void {
    try { if (this.hideIntro) localStorage.setItem(CoachComponent.INTRO_KEY, '1'); else localStorage.removeItem(CoachComponent.INTRO_KEY); } catch { /* storage unavailable */ }
    this.screen = 'preview';
  }
  readonly apiBase = environment.apiBaseUrl;
  /** Target skeleton (with hand 3D) for the Learn screen's 3D view. */
  referenceSkeleton: SkeletonFrame[] | null = null;
  liveFeedback: { text: string; tone: 'good' | 'warn' | 'neutral' } | null = null;
  private liveLoopRunning = false;
  private destroyed = false;
  private static readonly LIVE_WINDOW_MS = 1500;
  private static readonly PASSES_TO_ADVANCE = 2;

  /** Only signs the coach can fully support; falls back to all signs until
   * the list loads (or if it fails), so the page is never empty. */
  get practiceSigns(): Sign[] {
    if (this.gradable.size === 0) return this.signs;
    return this.signs.filter(s => this.gradable.has(s.word));
  }

  isLiveCheckable(st: PrepStepView): boolean {
    return this.hasReference && st.cps.length > 0;
  }

  /** Live prep: checks the CURRENT step's CPs together on one short
   * window of frames (manual CPs via one grouped check; facial on its own).
   * Two passes in a row (~3s held) -> confirm in the chat and advance; a
   * miss shows the most important reason inline and keeps checking. */
  private async runLiveChecks(): Promise<void> {
    if (this.liveLoopRunning) return;
    this.liveLoopRunning = true;
    const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
    let passes = 0;
    let lastStep = '';
    try {
      while (!this.destroyed && this.screen === 'ready') {
        const st = this.currentPrep;
        if (st && st.id !== lastStep) { passes = 0; lastStep = st.id; this.liveFeedback = null; }
        if (!st || !this.isLiveCheckable(st) || this.analyzing) { await sleep(400); continue; }
        await sleep(CoachComponent.LIVE_WINDOW_MS);
        if (this.currentPrep?.id !== st.id || this.screen !== 'ready' || this.analyzing) continue;

        const manual = st.cps.filter(cp => cp !== 'facial');
        const results: Record<string, StepResult | null> = {};
        if (manual.length) {
          const r = await this.coachService.checkGroup(this.targetSignWord, manual);
          for (const cp of manual) results[cp] = r?.[cp] ? { ...r[cp], type: 'step_result', step: cp } as StepResult : null;
        }
        if (st.cps.includes('facial')) results['facial'] = await this.coachService.checkStep(this.targetSignWord, 'facial');
        if (this.currentPrep?.id !== st.id) continue;

        const comments = st.cps.map(cp => this.buildComment(cp, results[cp] ?? null));
        this.comments = [...this.comments.filter(x => !st.cps.includes(x.step)), ...comments];
        const firstIssue = comments.find(c => c.tone === 'warn') ?? comments.find(c => c.tone === 'neutral');
        if (!firstIssue) {
          passes++;
          const isLast = this.prepIndex >= this.prepSteps.length - 1;
          if (passes >= CoachComponent.PASSES_TO_ADVANCE && isLast) {
            this.liveFeedback = { text: "Looks right! Record when you're ready.", tone: 'good' };
          } else if (passes >= CoachComponent.PASSES_TO_ADVANCE) {
            this.liveFeedback = { text: 'Looks right! Moving to the next step…', tone: 'good' };
            this.pushChat({ role: 'coach', text: `✓ ${st.label}: looks right.` });
            await sleep(1200);
            if (this.currentPrep?.id === st.id) this.prepIndex = Math.min(this.prepIndex + 1, this.prepSteps.length - 1);
            passes = 0;
          } else {
            this.liveFeedback = { text: 'Looks right, hold it a moment…', tone: 'good' };
          }
        } else {
          passes = 0;
          const label = firstIssue.step === 'orientation' ? 'Palm direction' : firstIssue.label;
          this.liveFeedback = { text: st.cps.length > 1 ? `${label}: ${firstIssue.text}` : firstIssue.text, tone: firstIssue.tone };
        }
      }
    } finally {
      this.liveLoopRunning = false;
    }
  }

  /** Prep steps: one CP per step, in the order you'd set up a sign (shape,
   * place, palm, hands, face), then movement (shown, never scored -- it
   * failed validation twice), then a final Analyze step. Each step shows
   * stills taken from the reference clip itself; hand-related steps crop to
   * the reference hand. Check results go to the chat, not this panel. */
  get prepSteps(): PrepStepView[] {
    const howto = this.referenceBuckets?.howto ?? {};
    const desc = this.targetSignInfo?.description ?? '';
    const need = (st: PrepStep) => this.requiredSteps.includes(st);
    const tone = (cps: PrepStep[]): PrepStepView['tone'] => {
      const cs = cps.map(cp => this.comments.find(c => c.step === cp)?.tone);
      if (cs.some(t => t === 'warn')) return 'warn';
      return cps.length && cs.every(t => t === 'good') ? 'good' : 'neutral';
    };
    const steps: PrepStepView[] = [];
    const shapeCps = (['handshape', 'orientation'] as PrepStep[]).filter(need);
    if (shapeCps.length) {
      steps.push({
        id: 'shape', label: 'Hand shape & palm', cps: shapeCps, tone: tone(shapeCps),
        instruction: [howto.handshape, howto.orientation].filter(Boolean).join(' ') || 'Match the hand shape and palm direction in the picture.',
        shots: [{ label: 'Shape and palm', frac: 0.5, zoom: true }],
      });
    }
    const placeCps = (['location', 'hand_count'] as PrepStep[]).filter(need);
    if (placeCps.length) {
      steps.push({
        id: 'place', label: 'Where to sign', cps: placeCps, tone: tone(placeCps),
        instruction: `${this.handPaths.length === 2 ? 'Use both hands. ' : this.handPaths.length === 1 ? 'Use one hand. ' : ''}${this.signStaysInPlace
          ? `Hold your hand${this.handPaths.length === 2 ? 's' : ''} where the dashed ring${this.handPaths.length === 2 ? 's are' : ' is'}. The movement is small, so copy it from the clip.`
          : `Start where the ${this.handPaths.length === 2 ? 'rings are' : 'ring is'}, then follow the arrow${this.handPaths.length === 2 ? 's' : ''}.`}`,
        shots: [{ label: this.signStaysInPlace ? 'Dashed ring = where the hand stays' : 'Ring = start, arrow = direction', frac: 0.5, zoom: false }],
      });
    }
    const faceCps = (['facial'] as PrepStep[]).filter(need);
    steps.push({
      id: 'move', label: faceCps.length ? 'Face, then record' : 'Ready to record', cps: faceCps, tone: tone(faceCps), shots: [], clip: true,
      instruction: `${faceCps.length ? `Add a ${this.nmsShortLabel} while you sign. ` : ''}Do the motion with the clip a couple of times. When it feels right, press Record.`,
    });
    return steps;
  }

  get currentPrep(): PrepStepView | null {
    const steps = this.prepSteps;
    return steps[Math.min(this.prepIndex, steps.length - 1)] ?? null;
  }

  stepLabel(step: PrepStep): string {
    return STEP_LABELS[step] ?? step;
  }

  ngOnInit(): void {
    try { this.hideIntro = localStorage.getItem(CoachComponent.INTRO_KEY) === '1'; } catch { this.hideIntro = false; }
    this.screen = this.hideIntro ? 'preview' : 'intro';
    this.buildIntroArt();
    // Supports deep-linking a specific sign in, e.g. from the dashboard's
    // "due for review" list (?sign=apple) -- falls back to the first sign
    // in the list exactly as before when no query param is present.
    const requestedSign = this.route.snapshot.queryParamMap.get('sign');
    this.signsApi.getSignsWithReference().subscribe({
      next: set => {
        this.gradable = set;
        if (set.size && !set.has(this.targetSignWord)) {
          // A link to a sign outside the practice set (e.g. from older
          // attempts) has no reference, path or 3D hand to show.
          if (requestedSign) this.unavailableNotice = `"${requestedSign}" isn't in the practice set yet, so here's ${this.practiceSigns[0]?.word ?? 'another sign'} instead.`;
          const first = this.practiceSigns[0];
          if (first) { this.targetSignWord = first.word; this.onTargetChange(); }
        }
      },
      error: () => {},
    });
    this.signsApi.getSigns().subscribe(signs => {
      this.signs = signs;
      if (requestedSign && signs.some(s => s.word === requestedSign)) {
        this.targetSignWord = requestedSign;
      } else if (signs.length > 0) {
        this.targetSignWord = (this.practiceSigns[0] ?? signs[0]).word;
      }
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
    this.skinAvailable = null;  // try the 3D skin again for the new sign
    this.referenceBuckets = null;
    this.prepIndex = 0;
    this.liveFeedback = null;
    this.comments = [];
    this.signsApi.getReferenceBuckets(this.targetSignWord).subscribe({
      next: buckets => { this.referenceBuckets = buckets; this.hasReference = true; },
      error: () => { this.referenceBuckets = null; this.hasReference = false; },
    });
    this.variations = [];
    this.signsApi.getVariations(this.targetSignWord).subscribe({ next: v => { this.variations = v; }, error: () => {} });
    this.referenceSkeleton = null;
    this.signsApi.getReferenceSkeleton(this.targetSignWord).subscribe({
      next: frames => { this.referenceSkeleton = frames; },
      error: () => { this.referenceSkeleton = null; },
    });
  }

  setPlaybackRate(rate: number): void {
    this.playbackRate = rate;
    this.previewVideosEl?.nativeElement.querySelectorAll('video').forEach(v => { v.playbackRate = rate; });
  }

  /** Seeks to the trim window's start (if this clip has one -- see
   * loadReferenceVideo) and sets playback rate, once metadata is ready. */
  onVideoLoaded(event: Event): void {
    const video = event.target as HTMLVideoElement;
    video.playbackRate = this.playbackRate;
    if (this.hasTrimWindow) video.currentTime = this.referenceTrimStart;
    if (this.stepMode) video.pause();
  }

  /** Manual loop over JUST the sign's frame_start-frame_end window, not the
   * whole source file -- some WLASL clips are a few seconds of isolated
   * sign, others are a brief window inside a much longer raw recording
   * (e.g. one "what" instance sits at frames 1748-1780 of a 70+ second
   * clip). Playing the raw file start-to-finish showed a long stretch of
   * unrelated footage before ever reaching the sign -- this keeps playback
   * inside just the real sign, however long the underlying file is. */
  enforceTrim(event: Event): void {
    if (!this.hasTrimWindow) return;
    const video = event.target as HTMLVideoElement;
    if (video.currentTime >= this.referenceTrimEnd! || video.currentTime < this.referenceTrimStart) {
      video.currentTime = this.referenceTrimStart;
    }
  }

  get hasTrimWindow(): boolean {
    return this.referenceTrimEnd !== null;
  }

  toggleStepMode(): void {
    this.stepMode = !this.stepMode;
    const videos = this.previewVideosEl?.nativeElement.querySelectorAll('video') ?? [];
    videos.forEach(v => {
      if (this.stepMode) {
        v.pause();
      } else {
        v.play().catch(() => {});
      }
    });
  }

  stepFrame(direction: 1 | -1): void {
    const videos = this.previewVideosEl?.nativeElement.querySelectorAll('video') ?? [];
    const start = this.hasTrimWindow ? this.referenceTrimStart : 0;
    const end = this.hasTrimWindow ? this.referenceTrimEnd! : null;
    videos.forEach(v => {
      const span = end ?? (v.duration || 0);
      let next = v.currentTime + direction * this.STEP_SECONDS;
      if (span > start) next = start + (((next - start) % (span - start)) + (span - start)) % (span - start); // wrap within [start, span)
      v.currentTime = next;
    });
  }

  loadReferenceVideo(): void {
    this.signsApi.getVideo(this.targetSignWord).subscribe({
      next: res => {
        this.referenceVideoUrl = `${environment.apiBaseUrl}${res.video_url}`;
        const hasKnownEnd = res.frame_end !== null && res.frame_end !== -1 && !!res.fps;
        this.referenceTrimStart = hasKnownEnd ? (res.frame_start ?? 0) / res.fps! : 0;
        this.referenceTrimEnd = hasKnownEnd ? res.frame_end! / res.fps! : null;
      },
      error: () => {
        this.referenceVideoUrl = null;
        this.referenceTrimStart = 0;
        this.referenceTrimEnd = null;
      },
    });
  }

  goToReady(): void {
    this.screen = 'ready';
    this.prepIndex = 0;
    this.liveFeedback = null;
    setTimeout(() => this.runLiveChecks(), 0);
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
    this.analyzeLines = [];
    this.analyzeHead = '';
    for (const step of this.requiredSteps) {
      await new Promise(r => setTimeout(r, CHECK_HOLD_MS));
      const result = await this.coachService.checkStep(this.targetSignWord, step);
      if (step === 'facial') this.hasAnalyzedFacial = true;
      this.comments.push(this.buildComment(step, result));
    }
    this.analyzing = false;
    this.postPrepResultsToChat();
  }

  /** Latest full Analyze result, shown inline on the final prep step (the
   * chat panel is hidden during prep) and kept in the chat history. */
  analyzeHead = '';
  analyzeLines: { text: string; tone: 'good' | 'warn' | 'neutral' }[] = [];

  private sanitizer = inject(DomSanitizer);

  /** Simple line drawings for the three intro steps (brand colors). */
  private buildIntroArt(): void {
    const ink = 'rgb(var(--color-ink-muted))', coral = '#F48C77', teal = '#75B4B4', accent = 'rgb(var(--color-accent))';
    const art = [
      // 1. a clip with a dotted path and arrow
      `<svg width="150" height="90" viewBox="0 0 150 90" fill="none"><rect x="10" y="8" width="130" height="74" rx="10" stroke="${ink}" stroke-width="2"/><path d="M40 62 C 55 30, 85 24, 108 40" stroke="${coral}" stroke-width="3" stroke-dasharray="5 5" stroke-linecap="round"/><path d="M104 32 L112 42 L100 45 Z" fill="${coral}"/><circle cx="40" cy="62" r="6" stroke="${coral}" stroke-width="3"/></svg>`,
      // 2. a camera frame with a check
      `<svg width="150" height="90" viewBox="0 0 150 90" fill="none"><path d="M22 20 v-8 h12 M128 20 v-8 h-12 M22 70 v8 h12 M128 70 v8 h-12" stroke="${ink}" stroke-width="2.5" stroke-linecap="round"/><circle cx="75" cy="38" r="11" stroke="${ink}" stroke-width="2"/><path d="M55 72 c2-14 38-14 40 0" stroke="${ink}" stroke-width="2"/><circle cx="108" cy="30" r="11" fill="${teal}"/><path d="M103 30 l4 4 l7-8" stroke="white" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
      // 3. record + countdown
      `<svg width="150" height="90" viewBox="0 0 150 90" fill="none"><circle cx="45" cy="45" r="20" stroke="${ink}" stroke-width="2"/><circle cx="45" cy="45" r="11" fill="#ef4444"/><text x="80" y="54" font-family="Georgia, serif" font-size="26" fill="${accent}">3·2·1</text></svg>`,
    ];
    const text = [
      { title: 'Learn the sign', body: 'Watch the sign. Arrows show where the hand starts, which way it moves, and where it ends. No camera yet.' },
      { title: 'Get set', body: 'The camera turns on. Copy each position and hold it. Mouna checks it and moves on by itself.' },
      { title: 'Your turn', body: 'Press Record. After 3-2-1, sign once. You get a short note on what to keep and what to fix.' },
    ];
    (this as { introSteps: { title: string; body: string; art: SafeHtml }[] }).introSteps.splice(0, 3,
      ...text.map((t, i) => ({ ...t, art: this.sanitizer.bypassSecurityTrustHtml(art[i]) })));
  }

  stepIndex(id: Screen): number { return this.screenSteps.findIndex(s => s.id === id); }

  get showChat(): boolean { return this.screen === 'record'; }

  private postPrepResultsToChat(): void {
    if (this.comments.length === 0) return;
    const lines = this.comments.map(c => ({ tone: c.tone, text: `${c.tone === 'good' ? '✓' : c.tone === 'warn' ? '!' : '·'} ${c.step === 'orientation' ? 'Palm direction' : c.label}: ${c.text}` }));
    const fixes = this.comments.filter(c => c.tone === 'warn').length;
    const head = fixes === 0 ? `Prep check for "${this.targetSignWord}": looking good.` : `Prep check for "${this.targetSignWord}": ${fixes} thing${fixes > 1 ? 's' : ''} to adjust.`;
    this.analyzeHead = head;
    this.analyzeLines = lines;
    this.pushChat({ role: 'coach', text: [head, ...lines.map(l => l.text)].join('\n') });
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
    if (result.uncertain) {
      // Signers vary on this part for this sign: never block prep on it.
      return { step, label, text: 'Signers vary on this part, so any close version is fine.', tone: 'good' };
    }
    if (result.match === null || result.match === undefined) {
      return { step, label, text: "Couldn't get a clear read — try adjusting your position.", tone: 'neutral' };
    }
    if (result.match) return { step, label, text: 'Looks right.', tone: 'good' };
    return { step, label, text: result.why || "Doesn't match the target yet.", tone: 'warn' };
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

  /** Record screen starts the 3-2-1 countdown by itself -- the Record
   * button on prep was the decision; no second click needed. */
  goToRecord(): void {
    this.screen = 'record';
    this.recordState = 'ready';
    setTimeout(() => {
      this.cameraViewRecord?.start();
      this.onRecordClick();
    }, 0);
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
    this.coachService.resetBuffer();
    this.startAttemptRecording();
    this.recordingTimer = setInterval(() => {
      const elapsed = Date.now() - this.recordingStartTime;
      this.recordingProgress = Math.min((elapsed / RECORDING_DURATION_MS) * 100, 100);
      if (elapsed >= RECORDING_DURATION_MS) {
        if (this.recordingTimer) clearInterval(this.recordingTimer);
        this.stopAttemptRecording();
        this.coachService.requestPrediction(this.targetSignWord);
      }
    }, 100);
  }

  /** Records the ACTUAL camera video client-side (MediaRecorder on the same
   * stream the frame-capture pipeline already reads from -- independent
   * consumers of one stream, nothing about keypoint extraction changes).
   * This is the real-human-next-to-real-human example: the reference video
   * is already a real fluent signer; this is the other real person in the
   * comparison, instead of a skeleton standing in for you. */
  private startAttemptRecording(): void {
    const stream = this.cameraViewRecord?.getStream();
    if (!stream) return;
    this.recordedChunks = [];
    try {
      this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    } catch {
      this.mediaRecorder = null; // unsupported mime/browser -- feedback screen just won't have this panel
      return;
    }
    this.mediaRecorder.ondataavailable = e => { if (e.data.size > 0) this.recordedChunks.push(e.data); };
    this.mediaRecorder.start();
  }

  private stopAttemptRecording(): void {
    if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') return;
    this.mediaRecorder.onstop = () => {
      if (this.recordedAttemptUrl) URL.revokeObjectURL(this.recordedAttemptUrl);
      this.recordedAttemptUrl = URL.createObjectURL(new Blob(this.recordedChunks, { type: 'video/webm' }));
    };
    this.mediaRecorder.stop();
  }

  /** Try again: the camera view was destroyed while results showed, so
   * restart it, then run the countdown like the first attempt. */
  retryRecording(): void {
    this.recordState = 'ready';
    this.lastResult = null;
    setTimeout(() => {
      this.cameraViewRecord?.start();
      this.onRecordClick();
    }, 0);
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
      if (msg.prediction && msg.prediction !== this.targetSignWord) {
        this.pushChat({ role: 'system', text: `Recognizer's own guess: "${msg.prediction}" (${Math.round((msg.confidence || 0) * 100)}%), a second opinion, not your grade` });
      }
    } else if (msg.type === 'coaching_update') {
      // The AI's wording of the result already shown (decided locally).
      // Swap it into that attempt's message; ignore it if a newer attempt
      // has been recorded since.
      if (!this.lastResult || msg.attempt_number !== this.lastResult.attempt_number) return;
      this.lastResult = { ...this.lastResult, feedback: msg.feedback, feedback_source: msg.feedback_source };
      const i = this.chatMessages.map(m => m.coachingData?.attempt_number).lastIndexOf(msg.attempt_number);
      if (i >= 0) this.chatMessages[i] = { ...this.chatMessages[i], text: msg.feedback || this.chatMessages[i].text, coachingData: this.lastResult };
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
    if (id === 'ready') { setTimeout(() => this.cameraViewReady?.start(), 0); setTimeout(() => this.runLiveChecks(), 0); }
    if (id === 'record') setTimeout(() => this.cameraViewRecord?.start(), 0);
  }

  ngOnDestroy(): void {
    this.destroyed = true;
    if (this.countdownTimer) clearInterval(this.countdownTimer);
    if (this.recordingTimer) clearInterval(this.recordingTimer);
    if (this.recordedAttemptUrl) URL.revokeObjectURL(this.recordedAttemptUrl);
    if (this.sessionStarted) this.coachService.endSession();
    this.subs.forEach(s => s.unsubscribe());
  }
}
