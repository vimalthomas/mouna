import { Component, OnDestroy, OnInit, ViewChild, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { environment } from '../../../environments/environment';
import { AuthService } from '../../core/services/auth.service';
import { PracticeApiService, PracticePool, PracticeSign, SentenceClip, SignScore, Story } from '../../core/services/practice-api.service';
import { CameraViewComponent } from '../../shared/components/camera-view/camera-view.component';
import { SignSkinComponent } from '../../shared/components/sign-skin/sign-skin.component';

type Mode = 'stories' | 'review';
const LEVEL: Record<string, string> = { fluent: 'Fluent', close: 'Close', needs_work: 'Needs work' };

/**
 * v5 Practice: short stories made only from the signs this learner has learned
 * (shown in English and in ASL order), the avatar signing the whole sentence,
 * and camera practice sign by sign -- each sign graded with the same scoring as
 * Learn, the sentence saved for the dashboard. Under the unlock threshold, or
 * in "Review", it lists the learned signs to revisit in Learn.
 */
@Component({
  selector: 'app-practice',
  standalone: true,
  imports: [RouterLink, CameraViewComponent, SignSkinComponent],
  template: `
    <div class="max-w-6xl mx-auto">
      <div class="flex items-end justify-between gap-4 flex-wrap mb-6">
        <div>
          <h1 class="font-display text-3xl text-ink">Practice</h1>
          <p class="text-sm text-ink-muted mt-1">Short stories made only from the signs you've learned. Watch them, then sign them yourself.</p>
        </div>
        <div class="flex items-center gap-3">
          <div class="flex items-center gap-1 rounded-full bg-surface-sunk border border-border/50 p-1" role="tablist">
            @for (m of modes; track m.id) {
              <button role="tab" [attr.aria-selected]="mode === m.id" (click)="mode = m.id"
                class="text-xs font-semibold px-3.5 py-1.5 rounded-full transition-colors"
                [class]="mode === m.id ? 'bg-accent text-white' : 'text-ink-muted hover:text-ink'">{{ m.label }}</button>
            }
          </div>
          @if (mode === 'stories' && pool?.unlocked) {
            <button (click)="newStory()" [disabled]="loadingStory || practising"
              class="px-4 h-9 rounded-xl bg-accent text-white text-sm font-medium disabled:opacity-50">Another story</button>
          }
        </div>
      </div>

      @if (!pool) {
        <p class="text-sm text-ink-muted">Loading your signs…</p>
      } @else if (mode === 'review' || !pool.unlocked) {
        @if (!pool.unlocked) {
          <div class="rounded-xl border border-border/60 bg-surface px-5 py-4 mb-6">
            <p class="text-ink font-medium">Learn {{ pool.min_signs - pool.learned.length }} more sign{{ pool.min_signs - pool.learned.length === 1 ? '' : 's' }} to unlock stories</p>
            <div class="h-2 rounded-full bg-surface-sunk overflow-hidden mt-3 max-w-md">
              <div class="h-full bg-accent rounded-full" [style.width.%]="100 * pool.learned.length / pool.min_signs"></div>
            </div>
            <p class="text-sm text-ink-muted mt-2">{{ pool.learned.length }} of {{ pool.min_signs }} learned. Stories use only signs you've learned, so you can sign every word.
              <a routerLink="/learn" class="text-accent">Learn a sign →</a></p>
          </div>
        }
        <h2 class="text-xs font-semibold uppercase tracking-wider text-ink-faint mb-3">Learned ({{ pool.learned.length }})</h2>
        @if (pool.learned.length) {
          <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3 mb-6">
            @for (s of pool.learned; track s.word) {
              <a [routerLink]="['/learn']" [queryParams]="{ sign: s.word }" class="rounded-xl border border-border/60 bg-surface overflow-hidden hover:border-accent/60 transition-colors">
                @if (s.video_url) {
                  <video [src]="apiBase + s.video_url + '#t=0.9'" muted playsinline preload="metadata" class="w-full aspect-[4/3] object-cover bg-black"
                    [style.transform]="lefty ? 'scaleX(-1)' : null" (mouseenter)="$any($event.target).play()" (mouseleave)="$any($event.target).pause()"></video>
                }
                <div class="px-3 py-2"><p class="text-sm font-semibold text-ink capitalize">{{ s.word }}</p><p class="text-[11px] text-ink-muted">Review in Learn →</p></div>
              </a>
            }
          </div>
        } @else {
          <p class="text-sm text-ink-muted mb-6">Nothing learned yet. A sign counts as learned after enough good tries in Learn.</p>
        }
        @if (pool.practising.length) {
          <h2 class="text-xs font-semibold uppercase tracking-wider text-ink-faint mb-3">Still practising ({{ pool.practising.length }})</h2>
          <div class="flex flex-wrap gap-2">
            @for (s of pool.practising; track s.word) {
              <a [routerLink]="['/learn']" [queryParams]="{ sign: s.word }" class="px-3 py-1.5 rounded-full border border-border/60 text-sm text-ink capitalize hover:border-accent/60">{{ s.word }}</a>
            }
          </div>
        }
      } @else {
        <!-- the story -->
        <section class="rounded-xl border border-border/60 bg-surface px-5 py-4 mb-4">
          @if (loadingStory) {
            <p class="text-sm text-ink-muted">Writing a story from your {{ pool.learned.length }} learned signs…</p>
          } @else if (error) {
            <p class="text-sm text-rose-500">{{ error }}</p>
          } @else if (story) {
            <div class="flex items-baseline justify-between gap-3">
              <span class="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">{{ story.title }}</span>
              <label class="flex items-center gap-1.5 text-xs text-ink-muted cursor-pointer select-none" title="Include one sign you're still practising">
                <input type="checkbox" [checked]="stretch" (change)="setStretch($any($event.target).checked)"> Add a sign I'm still practising
              </label>
            </div>
            @for (s of story.sentences; track $index) { <p class="font-display text-2xl text-ink mt-2 leading-snug">{{ s.english }}</p> }
            @if (story.stretch) { <p class="text-xs text-amber-500 mt-2">Includes <strong class="uppercase">{{ story.stretch }}</strong>, a sign you're still practising.</p> }
          }
        </section>

        @if (story && !loadingStory) {
          <!-- ASL structure -->
          <section class="rounded-xl border border-border/60 bg-surface px-5 py-4 mb-4">
            <p class="text-[11px] font-semibold uppercase tracking-wider text-ink-faint mb-2">ASL structure</p>
            @for (s of story.sentences; track $index) {
              <div class="flex flex-wrap items-center gap-1.5 mt-2">
                @for (g of s.gloss; track $index) { <span class="font-mono text-xs font-semibold tracking-wide px-2.5 py-1.5 rounded-lg bg-surface-sunk border border-border/60 text-ink">{{ g }}</span> }
                <span class="text-sm text-ink-muted ml-2">“{{ s.english }}”</span>
              </div>
            }
            @if (story.structure.length) {
              <ul class="list-disc pl-5 mt-3 text-sm text-ink-muted space-y-0.5">@for (n of story.structure; track $index) { <li>{{ n }}</li> }</ul>
            }
          </section>

          <!-- the signs, in order -->
          <p class="text-[11px] font-semibold uppercase tracking-wider text-ink-faint mb-2">The signs, in order <span class="normal-case tracking-normal font-normal">· hover to play</span></p>
          <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-5">
            @for (s of story.signs; track $index; let i = $index) {
              <div class="rounded-xl border bg-surface overflow-hidden transition-shadow" [class]="current === i ? 'border-accent ring-2 ring-accent' : resultClass(i)">
                @if (s.video_url) {
                  <video [src]="apiBase + s.video_url + '#t=0.9'" muted loop playsinline preload="metadata" class="w-full aspect-[4/3] object-cover bg-black"
                    [style.transform]="lefty ? 'scaleX(-1)' : null" (mouseenter)="$any($event.target).play()" (mouseleave)="$any($event.target).pause()"></video>
                }
                <div class="px-3 py-2">
                  <p class="text-sm font-semibold text-ink">{{ i + 1 }}. <span class="uppercase tracking-wide">{{ s.word }}</span>
                    @if (!s.learned) { <span class="ml-1 text-[10px] font-bold uppercase text-amber-500">practising</span> }</p>
                  <p class="text-[11px] text-ink-muted leading-snug">{{ s.description }}</p>
                </div>
              </div>
            }
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <!-- the avatar signs it -->
            <section class="rounded-xl border border-border/60 bg-surface p-4">
              <div class="flex items-center justify-between mb-2">
                <p class="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">Watch it signed</p>
                <span class="text-xs text-ink-faint">{{ clip ? (clip.n_frames / clip.fps).toFixed(1) + ' s' : 'loading…' }}</span>
              </div>
              @if (clip) {
                <app-sign-skin #skin [data]="clip" show="all" label="The story, signed" [speed]="speed" (position)="onPosition($event)" />
              } @else {
                <div class="aspect-[4/3] rounded-xl bg-surface-sunk"></div>
              }
              <div class="flex flex-wrap items-center gap-1.5 mt-3">
                @for (s of story.signs; track $index; let i = $index) {
                  <span class="font-mono text-xs font-semibold tracking-wide px-2.5 py-1.5 rounded-lg border"
                    [class]="playing === i ? 'bg-accent text-white border-accent' : 'bg-surface-sunk text-ink border-border/60'">{{ s.word.toUpperCase() }}</span>
                }
              </div>
              <div class="flex items-center gap-2 mt-3">
                <button (click)="skin?.restart()" class="text-xs font-semibold px-3 py-1 rounded-full border border-border/50 bg-surface-sunk text-ink">↺ From the start</button>
                @for (r of [0.5, 0.75, 1]; track r) {
                  <button (click)="speed = r" class="text-xs font-semibold px-3 py-1 rounded-full border"
                    [class]="speed === r ? 'bg-accent text-white border-accent' : 'bg-surface-sunk text-ink-muted border-border/50'">{{ r }}×</button>
                }
              </div>
              <p class="text-[11px] text-ink-faint mt-2">Each sign comes from a different fluent signer, shown on one avatar and blended together. Real signing joins signs more smoothly.</p>
            </section>

            <!-- practise it -->
            <section class="rounded-xl border border-border/60 bg-surface p-4">
              <p class="text-[11px] font-semibold uppercase tracking-wider text-ink-faint mb-2">Sign it yourself</p>
              <div class="relative rounded-xl overflow-hidden">
                <app-camera-view #cam [mirrorView]="true" [autoStart]="false" (frameCaptured)="onFrame($event)" />
                <div class="absolute inset-0 flex flex-col items-center justify-center text-white pointer-events-none text-center" style="text-shadow: 0 2px 8px rgba(0,0,0,.7)">
                  @if (overlay.kind === 'count') {
                    <p class="font-mono text-xl font-semibold tracking-wide">Next: {{ overlay.sign.toUpperCase() }}</p>
                    <p class="font-display text-6xl leading-none my-1">{{ overlay.n }}</p>
                    <p class="text-sm">{{ overlay.i + 1 }} of {{ story.signs.length }}</p>
                  } @else if (overlay.kind === 'rec') {
                    <span class="absolute top-3 left-3 bg-rose-600 text-white text-xs font-bold px-2 py-0.5 rounded-md">● REC</span>
                    <p class="font-mono text-2xl font-semibold tracking-wide">{{ overlay.sign.toUpperCase() }}</p>
                  } @else if (overlay.kind === 'check') {
                    <p class="text-lg">Checking {{ overlay.sign.toUpperCase() }}…</p>
                  } @else if (!practising) {
                    <p class="text-sm text-white/70 px-6">Your camera turns on when you press Start. Each sign is checked as you go.</p>
                  }
                </div>
              </div>
              <div class="flex items-center gap-3 mt-3 flex-wrap">
                @if (!practising) {
                  <button (click)="practise()" class="px-5 h-10 rounded-xl bg-accent text-white text-sm font-medium">{{ results.length ? 'Try the sentence again' : 'Start' }}</button>
                } @else {
                  <button (click)="stop()" class="px-4 h-10 rounded-xl border border-border/60 text-ink text-sm">Stop</button>
                }
                <span class="text-xs text-ink-muted">3 seconds to get ready for each sign, then 2.5 seconds to sign it.</span>
              </div>
              @if (results.length) {
                <div class="mt-4 divide-y divide-border/50 rounded-xl border border-border/60">
                  @for (r of results; track $index) {
                    <div class="px-4 py-2.5">
                      <div class="flex items-center gap-3">
                        <span class="font-mono text-xs font-semibold px-2 py-1 rounded-md border uppercase"
                          [class]="r.verdict === 'good' ? 'border-emerald-500/50 text-emerald-500' : 'border-amber-500/50 text-amber-500'">{{ r.sign }}</span>
                        <span class="text-sm" [class]="r.verdict === 'good' ? 'text-emerald-500' : 'text-amber-500'">{{ verdictText(r) }}</span>
                      </div>
                      @if (r.verdict !== 'good') { <p class="text-xs text-ink-muted mt-1">{{ noteFor(r) }}</p> }
                      @if (partsText(r)) { <p class="text-[11px] text-ink-faint mt-0.5">{{ partsText(r) }}</p> }
                    </div>
                  }
                </div>
                @if (results.length === story.signs.length) {
                  <p class="text-base text-ink mt-3">{{ goodCount === story.signs.length ? 'Whole sentence signed well.' : goodCount + ' of ' + story.signs.length + ' signs good. Try the others again, or the whole sentence.' }}
                    @if (saved) { <span class="text-xs text-ink-faint ml-1">Saved to your dashboard.</span> }</p>
                }
              }
            </section>
          </div>
        }
      }
    </div>
  `,
})
export class PracticeComponent implements OnInit, OnDestroy {
  @ViewChild('cam') cam?: CameraViewComponent;
  @ViewChild('skin') skin?: SignSkinComponent;
  private api = inject(PracticeApiService);
  private auth = inject(AuthService);
  readonly apiBase = environment.apiBaseUrl;
  readonly modes: { id: Mode; label: string }[] = [{ id: 'stories', label: 'Stories' }, { id: 'review', label: 'Review signs' }];

  mode: Mode = 'stories';
  pool: PracticePool | null = null;
  story: Story | null = null;
  clip: SentenceClip | null = null;
  loadingStory = false;
  error = '';
  stretch = false;
  speed = 1;
  playing = -1;           // sign the avatar is signing
  current = -1;           // sign being practised
  lefty = false;
  practising = false;
  results: SignScore[] = [];
  saved = false;
  overlay: { kind: 'none' | 'count' | 'rec' | 'check'; sign: string; n: number; i: number } = { kind: 'none', sign: '', n: 0, i: 0 };
  private recording = false;
  private frames: string[] = [];
  private subs: Subscription[] = [];

  ngOnInit(): void {
    try { this.stretch = localStorage.getItem('mouna_story_stretch') === '1'; } catch { /* private mode */ }
    this.subs.push(this.auth.user$.subscribe(u => (this.lefty = u?.dominant_hand === 'left')));
    this.api.pool().subscribe({
      next: p => { this.pool = p; if (p.unlocked) this.newStory(); },
      error: () => (this.error = "Couldn't load your signs."),
    });
  }

  ngOnDestroy(): void {
    this.stop();
    this.subs.forEach(s => s.unsubscribe());
  }

  setStretch(on: boolean): void {
    this.stretch = on;
    try { localStorage.setItem('mouna_story_stretch', on ? '1' : '0'); } catch { /* private mode */ }
  }

  newStory(): void {
    this.stop();
    this.loadingStory = true; this.error = ''; this.story = null; this.clip = null; this.results = []; this.saved = false;
    this.api.story(this.stretch).subscribe({
      next: s => {
        this.story = s; this.loadingStory = false;
        this.api.sentence(s.signs.map(x => x.word)).subscribe(c => (this.clip = c));
      },
      error: e => { this.loadingStory = false; this.error = e?.error?.detail || "Couldn't write a story right now. Try again."; },
    });
  }

  onPosition(frame: number): void {
    const segs = this.clip?.segments ?? [];
    this.playing = segs.findIndex(s => frame >= s.start - 4 && frame <= s.end + 4);
  }

  onFrame(b64: string): void {
    if (this.recording && this.frames.length < 60) this.frames.push(b64);
  }

  async practise(): Promise<void> {
    if (!this.story || this.practising) return;
    this.practising = true; this.results = []; this.saved = false;
    this.cam?.start();
    const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
    await sleep(1200);   // camera warm-up
    const signs = this.story.signs;
    for (let i = 0; i < signs.length && this.practising; i++) {
      const w = signs[i].word;
      this.current = i;
      for (let n = 3; n > 0 && this.practising; n--) { this.overlay = { kind: 'count', sign: w, n, i }; await sleep(1000); }
      if (!this.practising) break;
      this.overlay = { kind: 'rec', sign: w, n: 0, i };
      this.frames = []; this.recording = true;
      await sleep(2500);
      this.recording = false;
      if (!this.practising) break;
      this.overlay = { kind: 'check', sign: w, n: 0, i };
      const res = await new Promise<SignScore>(resolve => this.api.score(w, this.frames).subscribe({
        next: resolve,
        error: () => resolve({ sign: w, verdict: null, similarity: null, graded: false, hands_seen: 0, frames: this.frames.length, feedback: '', parts: {} }),
      }));
      this.results = [...this.results, res];
    }
    if (this.practising && this.results.length === signs.length) {
      const english = this.story.sentences.map(s => s.english).join(' ');
      this.api.saveAttempt(this.story.id, english, this.results.map(r => ({ sign: r.sign, verdict: r.verdict }))).subscribe(() => (this.saved = true));
    }
    this.stop();
  }

  stop(): void {
    this.practising = false; this.recording = false; this.current = -1;
    this.overlay = { kind: 'none', sign: '', n: 0, i: 0 };
    this.cam?.stop();
  }

  get goodCount(): number { return this.results.filter(r => r.verdict === 'good').length; }

  resultClass(i: number): string {
    const r = this.results[i];
    return !r ? 'border-border/60' : r.verdict === 'good' ? 'border-emerald-500/60' : 'border-amber-500/60';
  }

  verdictText(r: SignScore): string {
    if (r.verdict === 'good') return 'Good';
    if (!r.graded && r.verdict === null) return "Couldn't check this one";
    return r.verdict === 'wrong_sign' ? 'Looked like a different sign' : 'Needs work';
  }

  noteFor(r: SignScore): string {
    if (r.hands_seen < 3) return 'Your hands were hard to see. Keep them in the frame, in good light.';
    return (r.feedback || '').split('\n')[0];
  }

  partsText(r: SignScore): string {
    return Object.values(r.parts || {}).filter(p => p.level && p.level !== 'fluent').map(p => `${p.label}: ${LEVEL[p.level!]}`).join(' · ');
  }
}
