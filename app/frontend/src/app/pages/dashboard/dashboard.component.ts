import { PracticeApiService, PracticeSummary } from '../../core/services/practice-api.service';
import { Component, inject, OnInit } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { RouterLink } from '@angular/router';
import { UserProgressService } from '../../core/services/user-progress.service';
import { AuthService } from '../../core/services/auth.service';
import { SignsApiService } from '../../core/services/signs-api.service';
import { ReviewDueSign, Sign, SignProgressDetail, UserStats } from '../../core/models/sign.model';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [RouterLink, DecimalPipe],
  template: `
    <div class="max-w-5xl mx-auto pt-2">
      <!-- Greeting -->
      <header class="mb-8">
        <h1 class="font-display text-4xl text-ink">{{ greeting }}{{ firstName ? ', ' + firstName : '' }}</h1>
        <p class="text-ink-muted mt-2">{{ subline }}</p>
      </header>

      <!-- Numbers -->
      <section class="grid grid-cols-2 md:grid-cols-4 rounded-2xl bg-surface border border-border/70 mb-6 transition-opacity" [class.opacity-60]="loading">
        @for (s of stats; track s.label; let i = $index) {
          <div class="px-6 py-5" [class.md:border-l]="i > 0" [class.border-l]="i % 2 === 1" [class.border-t]="i > 1" [class.md:border-t-0]="i > 1" style="border-color: rgb(var(--color-border) / .7)">
            <div class="font-display text-[40px] leading-none text-ink tabular-nums">{{ s.value }}</div>
            <div class="text-sm text-ink mt-2">{{ s.label }}</div>
            <div class="text-xs text-ink-faint mt-0.5">{{ s.note }}</div>
          </div>
        }
      </section>

      <!-- Practice next -->
      @if (nextSign) {
        <a [routerLink]="['/learn']" [queryParams]="{ sign: nextSign.word }"
           class="group flex items-center justify-between gap-6 rounded-2xl bg-surface border border-border/70 px-6 py-5 mb-8 hover:border-accent/50 transition-colors">
          <div class="min-w-0">
            <div class="text-xs text-ink-faint">{{ nextReason }}</div>
            <div class="font-display text-3xl text-ink capitalize mt-1">{{ nextSign.word }}</div>
            <div class="text-sm text-ink-muted mt-1 truncate">{{ nextSign.description }}</div>
          </div>
          <span class="shrink-0 inline-flex items-center gap-2 h-11 px-5 rounded-xl bg-accent text-white text-sm font-medium group-hover:bg-accent/90 transition-colors">
            Learn
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M5 12h14m-6-6 6 6-6 6" /></svg>
          </span>
        </a>
      }

      <div class="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_300px] gap-6">
        <!-- Your signs -->
        <section>
          <div class="flex items-baseline justify-between mb-3 gap-3 flex-wrap">
            <div class="flex items-center gap-4">
              <h2 class="font-display text-2xl text-ink">Your progress</h2>
              <div class="flex items-center gap-1 rounded-full bg-surface-sunk border border-border/50 p-1" role="tablist">
                @for (t of tabs; track t.id) {
                  <button role="tab" [attr.aria-selected]="tab === t.id" (click)="tab = t.id"
                    class="text-xs font-semibold px-3 py-1 rounded-full transition-colors"
                    [class]="tab === t.id ? 'bg-accent text-white' : 'text-ink-muted hover:text-ink'">{{ t.label }}</button>
                }
              </div>
            </div>
            @if (tab === 'words') {
            <span class="text-xs text-ink-faint flex items-center gap-3">
              <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-emerald-600 dark:bg-emerald-400"></span>right</span>
              <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full border border-ink-faint"></span>missed</span>
              <span>last 5 tries</span>
            </span>
            }
          </div>
          @if (tab === 'sentences') {
            @if (practice?.sentences?.length) {
              <div class="rounded-2xl bg-surface border border-border/70 divide-y divide-border/60">
                @for (x of practice!.sentences; track x.id) {
                  <div class="px-5 py-3 flex items-start gap-4">
                    <div class="min-w-0 flex-1">
                      <p class="text-[15px] text-ink">{{ x.english }}</p>
                      <p class="font-mono text-[11px] tracking-wide text-ink-faint mt-0.5">{{ x.gloss }}</p>
                    </div>
                    <span class="text-xs shrink-0 tabular-nums" [class]="x.all_good ? 'text-emerald-700 dark:text-emerald-400' : 'text-ink-faint'">
                      {{ x.all_good ? 'Signed well' : x.signs_good + ' of ' + x.signs_total + ' signs good' }}</span>
                    <span class="text-xs text-ink-faint shrink-0 w-14 text-right">{{ ago(x.at) }}</span>
                  </div>
                }
              </div>
            } @else {
              <div class="rounded-2xl border border-dashed border-border px-6 py-10 text-center">
                <p class="text-ink">No sentences yet.</p>
                <p class="text-sm text-ink-muted mt-1">In <a routerLink="/practice" class="text-accent">Practice</a>, the signs you've learned become short stories to sign.</p>
              </div>
            }
          } @else if (shownSigns.length) {
            <div class="rounded-2xl bg-surface border border-border/70 divide-y divide-border/60">
              @for (s of shownSigns; track s.sign) {
                <a [routerLink]="['/learn']" [queryParams]="{ sign: s.sign }" class="flex items-center gap-4 px-5 py-3 hover:bg-surface-sunk/60 transition-colors first:rounded-t-2xl last:rounded-b-2xl">
                  <span class="text-[15px] text-ink capitalize w-32 truncate">{{ s.sign }}</span>
                  <span class="flex items-center gap-1.5" [attr.aria-label]="dotsLabel(s)">
                    @for (r of fiveSlots(s.recent); track $index) {
                      @if (r === null) {
                        <span class="w-2.5 h-2.5 rounded-full bg-border/60"></span>
                      } @else if (r) {
                        <span class="w-2.5 h-2.5 rounded-full bg-emerald-600 dark:bg-emerald-400"></span>
                      } @else {
                        <span class="w-2.5 h-2.5 rounded-full border-[1.5px] border-ink-faint"></span>
                      }
                    }
                  </span>
                  <span class="flex-1"></span>
                  @if (s.completed) {
                    <span class="text-xs text-emerald-700 dark:text-emerald-400 flex items-center gap-1">
                      <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.4"><path stroke-linecap="round" stroke-linejoin="round" d="m5 12.5 4.5 4.5L19 7.5" /></svg>
                      Learned
                    </span>
                  } @else {
                    <span class="text-xs text-ink-faint">{{ toLearn(s) }}</span>
                  }
                  <span class="text-xs text-ink-faint tabular-nums w-16 text-right">{{ s.correctAttempts }} of {{ s.totalAttempts }}</span>
                  <span class="text-xs tabular-nums w-24 text-right" [class]="inSentences(s.sign) ? 'text-ink-muted' : 'text-ink-faint/60'"
                    [title]="'Signed well inside practice sentences'">{{ inSentences(s.sign) || '–' }}</span>
                </a>
              }
            </div>
          } @else {
            <div class="rounded-2xl border border-dashed border-border px-6 py-10 text-center">
              <p class="text-ink">Nothing here yet.</p>
              <p class="text-sm text-ink-muted mt-1">Practice a sign and your progress will show up here.</p>
            </div>
          }
        </section>

        <!-- Review -->
        <aside>
          <h2 class="font-display text-2xl text-ink mb-3">Come back to</h2>
          @if (shownReview.length) {
            <div class="rounded-2xl bg-surface border border-border/70 divide-y divide-border/60">
              @for (r of shownReview; track r.sign) {
                <a [routerLink]="['/learn']" [queryParams]="{ sign: r.sign }" class="flex items-center justify-between px-5 py-3 hover:bg-surface-sunk/60 transition-colors first:rounded-t-2xl last:rounded-b-2xl">
                  <span class="text-[15px] text-ink capitalize">{{ r.sign }}</span>
                  <span class="text-xs text-ink-faint">{{ r.days_since_practiced | number:'1.0-0' }}d ago</span>
                </a>
              }
            </div>
          } @else {
            <p class="text-sm text-ink-muted leading-relaxed">Signs you've practiced come back here when it's time to review them, so they stick.</p>
          }
          <a routerLink="/practice" class="block mt-6 rounded-2xl bg-surface border border-border/70 px-5 py-4 hover:border-accent/50 transition-colors">
            <p class="text-xs text-ink-faint">Practice</p>
            <p class="text-[15px] text-ink mt-1">{{ (latest?.signsLearned ?? 0) >= 5 ? 'Sign a short story made from your ' + latest!.signsLearned + ' learned signs' : 'Learn ' + (5 - (latest?.signsLearned ?? 0)) + ' more signs to unlock stories' }} →</p>
          </a>
        </aside>
      </div>
    </div>
  `,
})
export class DashboardComponent implements OnInit {
  private progressService = inject(UserProgressService);
  private auth = inject(AuthService);
  private signsApi = inject(SignsApiService);
  private practiceApi = inject(PracticeApiService);

  stats: { label: string; value: string | number; note: string }[] = [];
  signs: SignProgressDetail[] = [];
  reviewQueue: ReviewDueSign[] = [];
  loading = true;
  firstName = '';
  private practiceSet: Sign[] = [];
  /** Practice-set words; lists only show these (others can't be coached). */
  private practiceWords = new Set<string>();

  get shownSigns(): SignProgressDetail[] { return this.practiceWords.size ? this.signs.filter(s => this.practiceWords.has(s.sign)) : this.signs; }
  get shownReview(): ReviewDueSign[] { return this.practiceWords.size ? this.reviewQueue.filter(r => this.practiceWords.has(r.sign)) : this.reviewQueue; }
  latest: UserStats | null = null;
  /** v5: Words (signs) and Sentences (practised stories) */
  readonly tabs: { id: 'words' | 'sentences'; label: string }[] = [{ id: 'words', label: 'Words' }, { id: 'sentences', label: 'Sentences' }];
  tab: 'words' | 'sentences' = 'words';
  practice: PracticeSummary | null = null;
  private wordStats = new Map<string, { in_sentences: number; good: number }>();

  inSentences(sign: string): string {
    const w = this.wordStats.get(sign);
    return w ? `${w.good}/${w.in_sentences} in sentences` : '';
  }

  ago(iso: string): string {
    const d = (Date.now() - new Date(iso.endsWith('Z') || iso.includes('+') ? iso : iso + 'Z').getTime()) / 86400000;
    return d < 1 ? 'today' : d < 2 ? 'yesterday' : `${Math.floor(d)}d ago`;
  }
  nextSign: Sign | null = null;
  nextReason = '';

  get greeting(): string {
    const h = new Date().getHours();
    return h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening';
  }

  get subline(): string {
    const st = this.latest;
    if (!st || st.totalSessions === 0) return 'Pick a sign, learn how it moves, then try it on camera.';
    if (st.streak >= 2) return `You've practiced ${st.streak} days in a row. Keep it going.`;
    if (st.streak === 1) return 'You practiced today. A few minutes tomorrow keeps it going.';
    return 'Welcome back. A few minutes of practice is enough.';
  }

  fiveSlots(recent: boolean[]): (boolean | null)[] {
    const r = (recent ?? []).slice(-5);
    return [...Array(5 - r.length).fill(null), ...r];
  }

  dotsLabel(s: SignProgressDetail): string {
    const r = s.recent ?? [];
    return `${r.filter(Boolean).length} of the last ${r.length} tries right`;
  }

  /** How far from "learned" (3 of the last 5 right). */
  toLearn(s: SignProgressDetail): string {
    const right = (s.recent ?? []).filter(Boolean).length;
    const need = Math.max(0, 3 - right);
    return need === 1 ? '1 more right to learn' : `${need} more right to learn`;
  }

  ngOnInit(): void {
    this.auth.user$.subscribe(u => { this.firstName = (u?.name || '').split(' ')[0]; });

    this.progressService.stats$.subscribe(st => {
      this.latest = st;
      this.signs = st.signs;
      this.buildStats();
      this.pickNext();
    });
    this.progressService.refresh().subscribe({ complete: () => (this.loading = false), error: () => (this.loading = false) });
    this.practiceApi.summary().subscribe({
      next: p => { this.practice = p; this.wordStats = new Map(p.words.map(w => [w.word, w])); this.buildStats(); },
      error: () => (this.practice = null),
    });
    this.progressService.getReviewQueue().subscribe({ next: due => (this.reviewQueue = due), error: () => (this.reviewQueue = []) });

    this.signsApi.getSignsWithReference().subscribe(set => {
      this.practiceWords = set;
      this.signsApi.getSigns().subscribe(all => {
        this.practiceSet = all.filter(s => set.has(s.word));
        this.buildStats();
        this.pickNext();
      });
    });
  }

  private buildStats(): void {
    const st = this.latest;
    if (!st) return;
    const total = this.practiceSet.length;
    this.stats = [
      { label: 'Day streak', value: st.streak, note: st.streak ? 'days in a row' : 'practice today to start one' },
      { label: 'Signs learned', value: st.signsLearned, note: total ? `of ${total} in the practice set` : '3 of the last 5 tries right' },
      { label: 'Sentences signed well', value: this.practice?.sentences_all_good ?? 0, note: this.practice?.sentences_total ? `of ${this.practice.sentences_total} tried in Practice` : 'in Practice stories' },
      { label: 'Accuracy', value: `${st.accuracy}%`, note: 'across all tries' },
    ];
  }

  /** Suggest the next sign: one in progress that isn't learned yet, else the
   * first practice-set sign never tried. */
  private pickNext(): void {
    if (!this.practiceSet.length) return;
    const bySign = new Map(this.signs.map(s => [s.sign, s]));
    const inProgress = this.practiceSet.find(s => bySign.has(s.word) && !bySign.get(s.word)!.completed);
    if (inProgress) { this.nextSign = inProgress; this.nextReason = 'Keep going with'; return; }
    const fresh = this.practiceSet.find(s => !bySign.has(s.word));
    if (fresh) { this.nextSign = fresh; this.nextReason = 'Try a new sign'; return; }
    this.nextSign = null;
  }
}
