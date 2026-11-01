import { Component, inject, OnInit } from '@angular/core';
import { DatePipe } from '@angular/common';
import { RouterLink } from '@angular/router';
import { UserProgressService } from '../../core/services/user-progress.service';
import { ThemeService } from '../../core/services/theme.service';
import { SignProgressDetail, UserStats } from '../../core/models/sign.model';

// Apple system colors: blue, green, orange, red
const CARD_RGB = [
  '10, 132, 255',   // systemBlue   – Sessions
  '48, 209, 88',    // systemGreen  – Signs Learned
  '255, 159, 10',   // systemOrange – Accuracy
  '255, 69, 58',    // systemRed    – Streak
];

const LIGHT_CARDS = [
  { bg: '#E5F1FF', border: '#B8DBFF', shadow: 'rgba(0,122,255,0.15)',  label: 'rgba(0,90,200,0.8)',  value: '#0055b8' },
  { bg: '#E3F9E7', border: '#B7EFC1', shadow: 'rgba(52,199,89,0.15)',  label: 'rgba(30,140,60,0.8)', value: '#1f8a3b' },
  { bg: '#FFF1E0', border: '#FFD9A8', shadow: 'rgba(255,149,0,0.15)',  label: 'rgba(180,100,0,0.8)', value: '#b35900' },
  { bg: '#FFE5E5', border: '#FFC1C1', shadow: 'rgba(255,59,48,0.15)',  label: 'rgba(180,30,30,0.8)', value: '#b31f1f' },
];

const DARK_VALUE  = ['#4da3ff', '#4cd671', '#ffb454', '#ff6b6b'];
const DARK_LABEL  = ['rgba(77,163,255,0.8)', 'rgba(76,214,113,0.8)', 'rgba(255,180,84,0.8)', 'rgba(255,107,107,0.8)'];

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [RouterLink, DatePipe],
  template: `
    <div class="max-w-6xl mx-auto">
      <div class="mb-8">
        <h2 class="text-2xl font-bold text-ink tracking-tight">Dashboard</h2>
        <p class="text-sm text-ink-muted mt-1">Track your sign language learning progress</p>
      </div>

      <!-- Stats Cards -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8 transition-opacity"
           [class.opacity-50]="loading">
        @for (card of statCards; track card.label; let i = $index) {
          <div class="rounded-2xl p-5 border" [style]="statCardStyle(i)">
            <p class="text-xs font-medium uppercase tracking-wider mb-2" [style.color]="statLabelColor(i)">{{ card.label }}</p>
            <p class="text-3xl font-bold" [style.color]="statValueColor(i)">{{ card.value }}</p>
          </div>
        }
      </div>

      <!-- Quick Actions -->
      <div class="grid grid-cols-1 gap-5 mb-8">
        <a routerLink="/coach"
          class="max-w-md rounded-2xl p-6 border transition-all duration-200 block group hover:-translate-y-0.5"
          [style]="actionCardStyle()">
          <div class="flex items-center gap-3 mb-2">
            <div class="w-10 h-10 rounded-xl flex items-center justify-center" [style]="actionIconStyle()">
              <svg class="w-5 h-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
              </svg>
            </div>
            <h3 class="text-lg font-semibold transition-colors" [style.color]="actionTitleColor()">AI Coach</h3>
          </div>
          <p class="text-sm text-ink-muted">Get real-time coaching powered by Gemini AI</p>
        </a>
      </div>

      <!-- Sign Progress detail -->
      @if (signs.length > 0) {
        <div class="rounded-2xl border mb-8 overflow-hidden" [style]="panelStyle">
          <div class="px-6 pt-6 pb-4">
            <h3 class="text-base font-semibold text-ink">Sign Progress</h3>
            <p class="text-xs text-ink-muted mt-0.5">Every sign you've practiced, most recent first</p>
          </div>
          <div>
            @for (s of signs; track s.sign) {
              <div class="flex items-center justify-between gap-4 px-6 py-3 border-t border-border/50">
                <div class="flex items-center gap-3 min-w-0">
                  @if (s.completed) {
                    <span class="shrink-0 inline-flex items-center justify-center w-5 h-5 rounded-full"
                          style="background: rgba(48,209,88,0.15);">
                      <svg class="w-3 h-3" style="color: #30d158;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                      </svg>
                    </span>
                  } @else {
                    <span class="shrink-0 w-5 h-5 rounded-full border-2 border-border"></span>
                  }
                  <span class="text-sm text-ink capitalize truncate">{{ s.sign }}</span>
                  @if (s.completed) {
                    <span class="shrink-0 text-[10px] font-medium uppercase tracking-wide px-1.5 py-0.5 rounded"
                          style="background: rgba(48,209,88,0.12); color: #30d158;">Learned</span>
                  } @else {
                    <span class="shrink-0 text-[10px] font-medium uppercase tracking-wide px-1.5 py-0.5 rounded bg-border/50 text-ink-muted">In Progress</span>
                  }
                </div>
                <div class="flex items-center gap-4 shrink-0">
                  @if (s.completedAt) {
                    <span class="text-xs text-ink-muted hidden sm:inline">{{ s.completedAt | date:'MMM d' }}</span>
                  }
                  <span class="text-xs text-ink-muted font-mono w-14 text-right">{{ s.correctAttempts }}/{{ s.totalAttempts }}</span>
                  <div class="w-20 rounded-full h-1.5 hidden md:block" [style]="progressTrackStyle">
                    <div class="h-1.5 rounded-full" [style.width.%]="s.accuracy"
                         [style.background]="s.completed ? '#30d158' : '#ff9500'"></div>
                  </div>
                  <span class="text-xs font-semibold w-10 text-right" [style.color]="s.completed ? '#30d158' : 'inherit'">{{ s.accuracy }}%</span>
                </div>
              </div>
            }
          </div>
        </div>
      }

      <!-- Weak Signs / Empty state -->
      @if (weakSigns.length > 0) {
        <div class="rounded-2xl p-6 border" [style]="panelStyle">
          <h3 class="text-base font-semibold text-ink mb-4">Signs to Improve</h3>
          <div class="space-y-3">
            @for (ws of weakSigns; track ws.sign) {
              <div class="flex items-center justify-between">
                <span class="text-sm text-ink-muted capitalize">{{ ws.sign }}</span>
                <div class="flex items-center gap-3">
                  <div class="w-28 rounded-full h-1.5" [style]="progressTrackStyle">
                    <div class="h-1.5 rounded-full" [style.width.%]="ws.accuracy"
                         style="background: #ff9500;"></div>
                  </div>
                  <span class="text-xs text-ink-muted font-mono w-10 text-right">{{ ws.accuracy }}%</span>
                </div>
              </div>
            }
          </div>
        </div>
      } @else if (hasPracticed) {
        <div class="rounded-2xl p-8 border text-center" [style]="panelStyle">
          <div class="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4"
               [style]="emptyIconStyle">
            <svg class="w-8 h-8 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5" />
            </svg>
          </div>
          <p class="text-ink font-medium">All signs looking good</p>
          <p class="text-sm text-ink-muted mt-1">No signs currently need improvement — keep practicing to learn more</p>
        </div>
      } @else {
        <div class="rounded-2xl p-8 border text-center" [style]="panelStyle">
          <div class="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4"
               [style]="emptyIconStyle">
            <svg class="w-8 h-8 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M4.26 10.147a60.436 60.436 0 00-.491 6.347A48.627 48.627 0 0112 20.904a48.627 48.627 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.57 50.57 0 00-2.658-.813A59.905 59.905 0 0112 3.493a59.902 59.902 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.697 50.697 0 0112 13.489a50.702 50.702 0 017.74-3.342" />
            </svg>
          </div>
          <p class="text-ink font-medium">No practice data yet</p>
          <p class="text-sm text-ink-muted mt-1">Start practicing to track your progress</p>
        </div>
      }
    </div>
  `,
})
export class DashboardComponent implements OnInit {
  private progressService = inject(UserProgressService);
  private themeService = inject(ThemeService);

  statCards: { label: string; value: string | number }[] = [];
  weakSigns: { sign: string; accuracy: number }[] = [];
  signs: SignProgressDetail[] = [];

  private get dark(): boolean { return this.themeService.isDark(); }

  statCardStyle(i: number): string {
    const rgb = CARD_RGB[i] ?? CARD_RGB[0];
    if (this.dark) {
      return `background: rgba(${rgb},0.14); border-color: rgba(${rgb},0.3);`;
    }
    const lc = LIGHT_CARDS[i] ?? LIGHT_CARDS[0];
    return `background: ${lc.bg}; border-color: ${lc.border}; box-shadow: 0 1px 3px ${lc.shadow};`;
  }

  statLabelColor(i: number): string {
    return this.dark ? (DARK_LABEL[i] ?? DARK_LABEL[0]) : (LIGHT_CARDS[i] ?? LIGHT_CARDS[0]).label;
  }

  statValueColor(i: number): string {
    return this.dark ? (DARK_VALUE[i] ?? DARK_VALUE[0]) : (LIGHT_CARDS[i] ?? LIGHT_CARDS[0]).value;
  }

  actionCardStyle(): string {
    if (this.dark) {
      return 'background: rgba(10,132,255,0.12); border-color: rgba(10,132,255,0.28);';
    }
    return 'background: #E5F1FF; border-color: #B8DBFF; box-shadow: 0 1px 3px rgba(0,122,255,0.15);';
  }

  actionIconStyle(): string {
    return this.dark
      ? 'background: rgba(10,132,255,0.18); border: 1px solid rgba(10,132,255,0.3);'
      : 'background: rgba(0,122,255,0.12); border: 1px solid rgba(0,122,255,0.25);';
  }

  actionTitleColor(): string {
    return this.dark ? '#4da3ff' : '#0055b8';
  }

  get panelStyle(): string {
    if (this.dark) {
      return 'background: #1c1c1e; border-color: #38383a;';
    }
    return 'background: #ffffff; border-color: #d1d1d6; box-shadow: 0 1px 3px rgba(0,0,0,0.06);';
  }

  get progressTrackStyle(): string {
    return this.dark ? 'background: rgba(10,132,255,0.18);' : 'background: rgba(0,122,255,0.14);';
  }

  get emptyIconStyle(): string {
    return this.dark
      ? 'background: rgba(10,132,255,0.14); border: 1px solid rgba(10,132,255,0.25);'
      : 'background: rgba(0,122,255,0.1); border: 1px solid rgba(0,122,255,0.25);';
  }

  loading = true;
  hasPracticed = false;

  ngOnInit(): void {
    this.progressService.stats$.subscribe(stats => {
      this.statCards = [
        { label: 'Sessions',      value: stats.totalSessions  },
        { label: 'Signs Learned', value: stats.signsLearned   },
        { label: 'Accuracy',      value: `${stats.accuracy}%` },
        { label: 'Streak',        value: stats.streak         },
      ];
      this.weakSigns = stats.weakSigns;
      this.signs = stats.signs;
      this.hasPracticed = stats.totalSessions > 0;
    });

    this.progressService.refresh().subscribe({
      complete: () => (this.loading = false),
      error: () => (this.loading = false),
    });
  }
}
