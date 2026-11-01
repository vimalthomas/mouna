import { Component, Input, Output, EventEmitter, inject } from '@angular/core';
import { Sign } from '../../../core/models/sign.model';
import { ThemeService } from '../../../core/services/theme.service';

const DARK_CARD: Record<string, { bg: string; border: string }> = {
  beginner:     { bg: 'linear-gradient(145deg, rgba(52,211,153,0.08) 0%, rgba(22,26,52,0.88) 100%)',   border: 'rgba(52,211,153,0.25)' },
  intermediate: { bg: 'linear-gradient(145deg, rgba(251,191,36,0.08) 0%, rgba(22,26,52,0.88) 100%)',   border: 'rgba(251,191,36,0.25)' },
  advanced:     { bg: 'linear-gradient(145deg, rgba(248,113,113,0.08) 0%, rgba(22,26,52,0.88) 100%)',  border: 'rgba(248,113,113,0.25)' },
};

const LIGHT_CARD: Record<string, { bg: string; border: string; shadow: string }> = {
  beginner:     { bg: 'linear-gradient(145deg, #d1fae5 0%, #f0fdf4 80%)',  border: '#34d399', shadow: 'rgba(52,211,153,0.25)' },
  intermediate: { bg: 'linear-gradient(145deg, #fef3c7 0%, #fffbeb 80%)',  border: '#fbbf24', shadow: 'rgba(251,191,36,0.25)' },
  advanced:     { bg: 'linear-gradient(145deg, #fee2e2 0%, #fff5f5 80%)',  border: '#f87171', shadow: 'rgba(248,113,113,0.25)' },
};

@Component({
  selector: 'app-sign-card',
  standalone: true,
  template: `
    <div
      class="rounded-xl p-5 transition-all duration-200 cursor-pointer border group hover:-translate-y-0.5"
      [style]="cardStyle"
      [class.opacity-50]="hasVideo === false"
      [class.cursor-not-allowed]="hasVideo === false"
      (click)="hasVideo !== false && cardClicked.emit(sign)"
    >
      <div class="flex items-start justify-between mb-3">
        <div class="flex items-center gap-2">
          <h3 class="text-base font-semibold capitalize group-hover:opacity-90 transition-opacity"
              [style.color]="wordColor">{{ sign.word }}</h3>
          @if (hasVideo === true) {
            <svg class="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
            </svg>
          }
          @if (hasVideo === false) {
            <span class="text-xs opacity-50" [style.color]="wordColor">No video</span>
          }
        </div>
        <span class="text-xs px-2 py-0.5 rounded-full font-medium"
              [style.background]="difficultyBg" [style.color]="difficultyColor">
          {{ sign.difficulty }}
        </span>
      </div>
      <p class="text-sm mb-4 leading-relaxed" [style.color]="descColor">{{ sign.description }}</p>
      <span class="text-xs px-2.5 py-1 rounded-full font-medium" [style]="categoryBadgeStyle">
        {{ sign.category }}
      </span>
    </div>
  `,
})
export class SignCardComponent {
  @Input({ required: true }) sign!: Sign;
  @Input() hasVideo: boolean | null = null;
  @Output() cardClicked = new EventEmitter<Sign>();

  private theme = inject(ThemeService);

  get cardStyle(): string {
    const diff = this.sign.difficulty;
    if (this.theme.isDark()) {
      const d = DARK_CARD[diff] ?? { bg: 'rgba(22,26,52,0.88)', border: 'rgba(99,102,241,0.2)' };
      return `background: ${d.bg}; border-color: ${d.border}; box-shadow: 0 2px 4px rgba(0,0,0,0.3);`;
    }
    const l = LIGHT_CARD[diff] ?? LIGHT_CARD['beginner'];
    return `background: ${l.bg}; border-color: ${l.border}; box-shadow: 0 4px 12px ${l.shadow};`;
  }

  get wordColor(): string {
    if (this.theme.isDark()) return '#f1f5f9';
    switch (this.sign.difficulty) {
      case 'beginner':     return '#065f46';
      case 'intermediate': return '#78350f';
      case 'advanced':     return '#7f1d1d';
      default:             return '#1e1b4b';
    }
  }

  get descColor(): string {
    if (this.theme.isDark()) return '#94a3b8';
    switch (this.sign.difficulty) {
      case 'beginner':     return '#065f46cc';
      case 'intermediate': return '#78350fcc';
      case 'advanced':     return '#7f1d1dcc';
      default:             return '#374151';
    }
  }

  get difficultyBg(): string {
    if (this.theme.isDark()) {
      switch (this.sign.difficulty) {
        case 'beginner':     return 'rgba(52,211,153,0.15)';
        case 'intermediate': return 'rgba(251,191,36,0.15)';
        case 'advanced':     return 'rgba(248,113,113,0.15)';
        default:             return 'rgba(100,116,139,0.15)';
      }
    }
    switch (this.sign.difficulty) {
      case 'beginner':     return 'rgba(6,95,70,0.12)';
      case 'intermediate': return 'rgba(120,53,15,0.12)';
      case 'advanced':     return 'rgba(127,29,29,0.12)';
      default:             return 'rgba(75,85,99,0.12)';
    }
  }

  get difficultyColor(): string {
    if (this.theme.isDark()) {
      switch (this.sign.difficulty) {
        case 'beginner':     return '#34d399';
        case 'intermediate': return '#fbbf24';
        case 'advanced':     return '#f87171';
        default:             return '#94a3b8';
      }
    }
    switch (this.sign.difficulty) {
      case 'beginner':     return '#047857';
      case 'intermediate': return '#b45309';
      case 'advanced':     return '#b91c1c';
      default:             return '#4b5563';
    }
  }

  get categoryBadgeStyle(): string {
    if (this.theme.isDark()) {
      return 'background: rgba(99,102,241,0.15); color: #a5b4fc;';
    }
    switch (this.sign.difficulty) {
      case 'beginner':     return 'background: rgba(6,95,70,0.1); color: #047857;';
      case 'intermediate': return 'background: rgba(120,53,15,0.1); color: #b45309;';
      case 'advanced':     return 'background: rgba(127,29,29,0.1); color: #b91c1c;';
      default:             return 'background: rgba(99,102,241,0.1); color: #4338ca;';
    }
  }
}
