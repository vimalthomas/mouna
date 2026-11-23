import { Component, EventEmitter, Input, Output } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Sign } from '../../../core/models/sign.model';

@Component({
  selector: 'app-sign-card',
  standalone: true,
  imports: [RouterLink],
  template: `
    <div
      class="group h-full flex flex-col rounded-2xl bg-surface border border-border/70 px-5 pt-4 pb-4 transition-colors"
      [class.hover:border-accent/50]="hasVideo !== false"
      [class.cursor-pointer]="hasVideo !== false"
      [class.opacity-60]="hasVideo === false && !comingSoon"
      (click)="hasVideo !== false && cardClicked.emit(sign)"
      [attr.role]="hasVideo !== false ? 'button' : null"
      [attr.tabindex]="hasVideo !== false ? 0 : null"
      (keydown.enter)="hasVideo !== false && cardClicked.emit(sign)"
    >
      <div class="flex items-baseline justify-between gap-3">
        <h3 class="font-display text-2xl text-ink capitalize">{{ sign.word.replace('_', ' ') }}</h3>
        @if (hasVideo !== false) {
          <svg class="w-4 h-4 text-ink-faint group-hover:text-accent transition-colors shrink-0" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.6-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5Z" />
          </svg>
        }
      </div>
      <p class="text-sm text-ink-muted leading-relaxed mt-1.5 flex-1">{{ sign.description }}</p>
      <div class="flex items-center justify-between mt-4 text-xs">
        <span class="flex items-center gap-2">
          @if (essential) { <span class="text-teal font-medium">Essential</span> }
          <span class="text-ink-faint">{{ sign.category }}@if (hasVideo === false && !comingSoon) { · no video }</span>
        </span>
        @if (practiceReady) {
          <a [routerLink]="['/coach']" [queryParams]="{ sign: sign.word }" (click)="$event.stopPropagation()"
             class="text-accent font-medium hover:underline underline-offset-4">Practice →</a>
        } @else if (comingSoon) {
          <span class="text-ink-faint">Coming soon</span>
        }
      </div>
    </div>
  `,
})
export class SignCardComponent {
  @Input({ required: true }) sign!: Sign;
  @Input() hasVideo: boolean | null = null;
  /** In the coach's practice set (fully supported, graded). */
  @Input() practiceReady = false;
  /** On ASL-CDI and frequent in ASL-LEX. */
  @Input() essential = false;
  /** Essential sign whose clips are still being added. */
  @Input() comingSoon = false;
  @Output() cardClicked = new EventEmitter<Sign>();
}
