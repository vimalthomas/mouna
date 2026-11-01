import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-confidence-bar',
  standalone: true,
  template: `
    <div class="flex items-center gap-3">
      <span class="text-sm text-slate-400 w-24 truncate capitalize">{{ label }}</span>
      <div class="flex-1 bg-slate-700 rounded-full h-2.5 overflow-hidden">
        <div
          class="h-full rounded-full transition-all duration-500 ease-out"
          [style.width.%]="value * 100"
          [style.background]="barColor"
        ></div>
      </div>
      <span class="text-sm font-mono w-14 text-right text-slate-300">{{ (value * 100).toFixed(1) }}%</span>
    </div>
  `,
})
export class ConfidenceBarComponent {
  @Input() label = '';
  @Input() value = 0;

  get barColor(): string {
    if (this.value > 0.8) return '#34d399';
    if (this.value > 0.5) return '#fbbf24';
    return '#f87171';
  }
}
