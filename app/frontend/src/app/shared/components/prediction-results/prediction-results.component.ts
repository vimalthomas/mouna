import { Component, Input } from '@angular/core';
import { PredictionResult } from '../../../core/models/sign.model';
import { ConfidenceBarComponent } from '../confidence-bar/confidence-bar.component';

@Component({
  selector: 'app-prediction-results',
  standalone: true,
  imports: [ConfidenceBarComponent],
  template: `
    @if (result) {
      <div class="bg-slate-800/80 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50">
        <div class="flex items-center justify-between mb-5">
          <h3 class="text-base font-semibold text-slate-200">Prediction</h3>
          @if (result.uncertain) {
            <span class="text-xs px-2.5 py-1 rounded-full bg-amber-500/15 text-amber-400 font-medium">
              Uncertain
            </span>
          }
        </div>

        <div class="text-4xl font-bold text-indigo-400 mb-6 capitalize">
          {{ result.prediction }}
        </div>

        <div class="space-y-3">
          @for (pred of result.top3; track pred.sign) {
            <app-confidence-bar [label]="pred.sign" [value]="pred.confidence" />
          }
        </div>

        @if (result.gemini_validation) {
          <div class="mt-4 p-3 rounded-xl border"
            [class]="result.gemini_validation.valid ? 'bg-emerald-500/10 border-emerald-500/20' : 'bg-amber-500/10 border-amber-500/20'">
            <div class="flex items-center gap-2 mb-1">
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"
                [class]="result.gemini_validation.valid ? 'text-emerald-400' : 'text-amber-400'">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25" />
              </svg>
              <span class="text-xs font-medium"
                [class]="result.gemini_validation.valid ? 'text-emerald-400' : 'text-amber-400'">
                Gemini AI: {{ result.gemini_validation.valid ? 'Looks correct' : 'Needs review' }}
              </span>
            </div>
            <p class="text-xs text-slate-400 leading-relaxed">{{ result.gemini_validation.explanation }}</p>
            @if (result.gemini_validation.suggestion) {
              <p class="text-xs text-indigo-400 mt-1">Suggestion: {{ result.gemini_validation.suggestion }}</p>
            }
          </div>
        }

        @if (result.inference_time_ms) {
          <p class="text-xs text-slate-500 mt-3 font-mono">
            Inference: {{ result.inference_time_ms }}ms
          </p>
        }
      </div>
    } @else {
      <div class="bg-slate-800/80 backdrop-blur-sm rounded-2xl p-8 text-center border border-slate-700/50">
        <div class="w-16 h-16 rounded-full bg-slate-700/50 flex items-center justify-center mx-auto mb-4">
          <svg class="w-8 h-8 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
          </svg>
        </div>
        <p class="text-base text-slate-400">No predictions yet</p>
        <p class="text-sm text-slate-500 mt-1">Start recording to see results</p>
      </div>
    }
  `,
})
export class PredictionResultsComponent {
  @Input() result: PredictionResult | null = null;
}
