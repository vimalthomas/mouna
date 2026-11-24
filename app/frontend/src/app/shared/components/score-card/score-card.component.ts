import { Component, Input } from '@angular/core';
import { HandCountCheck, ParameterScore } from '../../../core/models/sign.model';

const ORDER = ['handshape', 'location', 'orientation', 'movement', 'nonmanual'];
const LEVEL: Record<string, string> = { fluent: 'Fluent', close: 'Close', needs_work: 'Needs work' };
const CP_WORDS: Record<string, string> = {
  palm_x: 'which way the palm faces (sideways)', palm_y: 'which way the palm faces (up/down)', palm_z: 'which way the palm faces (toward you/away)',
  finger_thumb: 'thumb', finger_index: 'index finger', finger_middle: 'middle finger', finger_ring: 'ring finger',
  finger_little: 'little finger', thumb_out: 'thumb position', loc_x: 'hand position side to side', loc_y: 'hand height',
  dir_x: 'side-to-side motion', dir_y: 'up-and-down motion', strokes: 'number of strokes', travel: 'size of the movement', gap: 'hands touching',
};

/** The five parameters of the attempt, each scored against the range fluent
 * signers use. "Estimate" until the held-out benchmark shows the score is reliable. */
@Component({
  selector: 'app-score-card',
  standalone: true,
  template: `
    @if (rows.length || hand) {
      <div class="rounded-xl border border-border/60 divide-y divide-border/50 mb-3">
        @for (r of rows; track r.key) {
          <div class="px-4 py-3">
            <div class="grid items-center gap-3" style="grid-template-columns: minmax(110px, 150px) 104px 1fr auto">
              <span class="text-sm font-semibold text-ink">{{ r.p.label }}</span>
              <span class="flex items-center gap-1.5 text-xs font-semibold" [class]="levelClass(r.p.level)">
                <span class="w-2.5 h-2.5 rounded-full border-2 border-current" [class.bg-current]="r.p.level === 'fluent'"
                  [class.border-dashed]="!r.p.level"></span>
                {{ r.p.level ? levels[r.p.level] : 'Not measured yet' }}
              </span>
              <div class="h-2 rounded-full bg-surface-sunk overflow-hidden">
                @if (r.p.score !== null) {
                  <div class="h-full rounded-full" [style.width.%]="r.p.score" [class]="barClass(r.p.level)"></div>
                }
              </div>
              <span class="text-[11px]" [class]="r.p.reliable ? 'text-emerald-500 font-semibold' : 'text-ink-faint'"
                [title]="r.p.reliable ? 'Passed the held-out benchmark' : 'Not yet reliable on the held-out benchmark: take it as a hint'">
                {{ r.p.level ? (r.p.reliable ? 'trusted' : 'estimate') : '' }}</span>
            </div>
            @if (r.lines.length) {
              <ul class="mt-1.5 text-[13px] text-ink-muted space-y-0.5">
                @for (l of r.lines; track l) { <li>{{ l }}</li> }
              </ul>
            }
          </div>
        }
        @if (hand; as h) {
          <div class="px-4 py-3">
            <div class="grid items-center gap-3" style="grid-template-columns: minmax(110px, 150px) 104px 1fr auto">
              <span class="text-sm font-semibold text-ink">Hand count</span>
              <span class="flex items-center gap-1.5 text-xs font-semibold" [class]="h.cls">
                <span class="w-2.5 h-2.5 rounded-full border-2 border-current" [class.bg-current]="h.match === true"
                  [class.border-dashed]="h.match === null"></span>
                {{ h.text }}
              </span>
              <span></span>
              <span class="text-[11px]" [class]="h.trusted ? 'text-emerald-500 font-semibold' : 'text-ink-faint'"
                [title]="h.trusted ? 'Checked by AI with your video next to two fluent signers; passed the held-out benchmark' : 'Our own estimate: not yet reliable on the held-out benchmark'">
                {{ h.match === null ? '' : (h.trusted ? 'trusted' : 'estimate') }}</span>
            </div>
            @if (h.line) { <p class="mt-1.5 text-[13px] text-ink-muted">{{ h.line }}</p> }
          </div>
        }
      </div>
      <p class="text-[11px] text-ink-faint mb-4">
        Each part is compared with fluent signers of this sign (the joint angles of your hand, where it is next to your face and body,
        and the path it takes over time). Hand count is checked by AI, which looks at frames of your attempt next to two fluent signers
        along with these measurements; in testing it caught 76% of wrong hand counts and flagged 5% of correct signing. "Estimate" means the score hasn't yet passed our accuracy test, so treat it as a hint.
      </p>
    }
  `,
})
export class ScoreCardComponent {
  @Input() scores: Record<string, ParameterScore> | null | undefined = null;
  @Input() handCountLocal: boolean | null = null;
  @Input() handCountCheck: HandCountCheck | null = null;
  readonly levels = LEVEL;

  /** Hand count: the AI's answer when it has one, else our own check (an estimate). */
  get hand(): { text: string; cls: string; match: boolean | null; trusted: boolean; line: string } | null {
    const c = this.handCountCheck;
    const local = this.handCountLocal;
    if (!c && local === null) return null;
    const ai = c?.status === 'done' && (c.verdict === 'same' || c.verdict === 'different') ? c.verdict : null;
    const match = ai ? ai === 'same' : local;
    const pending = c?.status === 'pending';
    let line = '';
    if (ai === 'different') line = c?.note || 'The number of hands you used differs from fluent signers.';
    else if (pending) line = 'Checking with AI against fluent signers…';
    else if (!ai && c) line = "The AI check couldn't run this time, so this is our own estimate.";
    return {
      text: match === null ? 'Not measured' : match ? 'Matches' : 'Different',
      cls: match === null ? 'text-ink-faint' : match ? 'text-emerald-500' : 'text-rose-500',
      match, trusted: !!ai, line,
    };
  }

  get rows(): { key: string; p: ParameterScore; lines: string[] }[] {
    const s = this.scores;
    if (!s) return [];
    return ORDER.filter(k => s[k]).map(k => ({ key: k, p: s[k], lines: this.lines(k, s[k]) }));
  }

  private lines(key: string, p: ParameterScore): string[] {
    if (!p.level || p.level === 'fluent') return [];
    if (p.evidence?.length) {
      return p.evidence.slice(0, 2).map(e => {
        const deg = e.unit === '°';
        const f = (v: number) => (deg ? `${Math.round(v)}°` : v.toFixed(2));
        return `${this.dimName(e.dim)}: yours ${f(e.yours)}, fluent signers ${f(e.fluent[0])}–${f(e.fluent[1])}${deg ? '' : ' shoulder widths'}`;
      });
    }
    if (key === 'movement') return ["Your hand's path over time differs from how fluent signers move."];
    if (p.cp_worst) return [`Furthest from fluent: ${CP_WORDS[p.cp_worst] ?? p.cp_worst}.`];
    return [];
  }

  private dimName(d: string): string {
    const s = d.replace(/^little/, 'little finger').replace(/^(index|middle|ring) /, '$1 finger ');
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  levelClass(l: string | null): string {
    return l === 'fluent' ? 'text-emerald-500' : l === 'close' ? 'text-amber-500' : l === 'needs_work' ? 'text-rose-500' : 'text-ink-faint';
  }

  barClass(l: string | null): string {
    return l === 'fluent' ? 'bg-emerald-500' : l === 'close' ? 'bg-amber-500' : 'bg-rose-500';
  }
}
