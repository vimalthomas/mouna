import { Component, Input } from '@angular/core';
import { NgTemplateOutlet } from '@angular/common';
import { AttemptCharts } from '../../../core/models/sign.model';

interface LineSeries { key: 'you' | 'target'; label: string; values: number[] }

const W = 520;
const H = 200;
const PAD = { l: 40, r: 16, t: 14, b: 30 };

/**
 * You-vs-target charts for the results screen, from the same signals the CP
 * checks use: per-finger openness (bars, CP1), hand travel from the start
 * position over the sign (line, peaks = the big movements, CP2/CP3), and
 * brow-furrow over time for signs with a facial marker (line + threshold).
 * Colors: validated categorical pair (dataviz palette slots 1-2), with a
 * separate dark-mode step for each; identity is also carried by the legend
 * and direct labels, never color alone. Each chart has a hover readout and
 * a "show the numbers" table.
 */
@Component({
  selector: 'app-attempt-charts',
  standalone: true,
  styles: [`
    :host { --you: #2a78d6; --target: #eb6834; --grid: rgba(120,120,110,.18); --axis: rgba(120,120,110,.55); }
    :host-context(.dark) { --you: #3987e5; --target: #d95926; --grid: rgba(200,200,190,.12); --axis: rgba(200,200,190,.45); }
    .you { stroke: var(--you); fill: var(--you); }
    .target { stroke: var(--target); fill: var(--target); }
    .gridline { stroke: var(--grid); stroke-width: 1; }
    .axis { fill: currentColor; opacity: .7; font-size: 10px; }
    .hit { fill: transparent; stroke: none; }
  `],
  template: `
    @if (charts) {
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <!-- 1. Per-finger openness -->
        <section class="rounded-xl border border-border/50 bg-surface-sunk p-3 text-ink">
          <h4 class="text-sm font-semibold">How open each finger was</h4>
          <p class="text-[11px] text-ink-muted mb-1">Fingertip distance from the wrist, averaged over the sign (higher = more extended).</p>
          <ng-container [ngTemplateOutlet]="legend" />
          <svg [attr.viewBox]="'0 0 ' + W + ' ' + H" class="w-full h-auto" role="img" aria-label="Bar chart of finger openness, you versus target">
            @for (g of yTicks(fingerMax); track g) {
              <line class="gridline" [attr.x1]="PAD.l" [attr.x2]="W - PAD.r" [attr.y1]="yFor(g, fingerMax)" [attr.y2]="yFor(g, fingerMax)" />
              <text class="axis" [attr.x]="PAD.l - 6" [attr.y]="yFor(g, fingerMax) + 3" text-anchor="end">{{ tickLabel(g) }}</text>
            }
            @for (f of charts.fingers; track f; let i = $index) {
              <g (mouseenter)="hoverBar = i" (mouseleave)="hoverBar = null">
                <rect class="hit" [attr.x]="bandX(i)" [attr.y]="PAD.t" [attr.width]="bandW" [attr.height]="H - PAD.t - PAD.b" />
                <path class="you" [attr.d]="barPath(bandX(i) + bandW / 2 - barW - 1, charts.finger_openness.you[i], fingerMax)" />
                <path class="target" [attr.d]="barPath(bandX(i) + bandW / 2 + 1, charts.finger_openness.target[i], fingerMax)" />
                <text class="axis" [attr.x]="bandX(i) + bandW / 2" [attr.y]="H - PAD.b + 14" text-anchor="middle">{{ f }}</text>
              </g>
            }
            <line [attr.x1]="PAD.l" [attr.x2]="W - PAD.r" [attr.y1]="H - PAD.b" [attr.y2]="H - PAD.b" stroke="var(--axis)" />
          </svg>
          <p class="text-[11px] text-ink-muted h-4">
            @if (hoverBar !== null) {
              {{ charts.fingers[hoverBar] }}: you {{ charts.finger_openness.you[hoverBar].toFixed(2) }} · target {{ charts.finger_openness.target[hoverBar].toFixed(2) }}
            }
          </p>
          <details class="text-[11px] text-ink-muted">
            <summary class="cursor-pointer">Show the numbers</summary>
            <table class="mt-1 w-full text-left">
              <tr><th>Finger</th><th>You</th><th>Target</th></tr>
              @for (f of charts.fingers; track f; let i = $index) {
                <tr><td>{{ f }}</td><td>{{ charts.finger_openness.you[i].toFixed(2) }}</td><td>{{ charts.finger_openness.target[i].toFixed(2) }}</td></tr>
              }
            </table>
          </details>
        </section>

        <!-- 2. Hand travel over the sign -->
        <section class="rounded-xl border border-border/50 bg-surface-sunk p-3 text-ink">
          <h4 class="text-sm font-semibold">How far your hand traveled</h4>
          <p class="text-[11px] text-ink-muted mb-1">Distance from where the hand started, across the sign (shoulder widths). Peaks are the big movements.</p>
          <ng-container [ngTemplateOutlet]="legend" />
          <ng-container [ngTemplateOutlet]="lineChart" [ngTemplateOutletContext]="{ series: travelSeries, id: 'travel', threshold: null, unit: '' }" />
        </section>

        <!-- 3. Brow furrow (only signs with a facial marker) -->
        @if (browSeries.length) {
          <section class="rounded-xl border border-border/50 bg-surface-sunk p-3 text-ink">
            <h4 class="text-sm font-semibold">Brow furrow across the sign</h4>
            <p class="text-[11px] text-ink-muted mb-1">How lowered your eyebrows were (0 to 1). The dashed line is where the facial check counts it as a furrow. Early, unvalidated signal.</p>
            <ng-container [ngTemplateOutlet]="legend" />
            <ng-container [ngTemplateOutlet]="lineChart" [ngTemplateOutletContext]="{ series: browSeries, id: 'brow', threshold: charts.brow?.threshold ?? null, unit: '' }" />
          </section>
        }
      </div>
    }

    <ng-template #legend>
      <div class="flex items-center gap-4 text-[11px] text-ink-muted mb-1">
        <span class="flex items-center gap-1.5"><svg width="12" height="12"><rect class="you" width="12" height="12" rx="3" /></svg>You</span>
        <span class="flex items-center gap-1.5"><svg width="12" height="12"><rect class="target" width="12" height="12" rx="3" /></svg>Target</span>
      </div>
    </ng-template>

    <ng-template #lineChart let-series="series" let-id="id" let-threshold="threshold">
      @if (series.length) {
        <svg [attr.viewBox]="'0 0 ' + W + ' ' + H" class="w-full h-auto" role="img" [attr.aria-label]="'Line chart, you versus target'"
             (mousemove)="onLineHover($event, id)" (mouseleave)="hoverLine = null">
          @for (g of yTicks(lineMax(series, threshold)); track g) {
            <line class="gridline" [attr.x1]="PAD.l" [attr.x2]="W - PAD.r" [attr.y1]="yFor(g, lineMax(series, threshold))" [attr.y2]="yFor(g, lineMax(series, threshold))" />
            <text class="axis" [attr.x]="PAD.l - 6" [attr.y]="yFor(g, lineMax(series, threshold)) + 3" text-anchor="end">{{ tickLabel(g) }}</text>
          }
          <text class="axis" [attr.x]="PAD.l" [attr.y]="H - 8">start of sign</text>
          <text class="axis" [attr.x]="W - PAD.r" [attr.y]="H - 8" text-anchor="end">end</text>
          @if (threshold !== null) {
            <line [attr.x1]="PAD.l" [attr.x2]="W - PAD.r" [attr.y1]="yFor(threshold, lineMax(series, threshold))" [attr.y2]="yFor(threshold, lineMax(series, threshold))" stroke="var(--axis)" stroke-dasharray="4 4" />
          }
          @for (s of series; track s.key) {
            <polyline [class]="s.key" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" [attr.points]="points(s.values, lineMax(series, threshold))" style="fill: none" />
            <!-- peak marker with a surface ring, direct-labeled -->
            <circle [class]="s.key" [attr.cx]="xFor(peakIdx(s.values), s.values.length)" [attr.cy]="yFor(s.values[peakIdx(s.values)], lineMax(series, threshold))" r="4.5" stroke="rgb(var(--color-surface-sunk, var(--color-surface)))" stroke-width="2" />
            <text class="axis" [attr.x]="xFor(peakIdx(s.values), s.values.length)" [attr.y]="yFor(s.values[peakIdx(s.values)], lineMax(series, threshold)) - 9" [attr.text-anchor]="peakAnchor(s.values)">{{ s.label }} peak</text>
          }
          @if (hoverLine && hoverLine.id === id) {
            <line [attr.x1]="hoverLine.x" [attr.x2]="hoverLine.x" [attr.y1]="PAD.t" [attr.y2]="H - PAD.b" stroke="var(--axis)" />
          }
          <line [attr.x1]="PAD.l" [attr.x2]="W - PAD.r" [attr.y1]="H - PAD.b" [attr.y2]="H - PAD.b" stroke="var(--axis)" />
        </svg>
        <p class="text-[11px] text-ink-muted h-4">
          @if (hoverLine && hoverLine.id === id) {
            {{ (hoverLine.frac * 100).toFixed(0) }}% through the sign:
            @for (s of series; track s.key; let last = $last) { {{ s.label.toLowerCase() }} {{ s.values[hoverLine.idx].toFixed(2) }}{{ last ? '' : ' · ' }} }
          }
        </p>
        <details class="text-[11px] text-ink-muted">
          <summary class="cursor-pointer">Show the numbers</summary>
          <table class="mt-1 w-full text-left">
            <tr><th>Through sign</th>@for (s of series; track s.key) { <th>{{ s.label }}</th> }</tr>
            @for (i of every4(series[0].values.length); track i) {
              <tr><td>{{ (i / (series[0].values.length - 1) * 100).toFixed(0) }}%</td>@for (s of series; track s.key) { <td>{{ s.values[i].toFixed(2) }}</td> }</tr>
            }
          </table>
        </details>
      }
    </ng-template>
  `,
  imports: [NgTemplateOutlet],
})
export class AttemptChartsComponent {
  @Input() charts: AttemptCharts | null = null;

  readonly W = W;
  readonly H = H;
  readonly PAD = PAD;
  hoverBar: number | null = null;
  hoverLine: { id: string; x: number; idx: number; frac: number } | null = null;

  get fingerMax(): number {
    if (!this.charts) return 1;
    return Math.max(0.5, ...this.charts.finger_openness.you, ...this.charts.finger_openness.target) * 1.1;
  }

  get bandW(): number { return (W - PAD.l - PAD.r) / 5; }
  get barW(): number { return Math.min(22, this.bandW * 0.32); }
  bandX(i: number): number { return PAD.l + i * this.bandW; }

  yFor(v: number, max: number): number { return H - PAD.b - (Math.max(0, v) / max) * (H - PAD.t - PAD.b); }
  xFor(i: number, n: number): number { return PAD.l + (n > 1 ? i / (n - 1) : 0) * (W - PAD.l - PAD.r); }

  yTicks(max: number): number[] {
    const step = max / 3;
    return [0, step, step * 2, step * 3];
  }

  /** Bar rising from the baseline with 4px-rounded top corners only. */
  barPath(x: number, v: number, max: number): string {
    const w = this.barW;
    const y0 = H - PAD.b;
    const y = this.yFor(v, max);
    const r = Math.min(4, (y0 - y) / 2, w / 2);
    return `M${x},${y0} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y0} Z`;
  }

  get travelSeries(): LineSeries[] {
    if (!this.charts) return [];
    return [
      { key: 'you', label: 'You', values: this.charts.travel.you },
      { key: 'target', label: 'Target', values: this.charts.travel.target },
    ];
  }

  get browSeries(): LineSeries[] {
    const b = this.charts?.brow;
    if (!b) return [];
    const out: LineSeries[] = [];
    if (b.you) out.push({ key: 'you', label: 'You', values: b.you });
    if (b.target) out.push({ key: 'target', label: 'Target', values: b.target });
    return out;
  }

  lineMax(series: LineSeries[], threshold: number | null): number {
    const all = series.flatMap(s => s.values);
    return Math.max(threshold ?? 0, 0.1, ...all) * 1.15;
  }

  points(values: number[], max: number): string {
    return values.map((v, i) => `${this.xFor(i, values.length)},${this.yFor(v, max)}`).join(' ');
  }

  peakIdx(values: number[]): number {
    let best = 0;
    values.forEach((v, i) => { if (v > values[best]) best = i; });
    return best;
  }

  tickLabel(v: number): string { return v === 0 ? '0' : v < 1 ? v.toFixed(2) : v.toFixed(1); }

  /** Keep peak labels inside the plot near the left/right edges. */
  peakAnchor(values: number[]): string {
    const f = values.length > 1 ? this.peakIdx(values) / (values.length - 1) : 0.5;
    return f > 0.85 ? 'end' : f < 0.15 ? 'start' : 'middle';
  }

  every4(n: number): number[] {
    const out: number[] = [];
    for (let i = 0; i < n; i += 4) out.push(i);
    if (out[out.length - 1] !== n - 1) out.push(n - 1);
    return out;
  }

  onLineHover(event: MouseEvent, id: string): void {
    const svg = event.currentTarget as SVGSVGElement;
    const rect = svg.getBoundingClientRect();
    const xView = ((event.clientX - rect.left) / rect.width) * W;
    const series = id === 'brow' ? this.browSeries : this.travelSeries;
    const n = series[0]?.values.length ?? 0;
    if (n < 2) return;
    const frac = Math.min(1, Math.max(0, (xView - PAD.l) / (W - PAD.l - PAD.r)));
    const idx = Math.round(frac * (n - 1));
    this.hoverLine = { id, x: this.xFor(idx, n), idx, frac: idx / (n - 1) };
  }
}
