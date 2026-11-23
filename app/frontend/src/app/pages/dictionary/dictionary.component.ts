import { Component, inject, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { SignsApiService } from '../../core/services/signs-api.service';
import { Sign, SignCatalogEntry } from '../../core/models/sign.model';
import { SignCardComponent } from '../../shared/components/sign-card/sign-card.component';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-dictionary',
  standalone: true,
  imports: [FormsModule, SignCardComponent, RouterLink],
  template: `
    <div class="max-w-6xl mx-auto pt-2">
      <header class="flex items-end justify-between gap-6 flex-wrap mb-6">
        <div>
          <h1 class="font-display text-4xl text-ink">Dictionary</h1>
          <p class="text-ink-muted mt-2 max-w-2xl">Grouped by the categories of the ASL-CDI, a standard inventory of the signs people learn first. <span class="text-teal font-medium">Essential</span> signs are on that inventory and used often by deaf signers (ASL-LEX).</p>
        </div>
        <div class="relative w-full sm:w-72">
          <svg class="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-ink-faint" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" d="m21 21-5.2-5.2m0 0A7.5 7.5 0 1 0 5.2 5.2a7.5 7.5 0 0 0 10.6 10.6Z" />
          </svg>
          <input type="search" [(ngModel)]="searchTerm" (ngModelChange)="filterSigns()" placeholder="Search signs" aria-label="Search signs"
            class="w-full h-10 bg-surface border border-border/70 rounded-xl pl-10 pr-3 text-sm text-ink placeholder:text-ink-faint focus:outline-none focus:border-accent/60 transition-colors" />
        </div>
      </header>

      <!-- Categories -->
      <nav class="flex flex-wrap gap-1 mb-6 -mx-1" aria-label="Categories">
        @for (cat of ['essential'].concat(categories, ['all']); track cat) {
          <button (click)="selectCategory(cat)"
            class="shrink-0 h-8 px-3 rounded-lg text-sm transition-colors"
            [class]="selectedCategory === cat ? 'bg-ink text-background' : 'text-ink-muted hover:text-ink hover:bg-surface-sunk'"
          >{{ cat === 'all' ? 'All' : cat === 'essential' ? 'Essential' : cat }} <span class="opacity-60">{{ countFor(cat) }}</span></button>
        }
      </nav>

      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        @for (sign of filteredSigns; track sign.word) {
          <app-sign-card [sign]="sign" [hasVideo]="comingSoon(sign) ? false : videoSet.has(sign.word.toLowerCase().replace('_', ' '))" [practiceReady]="practiceSet.has(sign.word)" [essential]="!!catalog[sign.word]?.essential" [comingSoon]="comingSoon(sign)" (cardClicked)="openVideo($event)" />
        }
      </div>

      @if (filteredSigns.length === 0) {
        <div class="text-center py-20">
          <p class="text-ink">No signs match "{{ searchTerm }}".</p>
          <p class="text-sm text-ink-muted mt-1">Try another word or category.</p>
        </div>
      }

      <!-- Video -->
      @if (selectedSign) {
        <div class="fixed inset-0 bg-black/60 backdrop-blur-[2px] z-50 flex items-center justify-center p-4" (click)="closeVideo()">
          <div class="bg-surface rounded-2xl border border-border/70 shadow-2xl max-w-xl w-full overflow-hidden" (click)="$event.stopPropagation()" role="dialog" [attr.aria-label]="selectedSign.word">
            <div class="flex items-start justify-between px-6 pt-5 pb-4">
              <div>
                <h3 class="font-display text-3xl text-ink capitalize">{{ selectedSign.word.replace('_', ' ') }}</h3>
                <p class="text-sm text-ink-muted mt-1">{{ selectedSign.description }}</p>
              </div>
              <button (click)="closeVideo()" aria-label="Close" class="text-ink-faint hover:text-ink transition-colors p-1 -mr-1">
                <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12" /></svg>
              </button>
            </div>
            <div class="bg-black aspect-video flex items-center justify-center">
              @if (videoUrl) {
                <video [src]="videoUrl" autoplay loop controls [muted]="true" playsinline class="w-full h-full object-contain"></video>
              } @else if (videoLoading) {
                <div class="w-7 h-7 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
              } @else {
                <p class="text-sm text-white/60">No video for this sign yet.</p>
              }
            </div>
            <div class="flex items-center justify-between px-6 py-4">
              <span class="text-xs text-ink-faint">{{ selectedSign.category }}@if (catalog[selectedSign.word]?.frequency) { · used often: {{ catalog[selectedSign.word]!.frequency }} / 7 }</span>
              @if (practiceSet.has(selectedSign.word)) {
                <a [routerLink]="['/coach']" [queryParams]="{ sign: selectedSign.word }" class="h-9 px-4 inline-flex items-center rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-colors">Practice this sign</a>
              } @else {
                <span class="text-xs text-ink-faint">Not in the coached set yet</span>
              }
            </div>
          </div>
        </div>
      }
    </div>
  `,
})
export class DictionaryComponent implements OnInit {
  private signsApi = inject(SignsApiService);

  allSigns: Sign[] = [];
  filteredSigns: Sign[] = [];
  categories: string[] = [];
  searchTerm = '';
  selectedCategory = 'essential';

  selectedSign: Sign | null = null;
  videoUrl: string | null = null;
  videoLoading = false;
  videoSet = new Set<string>();
  practiceSet = new Set<string>();
  /** Only the practice signs are listed for now; the rest stay hidden until they're ready. */
  private practiceLoaded = false;

  /** Per-sign ASL-CDI category / frequency / essential (sign_catalog.json). */
  catalog: Record<string, SignCatalogEntry> = {};
  private rawSigns: Sign[] = [];

  ngOnInit(): void {
    this.signsApi.getSigns().subscribe(signs => { this.rawSigns = signs; this.rebuild(); });
    this.signsApi.getSignCatalog().subscribe({ next: c => { this.catalog = c; this.rebuild(); }, error: () => this.rebuild() });
    this.signsApi.getSignsWithVideo().subscribe(set => { this.videoSet = set; });
    this.signsApi.getSignsWithReference().subscribe({ next: set => { this.practiceSet = set; this.practiceLoaded = true; this.rebuild(); }, error: () => {} });
  }

  /** Essential signs outside the 100-sign vocabulary (clips being added). */
  comingSoon(sign: Sign): boolean { return !this.rawSigns.some(s => s.word === sign.word); }

  /** Re-label every sign with its ASL-CDI category, add the essential signs
   * that aren't in the vocabulary yet, and order: essential first, then by
   * how often deaf signers use the sign. */
  private rebuild(): void {
    const cat = this.catalog;
    const known = new Set(this.rawSigns.map(s => s.word));
    const extras: Sign[] = Object.entries(cat)
      .filter(([w, c]) => !known.has(w) && c.essential)
      .map(([w, c]) => ({ word: w, category: c.category, difficulty: '', description: 'Clips are being added for this sign.', manual_checks: [], nms_required: false, nms_type: null, nms_notes: null }));
    let all = [...this.rawSigns.map(s => ({ ...s, category: cat[s.word]?.category ?? 'Other signs' })), ...extras];
    // practice signs only (hidden until the list arrives, so nothing else flashes up)
    all = this.practiceLoaded ? all.filter(s => this.practiceSet.has(s.word)) : [];
    all.sort((a, b) => Number(!!cat[b.word]?.essential) - Number(!!cat[a.word]?.essential)
      || (cat[b.word]?.frequency ?? 0) - (cat[a.word]?.frequency ?? 0) || a.word.localeCompare(b.word));
    this.allSigns = all;
    const counts = new Map<string, number>();
    for (const s of all) counts.set(s.category, (counts.get(s.category) ?? 0) + 1);
    this.categories = [...counts.keys()].filter(c => c !== 'Other signs').sort((a, b) => counts.get(b)! - counts.get(a)! || a.localeCompare(b));
    if (counts.has('Other signs')) this.categories.push('Other signs');
    this.filterSigns();
  }

  countFor(cat: string): number {
    if (cat === 'all') return this.allSigns.length;
    if (cat === 'essential') return this.allSigns.filter(s => this.catalog[s.word]?.essential).length;
    return this.allSigns.filter(s => s.category === cat).length;
  }

  selectCategory(category: string): void {
    this.selectedCategory = category;
    this.filterSigns();
  }

  filterSigns(): void {
    let signs = this.allSigns;
    if (this.selectedCategory === 'essential') signs = signs.filter(s => this.catalog[s.word]?.essential);
    else if (this.selectedCategory !== 'all') signs = signs.filter(s => s.category === this.selectedCategory);
    if (this.searchTerm.trim()) {
      const term = this.searchTerm.toLowerCase();
      signs = signs.filter(s => s.word.toLowerCase().includes(term) || s.description.toLowerCase().includes(term));
    }
    this.filteredSigns = signs;
  }

  openVideo(sign: Sign): void {
    this.selectedSign = sign;
    this.videoUrl = null;
    this.videoLoading = true;

    this.signsApi.getVideo(sign.word).subscribe({
      next: (res) => {
        this.videoUrl = `${environment.apiBaseUrl}${res.video_url}`;
        this.videoLoading = false;
      },
      error: () => {
        this.videoUrl = null;
        this.videoLoading = false;
      },
    });
  }

  closeVideo(): void {
    this.selectedSign = null;
    this.videoUrl = null;
  }
}
