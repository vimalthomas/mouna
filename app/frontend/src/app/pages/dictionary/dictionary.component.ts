import { Component, inject, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { SignsApiService } from '../../core/services/signs-api.service';
import { Sign } from '../../core/models/sign.model';
import { SignCardComponent } from '../../shared/components/sign-card/sign-card.component';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-dictionary',
  standalone: true,
  imports: [FormsModule, SignCardComponent],
  template: `
    <div class="max-w-6xl mx-auto">
      <div class="mb-6">
        <h2 class="text-2xl font-bold text-white">Sign Dictionary</h2>
        <p class="text-sm text-slate-400 mt-1">Browse {{ allSigns.length }} ASL signs — click a card to watch the video</p>
      </div>

      <!-- Search -->
      <div class="mb-5">
        <div class="relative">
          <svg class="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
          </svg>
          <input
            type="text"
            [(ngModel)]="searchTerm"
            (ngModelChange)="filterSigns()"
            class="w-full bg-slate-800/80 border border-slate-700/50 rounded-xl pl-11 pr-4 py-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/25 transition-all text-sm"
          />
        </div>
      </div>

      <!-- Category Tabs -->
      <div class="flex flex-wrap gap-2 mb-6">
        <button
          (click)="selectCategory('all')"
          class="px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200"
          [class]="selectedCategory === 'all'
            ? 'bg-indigo-500 text-white shadow-lg shadow-indigo-500/25'
            : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-300'"
        >
          All ({{ allSigns.length }})
        </button>
        @for (cat of categories; track cat) {
          <button
            (click)="selectCategory(cat)"
            class="px-3 py-1.5 rounded-lg text-xs font-medium capitalize transition-all duration-200"
            [class]="selectedCategory === cat
              ? 'bg-indigo-500 text-white shadow-lg shadow-indigo-500/25'
              : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-300'"
          >
            {{ cat }}
          </button>
        }
      </div>

      <!-- Signs Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        @for (sign of filteredSigns; track sign.word) {
          <app-sign-card [sign]="sign" [hasVideo]="videoSet.has(sign.word.toLowerCase().replace('_', ' '))" (cardClicked)="openVideo($event)" />
        }
      </div>

      @if (filteredSigns.length === 0) {
        <div class="text-center py-16">
          <svg class="w-12 h-12 text-slate-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1">
            <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
          </svg>
          <p class="text-base text-slate-400">No signs found</p>
          <p class="text-sm text-slate-500 mt-1">Try a different search term or category</p>
        </div>
      }

      <!-- Video Modal -->
      @if (selectedSign) {
        <div
          class="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          (click)="closeVideo()"
        >
          <div
            class="bg-slate-800 rounded-2xl border border-slate-700/50 shadow-2xl max-w-lg w-full overflow-hidden"
            (click)="$event.stopPropagation()"
          >
            <!-- Header -->
            <div class="flex items-center justify-between px-5 py-4 border-b border-slate-700/50">
              <div>
                <h3 class="text-lg font-semibold text-white capitalize">{{ selectedSign.word }}</h3>
                <p class="text-xs text-slate-400 mt-0.5">{{ selectedSign.category }}</p>
              </div>
              <button
                (click)="closeVideo()"
                class="text-slate-400 hover:text-white transition-colors p-1"
              >
                <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <!-- Video -->
            @if (videoUrl) {
              <div class="bg-black">
                <video
                  [src]="videoUrl"
                  autoplay
                  loop
                  controls
                  playsinline
                  class="w-full"
                  style="max-height: 400px;"
                ></video>
              </div>
            } @else if (videoLoading) {
              <div class="flex items-center justify-center py-16 bg-slate-900">
                <div class="w-8 h-8 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin"></div>
              </div>
            } @else {
              <div class="flex items-center justify-center py-16 bg-slate-900">
                <div class="text-center">
                  <svg class="w-10 h-10 text-slate-600 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                  </svg>
                  <p class="text-sm text-slate-500">No video available</p>
                </div>
              </div>
            }

            <!-- Description -->
            <div class="px-5 py-4">
              <p class="text-sm text-slate-300 leading-relaxed">{{ selectedSign.description }}</p>
              <div class="flex items-center gap-2 mt-3">
                <span
                  class="text-xs px-2 py-0.5 rounded-full font-medium"
                  [style.background]="selectedSign.difficulty === 'beginner' ? 'rgba(52,211,153,0.15)' : 'rgba(251,191,36,0.15)'"
                  [style.color]="selectedSign.difficulty === 'beginner' ? '#34d399' : '#fbbf24'"
                >
                  {{ selectedSign.difficulty }}
                </span>
                <span class="text-xs px-2 py-0.5 rounded-full bg-indigo-500/15 text-indigo-400 font-medium">
                  {{ selectedSign.category }}
                </span>
              </div>
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
  selectedCategory = 'all';

  selectedSign: Sign | null = null;
  videoUrl: string | null = null;
  videoLoading = false;
  videoSet = new Set<string>();

  ngOnInit(): void {
    this.signsApi.getSigns().subscribe(signs => {
      this.allSigns = signs;
      this.filteredSigns = signs;
      this.categories = [...new Set(signs.map(s => s.category))].sort();
    });
    this.signsApi.getSignsWithVideo().subscribe(set => {
      this.videoSet = set;
    });
  }

  selectCategory(category: string): void {
    this.selectedCategory = category;
    this.filterSigns();
  }

  filterSigns(): void {
    let signs = this.allSigns;
    if (this.selectedCategory !== 'all') {
      signs = signs.filter(s => s.category === this.selectedCategory);
    }
    if (this.searchTerm.trim()) {
      const term = this.searchTerm.toLowerCase();
      signs = signs.filter(
        s => s.word.toLowerCase().includes(term) || s.description.toLowerCase().includes(term)
      );
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
