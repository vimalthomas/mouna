import { Component, inject } from '@angular/core';
import { AsyncPipe } from '@angular/common';
import { Router, RouterLink, RouterLinkActive } from '@angular/router';
import { ThemeService } from '../../../core/services/theme.service';
import { I18nService } from '../../../core/services/i18n.service';
import { TranslatePipe } from '../../../core/pipes/translate.pipe';
import { AuthService } from '../../../core/services/auth.service';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, TranslatePipe, AsyncPipe],
  template: `
    <nav class="w-60 min-w-60 h-screen flex flex-col bg-surface border-r border-border/70">

      <!-- Wordmark -->
      <a routerLink="/" class="block px-5 pt-7 pb-6" aria-label="mouna home">
        <img [src]="theme.isDark() ? 'brand/mouna-logo-dark.png' : 'brand/mouna-logo.png'" alt="mouna" class="w-full h-auto" />
      </a>

      <!-- Nav -->
      <div class="flex-1 px-3 space-y-0.5">
        <a routerLink="/" routerLinkActive="nav-active" [routerLinkActiveOptions]="{ exact: true }"
          class="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-ink-muted hover:text-ink hover:bg-surface-sunk transition-colors">
          <svg class="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.6">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3 10.5 12 4l9 6.5V20a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1v-9.5Z" />
          </svg>
          <span>{{ 'nav.dashboard' | translate }}</span>
        </a>
        <a routerLink="/coach" routerLinkActive="nav-active"
          class="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-ink-muted hover:text-ink hover:bg-surface-sunk transition-colors">
          <svg class="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.6">
            <path stroke-linecap="round" stroke-linejoin="round" d="M8 13V5.5a1.5 1.5 0 0 1 3 0V12m0-1.5v-6a1.5 1.5 0 0 1 3 0V12m0-5.5a1.5 1.5 0 0 1 3 0V14a7 7 0 0 1-7 7h-.6a6 6 0 0 1-4.7-2.3L2.6 15.4a1.6 1.6 0 0 1 2.4-2.1L8 16" />
          </svg>
          <span>{{ 'nav.coach' | translate }}</span>
        </a>
        <a routerLink="/dictionary" routerLinkActive="nav-active"
          class="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-ink-muted hover:text-ink hover:bg-surface-sunk transition-colors">
          <svg class="w-[18px] h-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.6">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 6.5C10.5 5.3 8.4 4.5 6 4.5c-1 0-2 .1-3 .4v14c1-.3 2-.4 3-.4 2.4 0 4.5.8 6 2m0-14c1.5-1.2 3.6-2 6-2 1 0 2 .1 3 .4v14c-1-.3-2-.4-3-.4-2.4 0-4.5.8-6 2m0-14v14" />
          </svg>
          <span>{{ 'nav.dictionary' | translate }}</span>
        </a>
      </div>

      <!-- Account + preferences -->
      <div class="px-4 pb-5 pt-4 border-t border-border/70 space-y-3">
        @if (auth.user$ | async; as user) {
          <div class="flex items-center gap-3 px-1">
            <span class="shrink-0 w-8 h-8 rounded-full bg-surface-sunk text-ink-muted text-xs font-semibold flex items-center justify-center uppercase">{{ initials(user.name || user.email) }}</span>
            <div class="min-w-0 flex-1">
              <p class="text-sm text-ink truncate">{{ user.name || 'You' }}</p>
              <p class="text-[11px] text-ink-faint truncate">{{ user.email }}</p>
            </div>
            <button (click)="onLogout()" title="Log out" aria-label="Log out"
              class="shrink-0 text-ink-faint hover:text-ink transition-colors p-1">
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.6">
                <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0 0 13.5 3h-6a2.25 2.25 0 0 0-2.25 2.25v13.5A2.25 2.25 0 0 0 7.5 21h6a2.25 2.25 0 0 0 2.25-2.25V15M12 9l-3 3m0 0 3 3m-3-3h12.75" />
              </svg>
            </button>
          </div>
        }
        <div class="flex items-center gap-2 px-1">
          <button (click)="theme.toggle()" [attr.aria-label]="theme.isDark() ? 'Switch to light mode' : 'Switch to dark mode'"
            class="h-8 px-2.5 rounded-lg border border-border/70 text-ink-muted hover:text-ink flex items-center gap-1.5 text-xs transition-colors">
            @if (theme.isDark()) {
              <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" /></svg>
              Dark
            } @else {
              <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="4" /><path stroke-linecap="round" d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
              Light
            }
          </button>
          <select
            [value]="i18n.lang()"
            (change)="i18n.setLang($any($event.target).value)"
            aria-label="Language"
            class="h-8 flex-1 min-w-0 bg-transparent border border-border/70 rounded-lg px-2 text-xs text-ink-muted focus:outline-none focus:border-accent/60 cursor-pointer">
            @for (entry of langEntries; track entry[0]) {
              <option [value]="entry[0]">{{ entry[1] }}</option>
            }
          </select>
        </div>
      </div>
    </nav>
  `,
})
export class NavbarComponent {
  theme = inject(ThemeService);
  i18n = inject(I18nService);
  auth = inject(AuthService);
  private router = inject(Router);
  langEntries = Object.entries(this.i18n.languages);

  initials(name: string): string {
    const parts = (name || '').split(/[\s@._-]+/).filter(Boolean);
    return ((parts[0]?.[0] ?? '') + (parts[1]?.[0] ?? '')).slice(0, 2) || '·';
  }

  onLogout(): void {
    this.auth.logout();
    this.router.navigateByUrl('/login');
  }
}
