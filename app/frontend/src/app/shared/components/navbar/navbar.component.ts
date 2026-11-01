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
    <nav class="w-64 min-w-64 h-screen flex flex-col border-r border-border"
         [style]="navStyle">

      <!-- Logo -->
      <div class="p-6 border-b border-border">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-xl flex items-center justify-center" [style]="logoIconStyle">
            <svg class="w-6 h-6 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
              <path stroke-linecap="round" stroke-linejoin="round" d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
            </svg>
          </div>
          <div>
            <h1 class="text-lg font-semibold tracking-tight text-accent">
              MOUNA
            </h1>
            <p class="text-xs text-ink-muted">{{ 'nav.subtitle' | translate }}</p>
          </div>
        </div>
      </div>

      <!-- Nav Links -->
      <div class="flex-1 py-4 px-3 space-y-1">
        <a routerLink="/" routerLinkActive="nav-active"
          [routerLinkActiveOptions]="{ exact: true }"
          class="flex items-center gap-3 px-4 py-2.5 rounded-lg text-ink-muted hover:bg-accent/10 hover:text-accent transition-all duration-200 text-sm">
          <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" />
          </svg>
          <span class="font-medium">{{ 'nav.dashboard' | translate }}</span>
        </a>

        <a routerLink="/dictionary" routerLinkActive="nav-active"
          class="flex items-center gap-3 px-4 py-2.5 rounded-lg text-ink-muted hover:bg-accent/10 hover:text-accent transition-all duration-200 text-sm">
          <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
          </svg>
          <span class="font-medium">{{ 'nav.dictionary' | translate }}</span>
        </a>

        <a routerLink="/coach" routerLinkActive="nav-active"
          class="flex items-center gap-3 px-4 py-2.5 rounded-lg text-ink-muted hover:bg-accent/10 hover:text-accent transition-all duration-200 text-sm">
          <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
          </svg>
          <span class="font-medium">{{ 'nav.coach' | translate }}</span>
        </a>
      </div>

      <!-- Bottom controls: user + theme + language -->
      <div class="p-4 border-t border-border space-y-3">
        <!-- User / logout -->
        @if (auth.user$ | async; as user) {
          <div class="flex items-center justify-between px-2 gap-2">
            <div class="min-w-0">
              <p class="text-sm text-ink font-medium truncate">{{ user.name }}</p>
              <p class="text-xs text-ink-muted truncate">{{ user.email }}</p>
            </div>
            <button (click)="onLogout()" title="Log out"
              class="shrink-0 text-ink-muted hover:text-accent transition-colors p-1.5">
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15M12 9l-3 3m0 0l3 3m-3-3h12.75" />
              </svg>
            </button>
          </div>
        }

        <!-- Theme toggle -->
        <div class="flex items-center justify-between px-2">
          <span class="text-xs text-ink-muted">{{ theme.isDark() ? 'Dark' : 'Light' }}</span>
          <button (click)="theme.toggle()"
            class="relative w-11 h-6 rounded-full transition-colors duration-300 focus:outline-none"
            [class]="theme.isDark() ? 'bg-accent' : 'bg-accent/60'">
            <span class="absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-lg transition-transform duration-300 flex items-center justify-center"
              [class.translate-x-5]="!theme.isDark()">
              @if (theme.isDark()) {
                <svg class="w-3 h-3 text-accent" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                </svg>
              } @else {
                <svg class="w-3 h-3 text-amber-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clip-rule="evenodd" />
                </svg>
              }
            </span>
          </button>
        </div>

        <!-- Language selector -->
        <div class="px-2">
          <select
            [value]="i18n.lang()"
            (change)="i18n.setLang($any($event.target).value)"
            class="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-xs text-ink-muted focus:outline-none focus:border-accent/50 cursor-pointer">
            @for (entry of langEntries; track entry[0]) {
              <option [value]="entry[0]">{{ entry[1] }}</option>
            }
          </select>
        </div>

        <div class="flex items-center gap-2 px-2">
          <div class="w-2 h-2 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50"></div>
          <p class="text-xs text-ink-muted">{{ 'nav.version' | translate }}</p>
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

  onLogout(): void {
    this.auth.logout();
    this.router.navigateByUrl('/login');
  }

  get navStyle(): string {
    return this.theme.isDark()
      ? 'background: #1c1c1e;'
      : 'background: #f2f2f7;';
  }

  get logoIconStyle(): string {
    return this.theme.isDark()
      ? 'background: rgba(10,132,255,0.18); border: 1px solid rgba(10,132,255,0.35);'
      : 'background: rgba(0,122,255,0.1); border: 1px solid rgba(0,122,255,0.25);';
  }
}
