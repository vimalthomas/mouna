import { Injectable, signal } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  private _isDark = signal(true);
  isDark = this._isDark.asReadonly();

  constructor() {
    const saved = localStorage.getItem('theme');
    this._apply(saved === 'light' ? 'light' : 'dark');
  }

  toggle(): void {
    this._apply(this._isDark() ? 'light' : 'dark');
  }

  private _apply(theme: 'dark' | 'light'): void {
    this._isDark.set(theme === 'dark');
    const root = document.documentElement;
    root.setAttribute('data-theme', theme);
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }
}
