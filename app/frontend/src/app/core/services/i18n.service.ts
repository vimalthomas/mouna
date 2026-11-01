import { Injectable, signal } from '@angular/core';
import { TRANSLATIONS } from '../i18n/translations';

export const SUPPORTED_LANGUAGES: Record<string, string> = {
  en: 'English',
  es: 'Español',
  fr: 'Français',
  ta: 'தமிழ்',
};

@Injectable({ providedIn: 'root' })
export class I18nService {
  private _lang = signal('en');
  lang = this._lang.asReadonly();
  languages = SUPPORTED_LANGUAGES;

  constructor() {
    const saved = localStorage.getItem('lang') ?? 'en';
    this._lang.set(SUPPORTED_LANGUAGES[saved] ? saved : 'en');
  }

  setLang(lang: string): void {
    if (!SUPPORTED_LANGUAGES[lang]) return;
    this._lang.set(lang);
    localStorage.setItem('lang', lang);
  }

  t(key: string): string {
    return TRANSLATIONS[this._lang()]?.[key] ?? TRANSLATIONS['en']?.[key] ?? key;
  }
}
