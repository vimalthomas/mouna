import { Component, OnDestroy, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { UserProfile } from '../../core/models/auth.model';
import { AuthService } from '../../core/services/auth.service';
import { I18nService } from '../../core/services/i18n.service';
import { ThemeService } from '../../core/services/theme.service';
import { LookPickerComponent } from '../../shared/components/look-picker/look-picker.component';
import { SignSkinComponent } from '../../shared/components/sign-skin/sign-skin.component';

/** v5 Profile: things set once -- name, the hand you sign with, the avatar's look, theme and language. */
@Component({
  selector: 'app-profile',
  standalone: true,
  imports: [FormsModule, LookPickerComponent, SignSkinComponent],
  template: `
    <div class="max-w-4xl mx-auto">
      <h1 class="font-display text-3xl text-ink mb-1">Profile</h1>
      <p class="text-sm text-ink-muted mb-6">Set these once; they apply everywhere in Mouna.</p>

      <section class="rounded-xl border border-border/60 bg-surface p-5 mb-4">
        <h2 class="text-sm font-semibold text-ink mb-3">You</h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <label class="block">
            <span class="text-xs text-ink-muted">Name</span>
            <div class="flex gap-2 mt-1">
              <input [(ngModel)]="name" [disabled]="auth.guestMode" maxlength="80"
                class="flex-1 bg-background border border-border/60 rounded-lg px-3 h-10 text-sm text-ink focus:outline-none focus:border-accent/60 disabled:opacity-50" />
              <button (click)="saveName()" [disabled]="auth.guestMode || !name.trim() || name === user?.name"
                class="px-4 h-10 rounded-lg bg-accent text-white text-sm disabled:opacity-40">Save</button>
            </div>
            @if (nameSaved) { <span class="text-[11px] text-emerald-500">Saved</span> }
          </label>
          <div>
            <span class="text-xs text-ink-muted">Email</span>
            <p class="text-sm text-ink mt-3">{{ auth.guestMode ? 'Demo, no sign-in' : user?.email }}</p>
          </div>
        </div>
      </section>

      <section class="rounded-xl border border-border/60 bg-surface p-5 mb-4">
        <h2 class="text-sm font-semibold text-ink">The hand you sign with</h2>
        <p class="text-xs text-ink-muted mt-1 mb-3">Left-handed signers lead with the left hand. If that's you, the example signers in Learn and Practice are shown mirrored, so you can copy them directly. Grading works for either hand.</p>
        <div class="flex gap-2" role="radiogroup" aria-label="Dominant hand">
          @for (h of hands; track h.id) {
            <button role="radio" [attr.aria-checked]="hand === h.id" (click)="setHand(h.id)" [disabled]="auth.guestMode"
              class="px-4 h-10 rounded-lg border text-sm font-medium disabled:opacity-50"
              [class]="hand === h.id ? 'bg-accent text-white border-accent' : 'bg-surface-sunk text-ink border-border/60'">{{ h.label }}</button>
          }
          @if (handSaved) { <span class="text-[11px] text-emerald-500 self-center ml-2">Saved</span> }
        </div>
      </section>

      <section class="rounded-xl border border-border/60 bg-surface p-5 mb-4">
        <h2 class="text-sm font-semibold text-ink mb-3">Avatar look</h2>
        <div class="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_280px] gap-5 items-start">
          <div><app-look-picker [alwaysOpen]="true" />
            <p class="text-xs text-ink-muted mt-3">The 3D avatar that shows signs in Learn and signs your stories in Practice. You can also change this from the "Avatar look" button next to any 3D view.</p></div>
          <app-sign-skin word="who" show="face" label="Preview" />
        </div>
      </section>

      <section class="rounded-xl border border-border/60 bg-surface p-5">
        <h2 class="text-sm font-semibold text-ink mb-3">Display</h2>
        <div class="flex flex-wrap items-center gap-3">
          <button (click)="theme.toggle()" class="px-4 h-10 rounded-lg border border-border/60 bg-surface-sunk text-sm text-ink">{{ theme.isDark() ? 'Dark' : 'Light' }} theme</button>
          <select [value]="i18n.lang()" (change)="i18n.setLang($any($event.target).value)" aria-label="Language"
            class="h-10 bg-surface-sunk border border-border/60 rounded-lg px-3 text-sm text-ink">
            @for (e of langs; track e[0]) { <option [value]="e[0]">{{ e[1] }}</option> }
          </select>
        </div>
      </section>
    </div>
  `,
})
export class ProfileComponent implements OnDestroy {
  auth = inject(AuthService);
  theme = inject(ThemeService);
  i18n = inject(I18nService);
  readonly langs = Object.entries(this.i18n.languages);
  readonly hands: { id: 'right' | 'left'; label: string }[] = [{ id: 'right', label: 'Right hand' }, { id: 'left', label: 'Left hand' }];
  user: UserProfile | null = null;
  name = '';
  hand: 'right' | 'left' = 'right';
  nameSaved = false;
  handSaved = false;
  private sub: Subscription = this.auth.user$.subscribe(u => {
    this.user = u;
    if (u) { this.name = u.name; this.hand = u.dominant_hand ?? 'right'; }
  });

  ngOnDestroy(): void { this.sub.unsubscribe(); }

  saveName(): void {
    this.auth.updateMe({ name: this.name.trim() }).subscribe(() => { this.nameSaved = true; setTimeout(() => (this.nameSaved = false), 2000); });
  }

  setHand(h: 'right' | 'left'): void {
    this.hand = h;
    this.auth.updateMe({ dominant_hand: h }).subscribe(() => { this.handSaved = true; setTimeout(() => (this.handSaved = false), 2000); });
  }
}
