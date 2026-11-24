import { Component, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { SignSkinComponent } from '../../shared/components/sign-skin/sign-skin.component';
import { AuthService } from '../../core/services/auth.service';
import { ThemeService } from '../../core/services/theme.service';

/** v5 Home for people who aren't signed in: what Mouna is for, how it works, how
 * the feedback stays honest, and a way in. Signed-in learners land on the dashboard. */
@Component({
  selector: 'app-welcome',
  standalone: true,
  imports: [RouterLink, SignSkinComponent],
  template: `
    <div class="min-h-screen bg-background text-ink">
      <header class="max-w-6xl mx-auto px-6 py-5 flex items-center justify-between">
        <img [src]="theme.isDark() ? 'brand/mouna-logo-dark.png' : 'brand/mouna-logo.png'" alt="mouna" class="h-12 w-auto" />
        <div class="flex items-center gap-2">
          <button (click)="theme.toggle()" class="h-9 px-3 rounded-lg border border-border/70 text-xs text-ink-muted hover:text-ink">{{ theme.isDark() ? 'Dark' : 'Light' }}</button>
          @if (signedIn) {
            <a routerLink="/" class="h-9 px-4 rounded-lg bg-accent text-white text-sm font-medium flex items-center">Go to your dashboard</a>
          } @else {
            <a routerLink="/login" class="h-9 px-4 rounded-lg border border-border/70 text-sm text-ink flex items-center">Sign in</a>
          }
        </div>
      </header>

      <main class="max-w-6xl mx-auto px-6">
        <!-- purpose -->
        <section class="py-14 md:py-20 grid grid-cols-1 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] gap-10 items-center">
          <div>
          <p class="text-xs font-semibold uppercase tracking-[0.18em] text-accent mb-4">voice of the unheard</p>
          <h1 class="font-display text-4xl md:text-5xl leading-tight text-ink">Learn American Sign Language, one sign at a time, with feedback you can trust.</h1>
          <p class="text-lg text-ink-muted mt-5 leading-relaxed">Mouna helps hearing people take their first steps in ASL so they can meet Deaf people a little closer to halfway.
            You watch fluent signers, sign in front of your camera, and get specific, honest feedback on each part of the sign.</p>
          <div class="flex flex-wrap gap-3 mt-8">
            <a [routerLink]="signedIn ? '/' : '/signup'" class="h-12 px-6 rounded-xl bg-accent text-white font-medium flex items-center">{{ signedIn ? 'Continue learning' : 'Start learning, free' }}</a>
            <a href="#how" class="h-12 px-6 rounded-xl border border-border/70 text-ink flex items-center">How it works</a>
          </div>
          <p class="text-xs text-ink-faint mt-4">A practice companion, not a replacement for Deaf teachers and the Deaf community.</p>
          </div>
          <div>
            <app-sign-skin word="who" show="all" label="WHO, signed by the 3D avatar" />
            <p class="text-xs text-ink-faint mt-2 text-center">Drag to turn it. Every sign can be watched from any angle, slowed down.</p>
          </div>
        </section>

        <!-- how it works -->
        <section id="how" class="py-12 border-t border-border/60">
          <h2 class="font-display text-3xl text-ink mb-8">How it works</h2>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-5">
            @for (s of steps; track s.title; let i = $index) {
              <div class="rounded-2xl border border-border/60 bg-surface p-6">
                <span class="w-8 h-8 rounded-full bg-accent text-white text-sm font-semibold flex items-center justify-center">{{ i + 1 }}</span>
                <h3 class="text-lg font-semibold text-ink mt-4">{{ s.title }}</h3>
                <p class="text-sm text-ink-muted mt-2 leading-relaxed">{{ s.text }}</p>
              </div>
            }
          </div>
        </section>

        <!-- honest feedback -->
        <section class="py-12 border-t border-border/60 grid grid-cols-1 md:grid-cols-2 gap-10 items-start">
          <div>
            <h2 class="font-display text-3xl text-ink">Feedback that says how sure it is</h2>
            <p class="text-ink-muted mt-4 leading-relaxed">Every sign is built from five parts. Mouna compares each part of your attempt with how fluent signers do that sign,
              not with one video. A part is marked <strong class="text-emerald-500">trusted</strong> only after it passed a test on signers it had never seen;
              otherwise it's shown as an <strong class="text-ink">estimate</strong>, a hint rather than a mistake.</p>
            <p class="text-ink-muted mt-3 leading-relaxed">AI helps in two tested places: it words the coach's notes, and it double-checks whether you used one hand or two.
              The grade itself is decided by Mouna's own measurements.</p>
          </div>
          <div class="rounded-2xl border border-border/60 bg-surface divide-y divide-border/50">
            @for (p of parts; track p.name) {
              <div class="px-5 py-3 flex items-center justify-between gap-4">
                <div><p class="text-sm font-semibold text-ink">{{ p.name }}</p><p class="text-xs text-ink-muted">{{ p.what }}</p></div>
              </div>
            }
          </div>
        </section>

        <!-- what's inside -->
        <section class="py-12 border-t border-border/60">
          <h2 class="font-display text-3xl text-ink mb-8">What's inside</h2>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            @for (f of features; track f.title) {
              <div class="rounded-2xl border border-border/60 bg-surface p-5">
                <p class="font-display text-3xl text-accent">{{ f.big }}</p>
                <p class="text-sm font-semibold text-ink mt-2">{{ f.title }}</p>
                <p class="text-xs text-ink-muted mt-1 leading-relaxed">{{ f.text }}</p>
              </div>
            }
          </div>
        </section>

        <section class="py-14 border-t border-border/60 text-center">
          <h2 class="font-display text-3xl text-ink">Ready for your first sign?</h2>
          <a [routerLink]="signedIn ? '/' : '/signup'" class="inline-flex h-12 px-6 mt-6 rounded-xl bg-accent text-white font-medium items-center">{{ signedIn ? 'Continue learning' : 'Start learning, free' }}</a>
        </section>
      </main>

      <footer class="border-t border-border/60">
        <div class="max-w-6xl mx-auto px-6 py-6 text-xs text-ink-faint leading-relaxed">
          A non-commercial research project. Signs from public research datasets (WLASL, ASL Citizen, MS-ASL); sign descriptions checked against ASL-LEX.
        </div>
      </footer>
    </div>
  `,
})
export class WelcomeComponent {
  theme = inject(ThemeService);
  private auth = inject(AuthService);
  readonly signedIn = this.auth.isAuthenticated;
  readonly steps = [
    { title: 'Learn', text: 'Watch a fluent signer and a 3D avatar from any angle. Short prep checks get your hand shape and position right, then you record the sign and see what to fix.' },
    { title: 'Practice', text: 'The signs you have learned come back as short stories, shown in English and in ASL order. Watch the avatar sign them, then sign them yourself, sign by sign.' },
    { title: 'Progress', text: 'Your dashboard keeps the words you have learned and the sentences you have signed well, and brings back the signs that need another look.' },
  ];
  readonly parts = [
    { name: 'Handshape', what: 'which fingers are bent or straight, and the thumb' },
    { name: 'Location', what: 'where the hand is next to your face and body' },
    { name: 'Palm orientation', what: 'which way the palm faces' },
    { name: 'Movement', what: 'the path your hand takes over time' },
    { name: 'Facial expression', what: 'part of many signs (shown, not yet scored)' },
  ];
  readonly features = [
    { big: '41', title: 'Practice signs', text: 'Everyday words, each checked against fluent signers.' },
    { big: '3D', title: 'Avatar', text: 'Face and hands you can turn and slow down.' },
    { big: '5', title: 'Parts of a sign', text: 'Scored separately, so feedback is specific.' },
    { big: 'AI', title: 'Coach', text: 'Plain-language notes and answers about the sign.' },
  ];
}
