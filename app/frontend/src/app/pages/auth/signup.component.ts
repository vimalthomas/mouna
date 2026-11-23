import { Component, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../../core/services/auth.service';
import { ThemeService } from '../../core/services/theme.service';

@Component({
  selector: 'app-signup',
  standalone: true,
  imports: [FormsModule, RouterLink],
  template: `
    <div class="min-h-screen flex items-center justify-center bg-background px-4">
      <div class="w-full max-w-sm">
        <div class="text-center mb-8">
          <img [src]="theme.isDark() ? 'brand/mouna-logo-dark.png' : 'brand/mouna-logo.png'" alt="mouna" class="h-24 w-auto mx-auto mb-2" />
          <p class="text-sm text-ink-muted mt-1">Create an account to start learning</p>
        </div>

        <form (ngSubmit)="onSubmit()" class="bg-surface border border-border/50 rounded-2xl p-6 space-y-4">
          @if (error) {
            <div class="rounded-lg bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-500">
              {{ error }}
            </div>
          }

          <div>
            <label class="block text-xs font-medium text-ink-muted mb-1.5">Name</label>
            <input
              type="text"
              [(ngModel)]="name"
              name="name"
              required
              autocomplete="name"
              class="w-full h-11 bg-background border border-border/50 rounded-xl px-4 text-sm text-ink focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/25 transition-all"
            />
          </div>

          <div>
            <label class="block text-xs font-medium text-ink-muted mb-1.5">Email</label>
            <input
              type="email"
              [(ngModel)]="email"
              name="email"
              required
              autocomplete="email"
              class="w-full h-11 bg-background border border-border/50 rounded-xl px-4 text-sm text-ink focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/25 transition-all"
            />
          </div>

          <div>
            <label class="block text-xs font-medium text-ink-muted mb-1.5">Password</label>
            <input
              type="password"
              [(ngModel)]="password"
              name="password"
              required
              minlength="8"
              autocomplete="new-password"
              class="w-full h-11 bg-background border border-border/50 rounded-xl px-4 text-sm text-ink focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/25 transition-all"
            />
            <p class="text-xs text-ink-muted mt-1">At least 8 characters</p>
          </div>

          <button
            type="submit"
            [disabled]="loading || !name || !email || password.length < 8"
            class="w-full h-11 bg-accent hover:bg-accent/90 disabled:opacity-50 text-white rounded-xl font-medium transition-all"
          >
            {{ loading ? 'Creating account...' : 'Sign Up' }}
          </button>
        </form>

        <p class="text-center text-sm text-ink-muted mt-5">
          Already have an account?
          <a routerLink="/login" class="text-accent hover:text-accent/80 font-medium">Sign in</a>
        </p>
      </div>
    </div>
  `,
})
export class SignupComponent {
  theme = inject(ThemeService);
  private authService = inject(AuthService);
  private router = inject(Router);

  name = '';
  email = '';
  password = '';
  loading = false;
  error = '';

  onSubmit(): void {
    if (!this.name || !this.email || this.password.length < 8) return;
    this.loading = true;
    this.error = '';

    this.authService.signup({ name: this.name, email: this.email, password: this.password }).subscribe({
      next: () => this.router.navigateByUrl('/'),
      error: (err) => {
        this.loading = false;
        this.error = err.status === 409 ? 'That email is already registered' : 'Something went wrong. Please try again.';
      },
    });
  }
}
