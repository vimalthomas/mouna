import { Component, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [FormsModule, RouterLink],
  template: `
    <div class="min-h-screen flex items-center justify-center bg-background px-4">
      <div class="w-full max-w-sm">
        <div class="text-center mb-8">
          <h1 class="text-2xl font-semibold tracking-tight text-accent">
            MOUNA
          </h1>
          <p class="text-sm text-ink-muted mt-1">Sign in to continue your practice</p>
        </div>

        <form (ngSubmit)="onSubmit()" class="bg-surface border border-border/50 rounded-2xl p-6 space-y-4">
          @if (error) {
            <div class="rounded-lg bg-red-500/10 border border-red-500/30 px-3 py-2 text-sm text-red-500">
              {{ error }}
            </div>
          }

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
              autocomplete="current-password"
              class="w-full h-11 bg-background border border-border/50 rounded-xl px-4 text-sm text-ink focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/25 transition-all"
            />
          </div>

          <button
            type="submit"
            [disabled]="loading || !email || !password"
            class="w-full h-11 bg-accent hover:bg-accent/90 disabled:opacity-50 text-white rounded-xl font-medium transition-all"
          >
            {{ loading ? 'Signing in...' : 'Sign In' }}
          </button>
        </form>

        <p class="text-center text-sm text-ink-muted mt-5">
          Don't have an account?
          <a routerLink="/signup" class="text-accent hover:text-accent/80 font-medium">Sign up</a>
        </p>
      </div>
    </div>
  `,
})
export class LoginComponent {
  private authService = inject(AuthService);
  private router = inject(Router);
  private route = inject(ActivatedRoute);

  email = '';
  password = '';
  loading = false;
  error = '';

  onSubmit(): void {
    if (!this.email || !this.password) return;
    this.loading = true;
    this.error = '';

    this.authService.login({ email: this.email, password: this.password }).subscribe({
      next: () => {
        const returnUrl = this.route.snapshot.queryParamMap.get('returnUrl') || '/';
        this.router.navigateByUrl(returnUrl);
      },
      error: (err) => {
        this.loading = false;
        this.error = err.status === 401 ? 'Invalid email or password' : 'Something went wrong. Please try again.';
      },
    });
  }
}
