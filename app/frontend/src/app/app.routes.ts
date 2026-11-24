import { Routes } from '@angular/router';
import { authGuard } from './core/guards/auth.guard';
import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from './core/services/auth.service';

/** Guest mode (demo deployment): there is no sign-in or sign-up, go straight in. */
/** v5: signed-out visitors land on Home (/welcome), not the sign-in form. */
/** In the public demo (guest mode) everyone counts as signed in, so first-time visitors see Home once. */
const homeOrWelcome: CanActivateFn = () => {
  const auth = inject(AuthService);
  let seen = false;
  try { seen = localStorage.getItem('mouna_seen_welcome') === '1'; } catch { /* private mode */ }
  if (!auth.isAuthenticated || (auth.guestMode && !seen)) return inject(Router).createUrlTree(['/welcome']);
  return true;
};
const noLoginInGuestMode: CanActivateFn = () => inject(AuthService).guestMode ? inject(Router).createUrlTree(['/']) : true;

export const routes: Routes = [
  {
    path: 'login',
    canActivate: [noLoginInGuestMode],
    loadComponent: () =>
      import('./pages/auth/login.component').then(m => m.LoginComponent),
  },
  {
    path: 'signup',
    canActivate: [noLoginInGuestMode],
    loadComponent: () =>
      import('./pages/auth/signup.component').then(m => m.SignupComponent),
  },
  {
    path: 'welcome',
    loadComponent: () =>
      import('./pages/welcome/welcome.component').then(m => m.WelcomeComponent),
  },
  {
    path: '',
    canActivate: [homeOrWelcome],
    loadComponent: () =>
      import('./pages/dashboard/dashboard.component').then(m => m.DashboardComponent),
  },
  {
    path: 'learn',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/coach/coach.component').then(m => m.CoachComponent),
  },
  {
    path: 'practice',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/practice/practice.component').then(m => m.PracticeComponent),
  },
  {
    path: 'profile',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/profile/profile.component').then(m => m.ProfileComponent),
  },
  {
    path: 'dictionary',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/dictionary/dictionary.component').then(m => m.DictionaryComponent),
  },
  // v4 links (/coach?sign=...) still work: Learn is the same page
  { path: 'coach', redirectTo: 'learn' },
  {
    path: 'debug',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/debug/debug.component').then(m => m.DebugComponent),
  },
  { path: '**', redirectTo: '' },
];
