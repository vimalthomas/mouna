import { Routes } from '@angular/router';
import { authGuard } from './core/guards/auth.guard';
import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from './core/services/auth.service';

/** Guest mode (demo deployment): there is no sign-in or sign-up, go straight in. */
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
    path: '',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/dashboard/dashboard.component').then(m => m.DashboardComponent),
  },
  {
    path: 'dictionary',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/dictionary/dictionary.component').then(m => m.DictionaryComponent),
  },
  {
    path: 'coach',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/coach/coach.component').then(m => m.CoachComponent),
  },
  {
    path: 'debug',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./pages/debug/debug.component').then(m => m.DebugComponent),
  },
  { path: '**', redirectTo: '' },
];
