import { Routes } from '@angular/router';
import { authGuard } from './core/guards/auth.guard';

export const routes: Routes = [
  {
    path: 'login',
    loadComponent: () =>
      import('./pages/auth/login.component').then(m => m.LoginComponent),
  },
  {
    path: 'signup',
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
