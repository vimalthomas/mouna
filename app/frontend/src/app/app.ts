import { Component, inject, signal } from '@angular/core';
import { NavigationEnd, Router, RouterOutlet } from '@angular/router';
import { filter } from 'rxjs';
import { NavbarComponent } from './shared/components/navbar/navbar.component';
import { ThemeService } from './core/services/theme.service';
import { AuthService } from './core/services/auth.service';

const STANDALONE_ROUTES = ['/login', '/signup', '/welcome'];  // no sidebar

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, NavbarComponent],
  template: `
    @if (isStandalone()) {
      <router-outlet />
    } @else {
      <div class="flex h-screen bg-background">
        <app-navbar />
        <main class="flex-1 overflow-y-auto p-8">
          <router-outlet />
        </main>
      </div>
    }
  `,
})
export class App {
  private theme = inject(ThemeService);
  private router = inject(Router);
  private authService = inject(AuthService);

  isStandalone = signal(STANDALONE_ROUTES.includes(this.router.url));

  constructor() {
    this.router.events
      .pipe(filter((event): event is NavigationEnd => event instanceof NavigationEnd))
      .subscribe(event => {
        this.isStandalone.set(STANDALONE_ROUTES.includes(event.urlAfterRedirects.split('?')[0]));
      });

    if (this.authService.guestMode) {
      this.authService.loadMe().subscribe({ error: () => {} });   // the shared guest
    } else if (this.authService.isAuthenticated) {
      this.authService.loadMe().subscribe({ error: () => this.authService.logout() });
    }
  }
}
