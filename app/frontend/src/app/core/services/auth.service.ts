import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, tap } from 'rxjs';
import { environment } from '../../../environments/environment';
import { AuthResponse, LoginRequest, SignupRequest, UserProfile } from '../models/auth.model';

const ACCESS_TOKEN_KEY = 'mouna_access_token';
const REFRESH_TOKEN_KEY = 'mouna_refresh_token';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private http = inject(HttpClient);
  private baseUrl = environment.apiBaseUrl;

  private userSubject = new BehaviorSubject<UserProfile | null>(null);
  user$ = this.userSubject.asObservable();

  get accessToken(): string | null {
    return localStorage.getItem(ACCESS_TOKEN_KEY);
  }

  get refreshToken(): string | null {
    return localStorage.getItem(REFRESH_TOKEN_KEY);
  }

  get isAuthenticated(): boolean {
    return !!this.accessToken;
  }

  signup(payload: SignupRequest): Observable<AuthResponse> {
    return this.http
      .post<AuthResponse>(`${this.baseUrl}/auth/signup`, payload)
      .pipe(tap(res => this.storeSession(res)));
  }

  login(payload: LoginRequest): Observable<AuthResponse> {
    return this.http
      .post<AuthResponse>(`${this.baseUrl}/auth/login`, payload)
      .pipe(tap(res => this.storeSession(res)));
  }

  refresh(): Observable<AuthResponse> {
    return this.http
      .post<AuthResponse>(`${this.baseUrl}/auth/refresh`, { refresh_token: this.refreshToken })
      .pipe(tap(res => this.storeSession(res)));
  }

  loadMe(): Observable<UserProfile> {
    return this.http
      .get<UserProfile>(`${this.baseUrl}/users/me`)
      .pipe(tap(user => this.userSubject.next(user)));
  }

  logout(): void {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    this.userSubject.next(null);
  }

  private storeSession(res: AuthResponse): void {
    localStorage.setItem(ACCESS_TOKEN_KEY, res.access_token);
    localStorage.setItem(REFRESH_TOKEN_KEY, res.refresh_token);
    this.userSubject.next(res.user);
  }
}
