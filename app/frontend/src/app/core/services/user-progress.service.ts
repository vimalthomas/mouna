import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject } from 'rxjs';
import { map, tap } from 'rxjs/operators';
import { environment } from '../../../environments/environment';
import { UserStats, UserStatsResponse, WeakSign } from '../models/sign.model';

const EMPTY_STATS: UserStats = {
  totalSessions: 0,
  signsLearned: 0,
  accuracy: 0,
  streak: 0,
  weakSigns: [],
  signs: [],
};

@Injectable({ providedIn: 'root' })
export class UserProgressService {
  private http = inject(HttpClient);
  private baseUrl = environment.apiBaseUrl;

  private statsSubject = new BehaviorSubject<UserStats>(EMPTY_STATS);
  stats$ = this.statsSubject.asObservable();

  refresh() {
    return this.http.get<UserStatsResponse>(`${this.baseUrl}/users/me/stats`).pipe(
      map(res => ({
        totalSessions: res.total_sessions,
        signsLearned: res.signs_learned,
        accuracy: res.accuracy,
        streak: res.streak,
        weakSigns: res.weak_signs,
        signs: res.signs.map(s => ({
          sign: s.sign,
          totalAttempts: s.total_attempts,
          correctAttempts: s.correct_attempts,
          accuracy: s.accuracy,
          completed: s.completed,
          completedAt: s.completed_at,
        })),
      })),
      tap(stats => this.statsSubject.next(stats))
    );
  }

  getWeakSigns(): WeakSign[] {
    return this.statsSubject.value.weakSigns;
  }

  reset(): void {
    this.statsSubject.next(EMPTY_STATS);
  }
}
