import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import { SkinData } from '../models/sign.model';

/** v5 Practice tab: stories from learned signs, the sentence avatar, sign-by-sign grading, history. */
export interface PracticeSign { word: string; description: string; video_url: string | null; learned: boolean; tries: number; }
export interface PracticePool { learned: PracticeSign[]; practising: PracticeSign[]; min_signs: number; unlocked: boolean; ai: boolean; demo?: boolean; }
export interface StorySentence { english: string; gloss: string[]; }
export interface Story {
  id: number; title: string; sentences: StorySentence[]; structure: string[];
  signs: PracticeSign[]; stretch: string | null; model: string; seconds: number;
}
export interface SentenceClip extends SkinData { segments: { sign: string; start: number; end: number }[]; }
export interface SignScore {
  sign: string; verdict: 'good' | 'needs_work' | 'wrong_sign' | null; similarity: number | null; graded: boolean;
  hands_seen: number; frames: number; feedback: string;
  parts: Record<string, { label: string; level: 'fluent' | 'close' | 'needs_work' | null; score: number | null; reliable: boolean }>;
}
export interface PracticeSummary {
  sentences: { id: number; english: string; gloss: string; signs_good: number; signs_total: number; all_good: boolean; at: string }[];
  sentences_total: number; sentences_all_good: number;
  words: { word: string; in_sentences: number; good: number; last: string | null }[];
}

@Injectable({ providedIn: 'root' })
export class PracticeApiService {
  private http = inject(HttpClient);
  private base = `${environment.apiBaseUrl}/practice`;

  pool(): Observable<PracticePool> { return this.http.get<PracticePool>(`${this.base}/pool`); }
  story(stretch: boolean): Observable<Story> { return this.http.post<Story>(`${this.base}/story`, { stretch }); }
  sentence(signs: string[]): Observable<SentenceClip> {
    return this.http.get<SentenceClip>(`${this.base}/sentence`, { params: { signs: signs.join(',') } });
  }
  score(sign: string, frames: string[]): Observable<SignScore> { return this.http.post<SignScore>(`${this.base}/score`, { sign, frames }); }
  saveAttempt(storyId: number | null, english: string, results: { sign: string; verdict: string | null }[]) {
    return this.http.post<{ id: number; signs_good: number; signs_total: number; all_good: boolean }>(
      `${this.base}/sentence-attempts`, { story_id: storyId, english, results });
  }
  summary(): Observable<PracticeSummary> { return this.http.get<PracticeSummary>(`${this.base}/summary`); }
}
