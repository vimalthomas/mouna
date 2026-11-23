import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, shareReplay } from 'rxjs';
import { map } from 'rxjs/operators';
import { environment } from '../../../environments/environment';
import { ReferenceBuckets, Sign, SignListResponse, VideoResponse, SkeletonFrame, SignCatalogEntry } from '../models/sign.model';

@Injectable({ providedIn: 'root' })
export class SignsApiService {
  private http = inject(HttpClient);
  private baseUrl = environment.apiBaseUrl;

  private signs$ = this.http
    .get<SignListResponse>(`${this.baseUrl}/signs`)
    .pipe(shareReplay(1));

  getSigns(): Observable<Sign[]> {
    return this.signs$.pipe(map(res => res.signs));
  }

  getVideo(word: string): Observable<VideoResponse> {
    return this.http.get<VideoResponse>(`${this.baseUrl}/video/${word}`);
  }

  getReferenceBuckets(word: string): Observable<ReferenceBuckets> {
    return this.http.get<ReferenceBuckets>(`${this.baseUrl}/signs/${word}/reference-buckets`);
  }

  /** ASL-CDI category, ASL-LEX frequency and "essential" flag per sign. */
  getSignCatalog(): Observable<Record<string, SignCatalogEntry>> {
    return this.http.get<Record<string, SignCatalogEntry>>(`${this.baseUrl}/sign-catalog`);
  }

  getVariations(word: string): Observable<{ video_url: string }[]> {
    return this.http.get<{ video_url: string }[]>(`${this.baseUrl}/signs/${word}/variations`);
  }

  getReferenceSkeleton(word: string): Observable<SkeletonFrame[]> {
    return this.http.get<SkeletonFrame[]>(`${this.baseUrl}/signs/${word}/reference-skeleton`);
  }

  /** Signs that can be graded (have a reviewed reference recording). */
  getSignsWithReference(): Observable<Set<string>> {
    return this.http.get<string[]>(`${this.baseUrl}/signs-with-reference`).pipe(map(words => new Set(words)));
  }

  getSignsWithVideo(): Observable<Set<string>> {
    return this.http
      .get<string[]>(`${this.baseUrl}/signs-with-video`)
      .pipe(map(list => new Set(list)), shareReplay(1));
  }

  getCategories(): Observable<string[]> {
    return this.getSigns().pipe(
      map(signs => [...new Set(signs.map(s => s.category))].sort())
    );
  }
}
