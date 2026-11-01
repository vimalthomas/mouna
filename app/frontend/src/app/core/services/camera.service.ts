import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, from, of } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';

@Injectable({ providedIn: 'root' })
export class CameraService {
  private streamSubject = new BehaviorSubject<MediaStream | null>(null);
  private errorSubject = new BehaviorSubject<string | null>(null);

  stream$ = this.streamSubject.asObservable();
  error$ = this.errorSubject.asObservable();

  startCamera(): Observable<MediaStream | null> {
    return from(
      navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      })
    ).pipe(
      tap(stream => {
        this.streamSubject.next(stream);
        this.errorSubject.next(null);
      }),
      catchError(err => {
        const message =
          err.name === 'NotAllowedError'
            ? 'Camera permission denied. Please allow camera access.'
            : err.name === 'NotFoundError'
              ? 'No camera found on this device.'
              : `Camera error: ${err.message}`;
        this.errorSubject.next(message);
        this.streamSubject.next(null);
        return of(null);
      })
    );
  }

  stopCamera(): void {
    const stream = this.streamSubject.value;
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      this.streamSubject.next(null);
    }
  }
}
