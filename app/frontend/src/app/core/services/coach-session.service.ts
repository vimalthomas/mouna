import { Injectable, inject } from '@angular/core';
import { BehaviorSubject, Subject, filter, take, timeout, catchError, of } from 'rxjs';
import { environment } from '../../../environments/environment';
import { WebSocketService, ConnectionState, WsConnection } from './websocket.service';
import { CoachingMessage, PrepStep, StepResult } from '../models/sign.model';
import { AuthService } from './auth.service';

@Injectable({ providedIn: 'root' })
export class CoachSessionService {
  private wsService = inject(WebSocketService);
  private authService = inject(AuthService);
  private coachingSubject = new Subject<CoachingMessage>();
  private connection: WsConnection<CoachingMessage> | null = null;

  coaching$ = this.coachingSubject.asObservable();
  sessionState$ = new BehaviorSubject<ConnectionState>('disconnected');
  sessionId$ = new BehaviorSubject<string | null>(null);

  // Live keypoint status from backend
  keypointStatus$ = new Subject<CoachingMessage>();

  startSession(targetSign: string): void {
    const token = this.authService.accessToken;
    const url = `${environment.wsBaseUrl}/ws/coach-frames?token=${encodeURIComponent(token || '')}`;
    this.connection = this.wsService.connect<CoachingMessage>(url);

    this.connection.state$.subscribe(state =>
      this.sessionState$.next(state)
    );

    this.connection.messages$.subscribe((msg: any) => {
      if (msg.session_id) {
        this.sessionId$.next(msg.session_id);
      }
      if (msg.type === 'keypoint_status') {
        this.keypointStatus$.next(msg);
      } else {
        this.coachingSubject.next(msg);
      }
    });

    setTimeout(() => {
      this.connection?.send({
        type: 'start_session',
        target_sign: targetSign,
      });
    }, 500);
  }

  sendFrame(base64Frame: string): void {
    this.connection?.send({ type: 'frame', frame: base64Frame });
  }

  /** Clears the backend's frame buffer so the next prediction covers only
   * frames sent from now on (i.e. exactly the recorded attempt). */
  resetBuffer(): void {
    this.connection?.send({ type: 'reset_buffer' });
  }

  requestPrediction(targetSign: string): void {
    this.connection?.send({ type: 'predict', target_sign: targetSign });
  }

  /** Several manual CP checks answered from ONE window of frames (a prep
   * group like "hand shape & palm"). Resolves with step -> result, or null. */
  checkGroup(targetSign: string, steps: PrepStep[]): Promise<Record<string, StepResult> | null> {
    return new Promise(resolve => {
      this.coaching$
        .pipe(
          filter((msg: any) => msg.type === 'group_result' || msg.type === 'error'),
          take(1),
          timeout(8000),
          catchError(() => of(null)),
        )
        .subscribe((msg: any) => resolve(msg && msg.type === 'group_result' ? msg.results : null));
      this.connection?.send({ type: 'check_group', target_sign: targetSign, steps });
    });
  }

  /** One discrete prep-step check (handshape/location/orientation/hand_count/facial).
   * Resolves with the matching step_result, or null on timeout/error. */
  checkStep(targetSign: string, step: PrepStep): Promise<StepResult | null> {
    return new Promise(resolve => {
      this.coaching$
        .pipe(
          filter((msg: any) => msg.type === 'step_result' && msg.step === step || msg.type === 'error'),
          take(1),
          timeout(8000),
          catchError(() => of(null)),
        )
        .subscribe(msg => resolve(msg && msg.type === 'step_result' ? (msg as StepResult) : null));
      this.connection?.send({ type: 'check_step', target_sign: targetSign, step });
    });
  }

  sendChat(text: string): void {
    this.connection?.send({ type: 'chat', text });
  }

  changeTarget(targetSign: string): void {
    this.connection?.send({
      type: 'change_target',
      target_sign: targetSign,
    });
  }

  endSession(): void {
    this.connection?.send({ type: 'end_session' });
    setTimeout(() => {
      this.connection?.close();
      this.connection = null;
      this.sessionState$.next('disconnected');
      this.sessionId$.next(null);
    }, 1000);
  }
}
