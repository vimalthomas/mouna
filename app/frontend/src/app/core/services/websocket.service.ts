import { Injectable } from '@angular/core';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { BehaviorSubject, Observable, Subject, timer } from 'rxjs';
import { retryWhen, tap, delayWhen } from 'rxjs/operators';

export type ConnectionState = 'connecting' | 'connected' | 'disconnected';

export interface WsConnection<T> {
  messages$: Observable<T>;
  state$: Observable<ConnectionState>;
  send: (data: unknown) => void;
  close: () => void;
}

@Injectable({ providedIn: 'root' })
export class WebSocketService {
  connect<T>(url: string): WsConnection<T> {
    const state = new BehaviorSubject<ConnectionState>('connecting');
    let retryAttempt = 0;

    const ws: WebSocketSubject<T> = webSocket({
      url,
      openObserver: {
        next: () => {
          state.next('connected');
          retryAttempt = 0;
        },
      },
      closeObserver: {
        next: () => state.next('disconnected'),
      },
    });

    const messages$ = ws.pipe(
      retryWhen(errors =>
        errors.pipe(
          tap(() => {
            state.next('disconnected');
            retryAttempt++;
          }),
          delayWhen(() => {
            const delay = Math.min(1000 * Math.pow(2, retryAttempt), 30000);
            state.next('connecting');
            return timer(delay);
          })
        )
      )
    );

    return {
      messages$,
      state$: state.asObservable(),
      send: (data: unknown) => ws.next(data as T),
      close: () => ws.complete(),
    };
  }
}
