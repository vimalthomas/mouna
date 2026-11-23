import { Injectable, signal } from '@angular/core';

/** How the 3D face and hands look (v3). Chosen once, applied to every skin view. */
export interface AvatarLook {
  skin: string;   // skin tone
  eyes: string;   // iris colour
  brows: string;  // eyebrow colour
  style: 'natural' | 'clay' | 'studio';
  /** cosmetic cues only (brows, lashes, lips, beard shadow); the face shape is always the signer's own */
  face: 'neutral' | 'female' | 'male';
}

export const SKIN_TONES = [
  { name: 'Light', value: '#f3d7c4' },
  { name: 'Fair', value: '#e8bf9f' },
  { name: 'Medium', value: '#d29c74' },
  { name: 'Tan', value: '#b07a52' },
  { name: 'Brown', value: '#8a5636' },
  { name: 'Deep', value: '#5e3823' },
];
export const EYE_COLOURS = [
  { name: 'Brown', value: '#5a3a22' },
  { name: 'Hazel', value: '#7a5a2e' },
  { name: 'Green', value: '#4f6b3a' },
  { name: 'Blue', value: '#3f6a9a' },
  { name: 'Grey', value: '#6b7480' },
];
export const BROW_COLOURS = [
  { name: 'Black', value: '#1f1a1a' },
  { name: 'Dark brown', value: '#3b2a20' },
  { name: 'Brown', value: '#6b4a30' },
  { name: 'Blond', value: '#a8875a' },
  { name: 'Grey', value: '#8c8a88' },
];
export const FACES: { id: AvatarLook['face']; name: string }[] = [
  { id: 'neutral', name: 'Neutral' },
  { id: 'female', name: 'Female' },
  { id: 'male', name: 'Male' },
];
export const STYLES: { id: AvatarLook['style']; name: string }[] = [
  { id: 'natural', name: 'Natural' },
  { id: 'clay', name: 'Soft clay' },
  { id: 'studio', name: 'Studio' },
];

const KEY = 'mouna_avatar_look';
const DEFAULT: AvatarLook = { skin: '#d29c74', eyes: '#5a3a22', brows: '#3b2a20', style: 'natural', face: 'neutral' };

@Injectable({ providedIn: 'root' })
export class AvatarLookService {
  readonly look = signal<AvatarLook>(this.read());

  set(patch: Partial<AvatarLook>): void {
    const next = { ...this.look(), ...patch };
    this.look.set(next);
    try { localStorage.setItem(KEY, JSON.stringify(next)); } catch { /* private mode: keep for this visit only */ }
  }

  private read(): AvatarLook {
    try {
      const v = JSON.parse(localStorage.getItem(KEY) || 'null');
      if (v && typeof v.skin === 'string') return { ...DEFAULT, ...v };
    } catch { /* ignore */ }
    return DEFAULT;
  }
}
