import { Component, inject } from '@angular/core';
import { AvatarLookService, BROW_COLOURS, EYE_COLOURS, FACES, SKIN_TONES, STYLES } from '../../../core/services/avatar-look.service';

/** Compact picker for the 3D face and hands: skin tone, eyes, brows, style. */
@Component({
  selector: 'app-look-picker',
  standalone: true,
  template: `
    <div class="flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-ink-muted">
      <div class="flex items-center gap-1.5" role="radiogroup" aria-label="Skin tone">
        <span class="mr-0.5">Skin</span>
        @for (t of skins; track t.value) {
          <button type="button" role="radio" [attr.aria-checked]="look().skin === t.value" [attr.aria-label]="t.name" [title]="t.name"
            (click)="svc.set({ skin: t.value })" class="w-5 h-5 rounded-full border-2 transition"
            [style.background]="t.value" [class]="look().skin === t.value ? 'border-accent scale-110' : 'border-transparent'"></button>
        }
      </div>
      <div class="flex items-center gap-1.5" role="radiogroup" aria-label="Eye colour">
        <span class="mr-0.5">Eyes</span>
        @for (t of eyes; track t.value) {
          <button type="button" role="radio" [attr.aria-checked]="look().eyes === t.value" [attr.aria-label]="t.name" [title]="t.name"
            (click)="svc.set({ eyes: t.value })" class="w-4 h-4 rounded-full border-2 transition"
            [style.background]="t.value" [class]="look().eyes === t.value ? 'border-accent scale-110' : 'border-transparent'"></button>
        }
      </div>
      <div class="flex items-center gap-1.5" role="radiogroup" aria-label="Eyebrow colour">
        <span class="mr-0.5">Brows</span>
        @for (t of brows; track t.value) {
          <button type="button" role="radio" [attr.aria-checked]="look().brows === t.value" [attr.aria-label]="t.name" [title]="t.name"
            (click)="svc.set({ brows: t.value })" class="w-4 h-4 rounded-full border-2 transition"
            [style.background]="t.value" [class]="look().brows === t.value ? 'border-accent scale-110' : 'border-transparent'"></button>
        }
      </div>
      <div class="flex items-center gap-1" role="radiogroup" aria-label="Face">
        @for (f of faces; track f.id) {
          <button type="button" role="radio" [attr.aria-checked]="look().face === f.id" (click)="svc.set({ face: f.id })"
            class="px-2 py-0.5 rounded-full border text-[11px] font-medium"
            [class]="look().face === f.id ? 'bg-accent text-white border-accent' : 'bg-surface-sunk text-ink-muted border-border/50'">{{ f.name }}</button>
        }
      </div>
      <div class="flex items-center gap-1" role="radiogroup" aria-label="Style">
        @for (s of styles; track s.id) {
          <button type="button" role="radio" [attr.aria-checked]="look().style === s.id" (click)="svc.set({ style: s.id })"
            class="px-2 py-0.5 rounded-full border text-[11px] font-medium"
            [class]="look().style === s.id ? 'bg-accent text-white border-accent' : 'bg-surface-sunk text-ink-muted border-border/50'">{{ s.name }}</button>
        }
      </div>
    </div>
  `,
})
export class LookPickerComponent {
  readonly svc = inject(AvatarLookService);
  readonly look = this.svc.look;
  readonly skins = SKIN_TONES;
  readonly eyes = EYE_COLOURS;
  readonly brows = BROW_COLOURS;
  readonly styles = STYLES;
  readonly faces = FACES;
}
