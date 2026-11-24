import {
  AfterViewInit, Component, ElementRef, EventEmitter, Input, NgZone, OnChanges, OnDestroy, Output, SimpleChanges, ViewChild, effect, inject,
} from '@angular/core';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { environment } from '../../../../environments/environment';
import { SkinData, SkinFrame } from '../../../core/models/sign.model';
import { AvatarLook, AvatarLookService } from '../../../core/services/avatar-look.service';

// palm points (wrist, thumb base, knuckles)
const PALM = [0, 1, 2, 5, 9, 13, 17];
const FINGER_MM = 10;  // finger radius; an adult finger is ~18-22 mm across
// The body tracker's depth is exaggerated ~4-5x (nose ~1.8 shoulder-widths in
// front of the shoulders instead of ~0.4), so depth is taken relative to the
// shoulders and scaled down to body proportions.
const POSE_DEPTH = 0.22;

let meshAssets: Promise<{ tris: number[][]; conn: Record<string, number[][]> }> | null = null;
function loadMeshAssets() {
  meshAssets ??= Promise.all([
    fetch('skin/face_triangles.json').then(r => r.json()),
    fetch('skin/face_connections.json').then(r => r.json()),
  ]).then(([tris, conn]) => ({ tris, conn }));
  return meshAssets;
}

type Hand = { g: THREE.Group; fingers: THREE.Mesh[]; tips: THREE.Mesh[]; palm: THREE.Mesh; pts: THREE.Vector3[] };

// Each finger as one smooth tapered tube: landmark chain and relative thickness at each point.
const FINGERS: { chain: number[]; r: number[] }[] = [
  { chain: [1, 2, 3, 4], r: [1.28, 1.12, 1.0, 0.9] },      // thumb
  { chain: [5, 6, 7, 8], r: [1.0, 0.93, 0.84, 0.76] },     // index
  { chain: [9, 10, 11, 12], r: [1.04, 0.96, 0.86, 0.78] }, // middle
  { chain: [13, 14, 15, 16], r: [0.98, 0.9, 0.81, 0.73] }, // ring
  { chain: [17, 18, 19, 20], r: [0.86, 0.79, 0.71, 0.65] },// pinky
];

/** A smooth tube through `pts` whose radius follows `radii` (one per point). */
function taperedTube(pts: THREE.Vector3[], radii: number[], segs = 18, around = 12): THREE.BufferGeometry {
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
  const frames = curve.computeFrenetFrames(segs, false);
  const pos: number[] = [], nor: number[] = [], idx: number[] = [];
  for (let i = 0; i <= segs; i++) {
    const t = i / segs, p = curve.getPointAt(t), N = frames.normals[i], B = frames.binormals[i];
    const f = t * (radii.length - 1), k = Math.min(radii.length - 2, Math.floor(f));
    const r = radii[k] + (radii[k + 1] - radii[k]) * (f - k);
    for (let j = 0; j < around; j++) {
      const a = (j / around) * Math.PI * 2, cx = Math.cos(a), cy = Math.sin(a);
      const n = new THREE.Vector3().addScaledVector(N, cx).addScaledVector(B, cy).normalize();
      pos.push(p.x + n.x * r, p.y + n.y * r, p.z + n.z * r); nor.push(n.x, n.y, n.z);
    }
  }
  for (let i = 0; i < segs; i++) for (let j = 0; j < around; j++) {
    const a = i * around + j, b = i * around + (j + 1) % around, c = (i + 1) * around + j, d = (i + 1) * around + (j + 1) % around;
    idx.push(a, b, c, b, d, c);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setIndex(idx);
  return g;
}
type Eye = { g: THREE.Group; ball: THREE.Mesh; iris: THREE.Mesh; pupil: THREE.Mesh; corners: [number, number]; ring: number[] };

// Face-mesh landmarks: eye corners, the 10 iris points (468-477), brow lines.
const EYES: { corners: [number, number]; ring: number[] }[] = [
  { corners: [33, 133], ring: [468, 469, 470, 471, 472] },
  { corners: [362, 263], ring: [473, 474, 475, 476, 477] },
];
const BROWS = [
  { upper: [70, 63, 105, 66, 107], lower: [46, 53, 52, 65, 55] },
  { upper: [300, 293, 334, 296, 336], lower: [276, 283, 282, 295, 285] },
];
const LIP = new THREE.Color('#b5636a');
// upper eyelids (corner to corner), for lashes
const LIDS = [[33, 246, 161, 160, 159, 158, 157, 173, 133], [362, 398, 384, 385, 386, 387, 388, 466, 263]];

/**
 * v3: face and/or hands as shaded 3D skin, floating on their own (no body).
 * The face is the tracker's 478-point mesh as a surface; each hand is
 * tapered finger segments on a solid palm, shaped from its 3D landmarks and
 * placed where it is in the video.
 *
 * Data: `word` loads a sign's example skin (GET /signs/{word}/skin), or
 * `data` passes one directly (the learner's own attempt). Timing follows the
 * <video> inside `videoHost` when given, else its own clock at `speed`.
 * `show` picks hands only, face only, or both, and frames the view on them.
 * Emits `available` = false when there is nothing to show.
 */
@Component({
  selector: 'app-sign-skin',
  standalone: true,
  template: `
    <div class="relative rounded-xl overflow-hidden border border-border/50" [style.aspect-ratio]="aspect" style="background: radial-gradient(circle at 50% 35%, #2c2a38, #121118);">
      <canvas #canvasRef class="w-full h-full block cursor-grab active:cursor-grabbing" style="touch-action: none"></canvas>
      <span class="absolute top-2 left-2 text-[11px] font-medium text-white/80 bg-black/35 px-2 py-0.5 rounded-md pointer-events-none">{{ label }}</span>
      @if (state === 'loading') {
        <div class="absolute inset-0 flex items-center justify-center text-white/60 text-xs">Loading 3D…</div>
      } @else if (state === 'none') {
        <div class="absolute inset-0 flex items-center justify-center text-white/60 text-xs px-6 text-center">{{ emptyText }}</div>
      } @else if (state === 'error') {
        <div class="absolute inset-0 flex items-center justify-center text-white/60 text-xs px-6 text-center">Couldn't load the 3D view. Reload the page to try again.</div>
      }
      <button (click)="resetView()" class="absolute bottom-2 right-2 text-[11px] font-semibold px-2.5 py-1 rounded-full bg-white/90 text-[#2b2433]">Front view</button>
    </div>
  `,
})
export class SignSkinComponent implements AfterViewInit, OnChanges, OnDestroy {
  @Input() word = '';
  @Input() data: SkinData | null = null;
  @Input() videoHost: HTMLElement | null = null;
  @Input() show: 'all' | 'hands' | 'face' = 'all';
  @Input() speed = 1;
  @Input() label = 'Face and hands (3D)';
  @Input() aspect = '4/3';
  @Input() emptyText = 'Nothing to show in 3D for this one.';
  @Output() available = new EventEmitter<boolean>();
  /** v5: the frame being shown (whole frames only, emitted when it changes) -- e.g. to highlight
   * which sign of a sentence is playing. */
  @Output() position = new EventEmitter<number>();
  private lastEmitted = -1;
  @ViewChild('canvasRef') private canvasRef!: ElementRef<HTMLCanvasElement>;

  state: 'loading' | 'ready' | 'none' | 'error' = 'loading';

  private zone = inject(NgZone);
  private renderer: THREE.WebGLRenderer | null = null;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(30, 4 / 3, 1, 100000);
  private controls: OrbitControls | null = null;
  private raf = 0;
  private resizeObs: ResizeObserver | null = null;
  private skin: SkinData | null = null;
  /** per frame: the nearest frame that has hands / a face (held when tracking missed one) */
  private nearHands: number[] = [];
  private nearFace: number[] = [];
  private scales = [1, 1];
  private frameBox = { cx: 0, cy: 0, size: 100 };
  private clockStart = performance.now();
  private loadToken = 0;
  private ready = false;
  /** hands close-up: the point the view is centred on, eased toward the hands each frame */
  private followAt: THREE.Vector3 | null = null;
  /** two hands far apart: one half of the view per hand (left/right on screen) */
  private split = false;
  private sideAt: (THREE.Vector3 | null)[] = [null, null];
  private sideCam = new THREE.PerspectiveCamera(30, 1, 1, 100000);

  private skinMat = new THREE.MeshPhysicalMaterial({ color: new THREE.Color('#d29c74'), roughness: 0.5, metalness: 0,
    sheen: 0.5, sheenColor: new THREE.Color('#ffd2b8'), sheenRoughness: 0.6 });
  private faceMat = new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.55, metalness: 0,
    sheen: 0.4, sheenColor: new THREE.Color('#ffd2b8'), side: THREE.DoubleSide });
  private browMat = new THREE.MeshStandardMaterial({ color: new THREE.Color('#3b2a20'), roughness: 0.9 });
  private scleraMat = new THREE.MeshPhysicalMaterial({ color: new THREE.Color('#f1ece6'), roughness: 0.25, clearcoat: 0.6 });
  private irisMat = new THREE.MeshPhysicalMaterial({ color: new THREE.Color('#5a3a22'), roughness: 0.3, clearcoat: 0.8, side: THREE.DoubleSide });
  private pupilMat = new THREE.MeshBasicMaterial({ color: 0x0b0b0d, side: THREE.DoubleSide });
  private regions: { lips: Set<number>; brows: Set<number>; beard: Set<number> } = { lips: new Set(), brows: new Set(), beard: new Set() };
  private lashMat = new THREE.MeshStandardMaterial({ color: 0x15100e, roughness: 0.8 });
  private lashes: THREE.Mesh[] = [];
  private faceStyle: AvatarLook['face'] = 'neutral';
  private eyes: Eye[] = [];
  private brows: THREE.Mesh[] = [];
  private lookSvc = inject(AvatarLookService);
  private lookEffect = effect(() => this.applyLook(this.lookSvc.look()));
  private faceGeo = new THREE.BufferGeometry();
  private faceMesh: THREE.Mesh | null = null;
  private hands: Hand[] = [];

  ngAfterViewInit(): void {
    this.zone.runOutsideAngular(() => this.init());
  }

  ngOnChanges(ch: SimpleChanges): void {
    if (this.ready && (ch['word'] || ch['data'] || ch['show'])) this.load();
  }

  ngOnDestroy(): void {
    cancelAnimationFrame(this.raf);
    this.resizeObs?.disconnect();
    this.controls?.dispose();
    this.renderer?.dispose();
    this.hands.forEach(h => { h.palm.geometry.dispose(); h.fingers.forEach(f => f.geometry.dispose()); });
    this.faceGeo.dispose();
  }

  /** Play from the start again (own clock only). */
  restart(): void { this.clockStart = performance.now(); }

  resetView(): void {
    const b = this.frameBox, dist = b.size * (this.show === 'all' ? 2.1 : 2.3);
    // when following the hands, stand back from the hands themselves (they can be well in front of the body)
    const z = this.show === 'hands' && this.followAt ? this.followAt.z : 0;
    if (this.show === 'hands' && this.followAt) { b.cx = this.followAt.x; b.cy = this.followAt.y; }
    this.camera.position.set(b.cx, b.cy, z + dist);
    this.camera.near = dist / 50; this.camera.far = dist * 20; this.camera.updateProjectionMatrix();
    this.controls?.target.set(b.cx, b.cy, z);
    this.controls?.update();
  }

  private async init(): Promise<void> {
    const canvas = this.canvasRef.nativeElement;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;  // soft reflected light all round
    this.scene.environmentIntensity = 0.3;  // a touch, so skin keeps its warm tone
    this.scene.add(new THREE.HemisphereLight(0xfff4ea, 0x3a3550, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 1.4); key.position.set(-0.6, 0.9, 1.2); this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x9fb4ff, 0.7); rim.position.set(1, 0.2, -0.8); this.scene.add(rim);

    const { tris, conn } = await loadMeshAssets();
    this.regions = {
      lips: new Set(conn['FACE_LANDMARKS_LIPS'].flat()),
      brows: new Set([...conn['FACE_LANDMARKS_LEFT_EYEBROW'], ...conn['FACE_LANDMARKS_RIGHT_EYEBROW']].flat()),
      beard: this.regions.beard,
    };
    this.faceGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(478 * 3), 3));
    this.faceGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(478 * 3), 3));
    this.faceGeo.setIndex(tris.flat());
    this.faceMesh = new THREE.Mesh(this.faceGeo, this.faceMat);
    // its points move every frame: never skip drawing it based on where it was first drawn
    // (bug: after switching signs the face was culled and only the eyeballs showed)
    this.faceMesh.frustumCulled = false;
    this.scene.add(this.faceMesh);
    this.eyes = EYES.map(e => this.makeEye(e.corners, e.ring));
    this.brows = BROWS.map(() => { const m = new THREE.Mesh(new THREE.BufferGeometry(), this.browMat); this.scene.add(m); return m; });
    this.lashes = LIDS.map(() => { const m = new THREE.Mesh(new THREE.BufferGeometry(), this.lashMat); this.scene.add(m); return m; });
    this.applyLook(this.lookSvc.look());
    this.hands = [this.makeHand(), this.makeHand()];

    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(canvas);
    this.resize();
    this.ready = true;
    await this.load();
    const loop = () => { this.raf = requestAnimationFrame(loop); this.render(); };
    loop();
  }

  private async load(): Promise<void> {
    const token = ++this.loadToken;
    this.skin = null;
    this.setState('loading');
    // nothing chosen yet (the page sets the sign a moment later): wait, don't report "none"
    if (!this.data && !this.word) return;
    let d: SkinData | null = this.data;
    let missing = false;
    // a busy page (several videos streaming) can drop a request: retry before giving up
    for (let attempt = 0; !d && !missing && this.word && attempt < 4; attempt++) {
      try {
        const r = await fetch(`${environment.apiBaseUrl}/signs/${encodeURIComponent(this.word)}/skin`);
        if (r.status === 404) missing = true;
        else if (r.ok) d = await r.json();
      } catch { /* network hiccup: retry */ }
      if (token !== this.loadToken) return;
      if (!d && !missing) await new Promise(res => setTimeout(res, 700 * (attempt + 1)));
    }
    if (token !== this.loadToken) return;
    if (!d && !missing) { this.setState('error'); return; }   // couldn't reach the server: say so, no fallback
    if (!d || !this.hasContent(d)) { this.setState('none'); return; }
    this.skin = d;
    const ff = d.frames.find(f => f.f)?.f;
    if (ff) {
      const noseBase = ff[2][1], chin = ff[152][1];
      this.regions.beard = new Set(ff.map((p, k) => (p[1] > noseBase + (chin - noseBase) * 0.08 && !this.regions.lips.has(k) ? k : -1)).filter(k => k >= 0));
      this.applyLook(this.lookSvc.look());
    }
    this.nearHands = this.nearest(d, f => f.h.some(Boolean));
    this.nearFace = this.nearest(d, f => !!f.f);
    this.scales = [0, 1].map(k => this.handScale(d!, k));
    this.frame(d);
    this.clockStart = performance.now();
    this.setState('ready');
  }

  private hasContent(d: SkinData): boolean {
    const hands = d.frames.some(f => f.h.some(Boolean)), face = d.frames.some(f => f.f);
    return this.show === 'hands' ? hands : this.show === 'face' ? face : hands || face;
  }

  private setState(s: 'loading' | 'ready' | 'none' | 'error'): void {
    this.zone.run(() => {
      this.state = s;
      if (s === 'ready' || s === 'none') this.available.emit(s === 'ready');
    });
  }

  /** Skin tone, eye and brow colour, and surface style (AvatarLookService). */
  private applyLook(l: AvatarLook): void {
    const skin = new THREE.Color(l.skin), brow = new THREE.Color(l.brows);
    this.faceStyle = l.face;
    const lipMix = l.face === 'female' ? 0.78 : l.face === 'male' ? 0.25 : 0.4;
    this.skinMat.color.copy(skin);
    this.browMat.color.copy(brow);
    this.irisMat.color.set(l.eyes);
    const col = this.faceGeo.getAttribute('color') as THREE.BufferAttribute | undefined;
    if (col) {
      for (let i = 0; i < 478; i++) {
        const c = skin.clone();
        if (this.regions.lips.has(i)) c.multiplyScalar(0.82).lerp(LIP, lipMix);   // lips: a deeper, redder version of this skin tone
        else if (l.face === 'male' && this.regions.beard.has(i)) c.lerp(skin.clone().multiplyScalar(0.5).lerp(brow, 0.35), 0.55);  // stubble shadow
        else if (this.regions.brows.has(i)) c.lerp(brow, 0.25);
        col.setXYZ(i, c.r, c.g, c.b);
      }
      col.needsUpdate = true;
    }
    const style = { natural: { rough: 0.5, sheen: 0.5, coat: 0 }, clay: { rough: 0.85, sheen: 0.05, coat: 0 }, studio: { rough: 0.35, sheen: 0.3, coat: 0.35 } }[l.style];
    for (const m of [this.skinMat, this.faceMat]) {
      m.roughness = style.rough; m.sheen = style.sheen; m.clearcoat = style.coat; m.needsUpdate = true;
    }
  }

  private makeEye(corners: [number, number], ring: number[]): Eye {
    const g = new THREE.Group();
    const ball = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 18), this.scleraMat);
    const iris = new THREE.Mesh(new THREE.CircleGeometry(1, 28), this.irisMat);
    const pupil = new THREE.Mesh(new THREE.CircleGeometry(1, 20), this.pupilMat);
    g.add(ball, iris, pupil);
    this.scene.add(g);
    return { g, ball, iris, pupil, corners, ring };
  }

  /** Eyeballs behind the face's eye openings, iris and pupil where the tracker saw the iris. */
  private updateEyes(F: THREE.Vector3[], show: boolean): void {
    for (const e of this.eyes) e.g.visible = show;
    if (!show) return;
    const n = new THREE.Vector3().subVectors(F[263], F[33]).cross(new THREE.Vector3().subVectors(F[10], F[152])).normalize();
    if (n.z < 0) n.negate();   // face normal, toward the viewer
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
    for (const e of this.eyes) {
      const a = F[e.corners[0]], b = F[e.corners[1]];
      const w = a.distanceTo(b);
      const mid = a.clone().add(b).multiplyScalar(0.5);
      // iris centre, kept inside the eye opening (noisy webcam tracking can throw it outside)
      const c = F[e.ring[0]].clone();
      const off = c.clone().sub(mid);
      if (off.length() > w * 0.25) c.copy(mid).addScaledVector(off.normalize(), w * 0.25);
      // iris radius from the tracked ring, held to a human range (about 0.2 eye widths);
      // the raw ring jumps frame to frame on webcam video
      const measured = e.ring.slice(1).reduce((s, k) => s + F[k].distanceTo(F[e.ring[0]]), 0) / 4;
      const ir = Math.min(w * 0.24, Math.max(w * 0.17, measured || w * 0.2));
      const R = w * 0.58;
      e.ball.position.copy(c).addScaledVector(n, -R * 0.92);
      e.ball.scale.setScalar(R);
      // the eyeball's front is at +0.08 R: iris and pupil sit just in front of it (they used to be
      // behind / exactly on it, so the iris hid inside the ball and the pupil flickered)
      e.iris.position.copy(c).addScaledVector(n, R * 0.11); e.iris.quaternion.copy(q); e.iris.scale.setScalar(ir);
      e.pupil.position.copy(c).addScaledVector(n, R * 0.125); e.pupil.quaternion.copy(q); e.pupil.scale.setScalar(ir * 0.45);
    }
  }

  /** Upper lashes (female): a fine dark strip along each upper eyelid, tipped slightly out and up. */
  private updateLashes(F: THREE.Vector3[], show: boolean): void {
    this.lashes.forEach((m, k) => {
      m.visible = show && this.faceStyle === 'female';
      if (!m.visible) return;
      const n = new THREE.Vector3().subVectors(F[263], F[33]).cross(new THREE.Vector3().subVectors(F[10], F[152])).normalize();
      if (n.z < 0) n.negate();
      const up = new THREE.Vector3().subVectors(F[10], F[152]).normalize();
      const w = F[LIDS[k][0]].distanceTo(F[LIDS[k][LIDS[k].length - 1]]);
      const pts = LIDS[k].map((j, i, a) => F[j].clone().addScaledVector(n, w * 0.06).addScaledVector(up, w * 0.04 * Math.sin(Math.PI * i / (a.length - 1))));
      const radii = LIDS[k].map((_, i, a) => w * (0.018 + 0.03 * Math.sin(Math.PI * i / (a.length - 1)) * (0.6 + 0.4 * i / (a.length - 1))));
      m.geometry.dispose();
      m.geometry = taperedTube(pts, radii, 20, 8);
    });
  }

  /** Eyebrows: a soft tapered strip along the middle of each tracked brow. */
  private updateBrows(F: THREE.Vector3[], show: boolean): void {
    BROWS.forEach((b, k) => {
      const m = this.brows[k];
      m.visible = show;
      if (!show) return;
      const mid = b.upper.map((u, i) => F[u].clone().lerp(F[b.lower[i]], 0.5));
      const thick = b.upper.map((u, i) => F[u].distanceTo(F[b.lower[i]]));
      const curve = new THREE.CatmullRomCurve3(mid);
      const segs = 24, around = 8, pos: number[] = [], idx: number[] = [];
      const fw = F[263].distanceTo(F[33]);
      // lift off the skin along the face's own direction (not just toward the camera)
      const out = new THREE.Vector3().subVectors(F[263], F[33]).cross(new THREE.Vector3().subVectors(F[10], F[152])).normalize();
      if (out.z < 0) out.negate();
      const upDir = new THREE.Vector3().subVectors(F[10], F[152]).normalize();
      for (let i = 0; i <= segs; i++) {
        const t = i / segs, p = curve.getPoint(t), tan = curve.getTangent(t);
        const side = out.clone().cross(tan).normalize();
        // fuller at the inner end, tapering toward the tail
        const weight = this.faceStyle === 'female' ? 0.48 : this.faceStyle === 'male' ? 1.35 : 1;
        const r = weight * Math.max(thick[Math.min(4, Math.round(t * 4))] * 0.62, fw * 0.035) * (0.6 + 0.4 * Math.sin(Math.PI * Math.min(1, 0.15 + t * 0.95)));
        if (this.faceStyle === 'female') p.addScaledVector(upDir, Math.sin(Math.PI * Math.min(1, t * 1.25)) * fw * 0.04);   // an arch, peaking past the middle
        for (let j = 0; j < around; j++) {
          const a = (j / around) * Math.PI * 2;
          const v = p.clone().addScaledVector(side, Math.cos(a) * r).addScaledVector(out, Math.sin(a) * r * 0.5 + fw * 0.03);
          pos.push(v.x, v.y, v.z);
        }
      }
      for (let i = 0; i < segs; i++) for (let j = 0; j < around; j++) {
        const a = i * around + j, b2 = i * around + (j + 1) % around, c = (i + 1) * around + j, d = (i + 1) * around + (j + 1) % around;
        idx.push(a, c, b2, b2, c, d);
      }
      m.geometry.dispose();
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      g.setIndex(idx); g.computeVertexNormals();
      m.geometry = g;
    });
  }

  private makeHand(): Hand {
    const g = new THREE.Group();
    const sph = new THREE.SphereGeometry(1, 16, 12);
    const fingers = FINGERS.map(() => { const m = new THREE.Mesh(new THREE.BufferGeometry(), this.skinMat); g.add(m); return m; });
    const tips = FINGERS.map(() => { const m = new THREE.Mesh(sph, this.skinMat); g.add(m); return m; });  // rounded fingertips
    const palm = new THREE.Mesh(new THREE.BufferGeometry(), this.skinMat); g.add(palm);
    this.scene.add(g);
    return { g, fingers, tips, palm, pts: [] };
  }

  /** For each frame, the closest frame (earlier preferred) where `has` is true. */
  private nearest(d: SkinData, has: (f: SkinData['frames'][number]) => boolean): number[] {
    const ok = d.frames.map(has), out: number[] = [];
    for (let i = 0; i < ok.length; i++) {
      let best = i;
      if (!ok[i]) for (let k = 1; k < ok.length; k++) {
        if (i - k >= 0 && ok[i - k]) { best = i - k; break; }
        if (i + k < ok.length && ok[i + k]) { best = i + k; break; }
      }
      out.push(best);
    }
    return out;
  }

  /** A hand is "raised" (signing) when its wrist is above about the belly. */
  private raised(f: SkinData['frames'][number], h: number[][]): boolean {
    const p = f.p;
    if (!p) return true;
    const sw = Math.hypot(p[11][0] - p[12][0], p[11][1] - p[12][1]) || 1;
    return h[0][1] < (p[11][1] + p[12][1]) / 2 + 0.85 * sw;
  }

  /** 3D-shape units -> video pixels for a hand: its typical size on video / in 3D. */
  private handScale(d: SkinData, k: number): number {
    const r: number[] = [];
    for (const f of d.frames) {
      const h = f.h[k], w = f.w?.[k];
      if (!h || !w) continue;
      const a = Math.hypot(h[9][0] - h[0][0], h[9][1] - h[0][1]);
      const b = Math.hypot(w[9][0] - w[0][0], w[9][1] - w[0][1], w[9][2] - w[0][2]);
      if (b > 1e-6) r.push(a / b);
    }
    r.sort((x, y) => x - y);
    return r.length ? r[Math.floor(r.length / 2)] : 1;
  }

  /** Frame the view on what is shown (over the whole clip). */
  private frame(d: SkinData): void {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    const add = (p: number[]) => { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); };
    if (this.show === 'face') for (const f of d.frames) f.f?.forEach(add);
    if (this.show === 'hands') {
      // close-up sized from the hand itself (wrist -> middle fingertip), and
      // following the hands as they move -- see follow(). Two hands far apart
      // get a split view, one half per hand, so neither ends up tiny.
      const lens: number[] = [], spreads: number[] = [];
      let both = 0, any = 0;
      for (const f of d.frames) {
        const hs = f.h.filter(Boolean) as number[][][];
        hs.forEach(h => lens.push(Math.hypot(h[12][0] - h[0][0], h[12][1] - h[0][1])));
        if (hs.length) any++;
        if (hs.length === 2) { both++; spreads.push(Math.hypot(hs[0][9][0] - hs[1][9][0], hs[0][9][1] - hs[1][9][1])); }
      }
      const med = (a: number[]) => (a.sort((x, y) => x - y), a.length ? a[Math.floor(a.length / 2)] : 0);
      const handLen = Math.max(med(lens), 20), spread = med(spreads);
      this.split = any > 0 && both / any >= 0.3 && spread > 2.6 * handLen;
      // capped at 4 hand-lengths: a stray far-off detection must not shrink the hands
      this.frameBox = { cx: 0, cy: 0, size: this.split ? handLen * 1.9 : Math.min(Math.max(handLen * 2.2, spread + handLen * 1.3), handLen * 4) };
      this.followAt = null;
      this.sideAt = [null, null];
      this.resetView();
      return;
    }
    if (this.show === 'all') {
      // face and hands together, framed from the face: face in the top third,
      // signing space (down to about the chest) below it, the same for every
      // clip -- so "you" and "target" side by side are at the same height and size
      const hs: number[] = [], cx: number[] = [], cy: number[] = [];
      for (const f of d.frames) {
        if (!f.f) continue;
        const ys = f.f.map(p => p[1]), xs = f.f.map(p => p[0]);
        hs.push(Math.max(...ys) - Math.min(...ys)); cx.push((Math.max(...xs) + Math.min(...xs)) / 2); cy.push((Math.max(...ys) + Math.min(...ys)) / 2);
      }
      if (hs.length) {
        const med = (a: number[]) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
        const fh = med(hs);
        this.frameBox = { cx: med(cx), cy: -(med(cy) + 0.95 * fh), size: 3.3 * fh };
        this.resetView();
        return;
      }
      for (const f of d.frames) f.h.forEach(h => { if (h && this.raised(f, h)) h.forEach(add); });
    }
    if (!isFinite(x0)) { x0 = 0; y0 = 0; x1 = d.width; y1 = d.height; }
    this.frameBox = { cx: (x0 + x1) / 2, cy: -(y0 + y1) / 2, size: Math.max(x1 - x0, y1 - y0, 40) };
    this.resetView();
  }

  private resize(): void {
    if (!this.renderer) return;
    const c = this.canvasRef.nativeElement, w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  /** Position in the clip, in frames (fractional): the followed video's time, or our own clock. */
  private framePos(d: SkinData): number {
    const fixed = new URLSearchParams(location.search).get('skinFrame');  // dev aid: hold one frame
    if (fixed !== null) return Math.min(d.n_frames - 1, Math.max(0, +fixed));
    const v = this.videoHost?.querySelector('video');
    const t = v && v.readyState >= 1 ? v.currentTime
      : ((performance.now() - this.clockStart) / 1000 * this.speed) % (d.n_frames / d.fps);
    return Math.min(d.n_frames - 1, Math.max(0, t * d.fps));
  }

  /** The frame as drawn at frame i (hands / face from the nearest frame that has them). */
  private frameAt(d: SkinData, i: number): SkinFrame {
    const fh = d.frames[this.nearHands[i] ?? i], ff = d.frames[this.nearFace[i] ?? i];
    return { ...d.frames[i], h: fh.h, w: fh.w, f: ff.f, p: d.frames[i].p ?? fh.p ?? ff.p };
  }

  /** Blend two frames so playback moves smoothly between recorded frames (learner clips are only
   * 10 fps). Parts present in only one of them are taken from the nearer frame. */
  private blend(a: SkinFrame, b: SkinFrame, t: number): SkinFrame {
    if (t <= 0.001) return a;
    const mix = (x: number[][] | null | undefined, y: number[][] | null | undefined) =>
      x && y && x.length === y.length ? x.map((p, k) => p.map((v, j) => v + (y[k][j] - v) * t)) : (t < 0.5 ? x : y) ?? null;
    const near = t < 0.5 ? a : b;
    return {
      ...near,
      h: [0, 1].map(k => mix(a.h[k], b.h[k])) as SkinFrame['h'],
      w: [0, 1].map(k => mix(a.w?.[k], b.w?.[k])) as SkinFrame['w'],
      p: mix(a.p, b.p) as SkinFrame['p'],
      f: mix(a.f, b.f) as SkinFrame['f'],
    };
  }

  private render(): void {
    if (!this.renderer) return;
    const d = this.skin;
    const showFace = this.show !== 'hands', showHands = this.show !== 'face';
    if (d && this.faceMesh) {
      const pos = this.framePos(d), i0 = Math.floor(pos), i1 = Math.min(d.n_frames - 1, i0 + 1);
      if (i0 !== this.lastEmitted && this.position.observed) { this.lastEmitted = i0; this.zone.run(() => this.position.emit(i0)); }
      const f = this.blend(this.frameAt(d, i0), this.frameAt(d, i1), pos - i0);
      const pose = f.p;
      const sw = pose ? Math.hypot(pose[11][0] - pose[12][0], pose[11][1] - pose[12][1]) || 1 : 1;
      const midZ = pose ? (pose[11][2] + pose[12][2]) / 2 : 0;
      const depth = (q: number[]) => -(q[2] - midZ) * POSE_DEPTH;  // toward the camera = +z, shoulders at 0
      this.faceMesh.visible = showFace && !!f.f;
      if (f.f && showFace) {
        const fw = Math.max(...f.f.map(p => p[0])) - Math.min(...f.f.map(p => p[0]));
        const pos = this.faceGeo.attributes['position'] as THREE.BufferAttribute;
        // nose depth from the body tracker; without it the face sits behind the hands
        const noseZ = pose ? Math.min(Math.max(depth(pose[0]), 0.15 * sw), 0.7 * sw) : -0.45 * fw, n1 = f.f[1][2] || 0;
        f.f.forEach((p, k) => pos.setXYZ(k, p[0], -p[1], noseZ - ((p[2] || 0) - n1)));
        pos.needsUpdate = true;
        this.faceGeo.computeVertexNormals();
        const F = f.f.map((_, k) => new THREE.Vector3(pos.getX(k), pos.getY(k), pos.getZ(k)));
        this.updateEyes(F, f.f.length >= 478);
        this.updateBrows(F, true);
        this.updateLashes(F, true);
      } else {
        this.updateEyes([], false);
        this.updateBrows([], false);
        this.updateLashes([], false);
      }
      for (let k = 0; k < 2; k++) {
        const h = showHands ? f.h[k] : null;
        let wz = 0;
        if (pose && h) {
          const pk = Math.hypot(pose[15][0] - h[0][0], pose[15][1] - h[0][1]) < Math.hypot(pose[16][0] - h[0][0], pose[16][1] - h[0][1]) ? 15 : 16;
          wz = Math.min(Math.max(depth(pose[pk]), -0.3 * sw), 1.6 * sw);  // a hand can't reach further than an arm
        }
        try { this.updateHand(this.hands[k], h, f.w?.[k] ?? null, this.scales[k], wz); }
        catch { this.hands[k].g.visible = false; }
      }
    } else {
      this.hands.forEach(h => (h.g.visible = false));
      if (this.faceMesh) this.faceMesh.visible = false;
      this.updateEyes([], false);
      this.updateBrows([], false);
      this.updateLashes([], false);
    }
    if (this.show === 'hands' && d) this.follow();
    this.controls?.update();
    const visible = this.hands.filter(h => h.g.visible);
    if (this.split && visible.length === 2) this.renderSplit(visible);
    else this.renderer.render(this.scene, this.camera);
  }

  /** Two hands far apart: left half follows the hand on the left, right half the other.
   * Both halves use the main view's angle and zoom, so dragging turns both. */
  private renderSplit(visible: Hand[]): void {
    const r = this.renderer!, c = this.canvasRef.nativeElement, W = c.clientWidth, H = c.clientHeight;
    const centre = (h: Hand) => h.pts.reduce((v, p) => v.add(p), new THREE.Vector3()).multiplyScalar(1 / Math.max(1, h.pts.length));
    const sides = visible.map(centre).sort((a, b) => a.x - b.x);
    const offset = this.camera.position.clone().sub(this.controls!.target);
    r.setScissorTest(true);
    for (let k = 0; k < 2; k++) {
      this.sideAt[k] = this.sideAt[k] ? this.sideAt[k]!.lerp(sides[k], 0.2) : sides[k].clone();
      this.sideCam.fov = this.camera.fov; this.sideCam.aspect = (W / 2) / H;
      this.sideCam.near = this.camera.near; this.sideCam.far = this.camera.far;
      this.sideCam.position.copy(this.sideAt[k]!).add(offset);
      this.sideCam.lookAt(this.sideAt[k]!);
      this.sideCam.updateProjectionMatrix();
      r.setViewport(k * W / 2, 0, W / 2, H);
      r.setScissor(k * W / 2, 0, W / 2, H);
      r.render(this.scene, this.sideCam);
    }
    r.setScissorTest(false);
    r.setViewport(0, 0, W, H);
  }

  /** Hands close-up: ease the view's centre toward the visible hands. */
  private follow(): void {
    const pts = this.hands.filter(h => h.g.visible).flatMap(h => h.pts);
    if (!pts.length || !this.controls) return;
    const c = pts.reduce((v, p) => v.add(p), new THREE.Vector3()).multiplyScalar(1 / pts.length);
    if (!this.followAt) {
      this.followAt = c.clone();
      this.resetView();
      return;
    }
    const step = c.clone().sub(this.followAt).multiplyScalar(0.3);
    this.followAt.add(step);
    this.camera.position.add(step);
    this.controls.target.add(step);
  }

  private updateHand(H: Hand, img: number[][] | null, world: number[][] | null, scale: number, wristZ = 0): void {
    if (!img || !world) { H.g.visible = false; return; }
    H.g.visible = true;
    const w0 = world[0], i0 = img[0];
    const P = world.map(p => new THREE.Vector3(i0[0] + (p[0] - w0[0]) * scale, -(i0[1] + (p[1] - w0[1]) * scale), wristZ - (p[2] - w0[2]) * scale));
    // finger thickness from this hand's own size (wrist -> middle knuckle is ~90 mm)
    const r = FINGER_MM * P[9].distanceTo(P[0]) / 90;
    H.pts = P;
    FINGERS.forEach((fg, k) => {
      // start a little inside the palm so finger and palm read as one piece
      const pts = fg.chain.map(j => P[j].clone());
      pts[0].lerp(P[0], fg.chain[0] === 1 ? 0.25 : 0.12);
      const radii = fg.r.map(x => x * r);
      H.fingers[k].geometry.dispose();
      H.fingers[k].geometry = taperedTube(pts, radii);
      H.tips[k].position.copy(pts[pts.length - 1]);
      H.tips[k].scale.setScalar(radii[radii.length - 1]);
    });
    const n = new THREE.Vector3().subVectors(P[5], P[0]).cross(new THREE.Vector3().subVectors(P[17], P[0])).normalize();
    const c = PALM.reduce((v, j) => v.add(P[j]), new THREE.Vector3()).multiplyScalar(1 / PALM.length);
    // rounded palm: the hull of a small ball of points around each palm landmark
    // (a flattened ball, thinner across the palm), so the outline has soft edges
    const t1 = new THREE.Vector3().subVectors(P[5], P[17]).normalize(), t2 = new THREE.Vector3().crossVectors(n, t1).normalize();
    const pts: THREE.Vector3[] = [];
    for (const j of PALM) {
      const q = P[j].clone().lerp(c, j === 0 ? 0.1 : 0.02);
      const R = r * (j === 0 ? 1.5 : j === 1 || j === 2 ? 1.25 : 1.1);
      for (let a = 0; a < 10; a++) for (const e of [-0.75, -0.3, 0.3, 0.75]) {
        const th = (a / 10) * Math.PI * 2, ring = Math.sqrt(1 - e * e);
        pts.push(q.clone().addScaledVector(t1, Math.cos(th) * ring * R).addScaledVector(t2, Math.sin(th) * ring * R).addScaledVector(n, e * R * 0.95));
      }
    }
    H.palm.geometry.dispose();
    const hull = new ConvexGeometry(pts);
    hull.deleteAttribute('normal'); hull.deleteAttribute('uv');
    H.palm.geometry = mergeVertices(hull, 1e-3);
    H.palm.geometry.computeVertexNormals();
  }
}
