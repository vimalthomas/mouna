import {
  AfterViewInit, Component, ElementRef, EventEmitter, Input, NgZone, OnChanges, OnDestroy, Output, SimpleChanges, ViewChild, inject,
} from '@angular/core';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { environment } from '../../../../environments/environment';
import { SkinData } from '../../../core/models/sign.model';

// finger segments (parent, child, relative thickness); palm points
const SEGMENTS: [number, number, number][] = [[1, 2, 1.05], [2, 3, 0.95], [3, 4, 0.85], [5, 6, 0.92], [6, 7, 0.82], [7, 8, 0.74],
  [9, 10, 0.95], [10, 11, 0.84], [11, 12, 0.75], [13, 14, 0.9], [14, 15, 0.8], [15, 16, 0.72], [17, 18, 0.8], [18, 19, 0.72], [19, 20, 0.65]];
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

type Hand = { g: THREE.Group; segs: THREE.Mesh[]; joints: THREE.Mesh[]; palm: THREE.Mesh };

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
  @ViewChild('canvasRef') private canvasRef!: ElementRef<HTMLCanvasElement>;

  state: 'loading' | 'ready' | 'none' = 'loading';

  private zone = inject(NgZone);
  private renderer: THREE.WebGLRenderer | null = null;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(30, 4 / 3, 1, 100000);
  private controls: OrbitControls | null = null;
  private raf = 0;
  private resizeObs: ResizeObserver | null = null;
  private skin: SkinData | null = null;
  private scales = [1, 1];
  private frameBox = { cx: 0, cy: 0, size: 100 };
  private clockStart = performance.now();
  private loadToken = 0;
  private ready = false;
  /** hands close-up: the point the view is centred on, eased toward the hands each frame */
  private followAt: THREE.Vector3 | null = null;

  private skinMat = new THREE.MeshPhysicalMaterial({ color: new THREE.Color('#e3b394'), roughness: 0.5, metalness: 0,
    sheen: 0.5, sheenColor: new THREE.Color('#ffd2b8'), sheenRoughness: 0.6 });
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
    this.hands.forEach(h => h.palm.geometry.dispose());
    this.faceGeo.dispose();
  }

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
    const skin = new THREE.Color('#e3b394');
    const tint = (ids: number[][], c: string) => ({ set: new Set(ids.flat()), color: new THREE.Color(c) });
    const tints = [tint(conn['FACE_LANDMARKS_LIPS'], '#b5636a'),
      tint([...conn['FACE_LANDMARKS_LEFT_EYEBROW'], ...conn['FACE_LANDMARKS_RIGHT_EYEBROW']], '#7a5546'),
      tint([...conn['FACE_LANDMARKS_LEFT_EYE'], ...conn['FACE_LANDMARKS_RIGHT_EYE']], '#d9a88a')];
    const colors = new Float32Array(478 * 3);
    for (let i = 0; i < 478; i++) {
      const c = tints.find(t => t.set.has(i))?.color ?? skin;
      colors.set([c.r, c.g, c.b], i * 3);
    }
    this.faceGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(478 * 3), 3));
    this.faceGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    this.faceGeo.setIndex(tris.flat());
    this.faceMesh = new THREE.Mesh(this.faceGeo, new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.55, metalness: 0,
      sheen: 0.4, sheenColor: new THREE.Color('#ffd2b8'), side: THREE.DoubleSide }));
    this.scene.add(this.faceMesh);
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
    if (!d && this.word) {
      try {
        const r = await fetch(`${environment.apiBaseUrl}/signs/${encodeURIComponent(this.word)}/skin`);
        d = r.ok ? await r.json() : null;
      } catch { d = null; }
    }
    if (token !== this.loadToken) return;
    if (!d || !this.hasContent(d)) { this.setState('none'); return; }
    this.skin = d;
    this.scales = [0, 1].map(k => this.handScale(d!, k));
    this.frame(d);
    this.clockStart = performance.now();
    this.setState('ready');
  }

  private hasContent(d: SkinData): boolean {
    const hands = d.frames.some(f => f.h.some(Boolean)), face = d.frames.some(f => f.f);
    return this.show === 'hands' ? hands : this.show === 'face' ? face : hands || face;
  }

  private setState(s: 'loading' | 'ready' | 'none'): void {
    this.zone.run(() => {
      this.state = s;
      if (s !== 'loading') this.available.emit(s === 'ready');
    });
  }

  private makeHand(): Hand {
    const g = new THREE.Group();
    const cyl = new THREE.CylinderGeometry(1, 1, 1, 18, 1, true), sph = new THREE.SphereGeometry(1, 18, 14);
    const segs = SEGMENTS.map(() => { const m = new THREE.Mesh(cyl, this.skinMat); g.add(m); return m; });
    const joints = Array.from({ length: 21 }, () => { const m = new THREE.Mesh(sph, this.skinMat); g.add(m); return m; });
    const palm = new THREE.Mesh(new THREE.BufferGeometry(), this.skinMat); g.add(palm);
    this.scene.add(g);
    return { g, segs, joints, palm };
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
    for (const f of d.frames) {
      if (this.show !== 'hands') f.f?.forEach(add);
      if (this.show !== 'face') f.h.forEach(h => h?.forEach(add));
    }
    if (this.show === 'hands') {
      // close-up: size the view to the hands themselves (typical spread of the
      // hands in one frame), and follow them as they move -- see follow()
      const spans: number[] = [];
      for (const f of d.frames) {
        const pts = f.h.filter(Boolean).flat() as number[][];
        if (!pts.length) continue;
        const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
        spans.push(Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)));
      }
      spans.sort((a, b) => a - b);
      const span = spans.length ? spans[Math.floor(spans.length * 0.75)] : 100;
      this.frameBox = { cx: 0, cy: 0, size: span * 1.15 };
      this.followAt = null;
      this.resetView();
      return;
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

  /** Frame to show: the followed video's current frame, or our own clock. */
  private frameIndex(d: SkinData): number {
    const fixed = new URLSearchParams(location.search).get('skinFrame');  // dev aid: hold one frame
    if (fixed !== null) return Math.min(d.n_frames - 1, Math.max(0, +fixed));
    const v = this.videoHost?.querySelector('video');
    const t = v && v.readyState >= 1 ? v.currentTime
      : ((performance.now() - this.clockStart) / 1000 * this.speed) % (d.n_frames / d.fps);
    return Math.min(d.n_frames - 1, Math.max(0, Math.round(t * d.fps)));
  }

  private render(): void {
    if (!this.renderer) return;
    const d = this.skin;
    const showFace = this.show !== 'hands', showHands = this.show !== 'face';
    if (d && this.faceMesh) {
      const f = d.frames[this.frameIndex(d)];
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
    }
    if (this.show === 'hands' && d) this.follow();
    this.controls?.update();
    this.renderer.render(this.scene, this.camera);
  }

  /** Hands close-up: ease the view's centre toward the visible hands. */
  private follow(): void {
    const pts = this.hands.filter(h => h.g.visible).flatMap(h => h.joints.map(j => j.position));
    if (!pts.length || !this.controls) return;
    const c = pts.reduce((v, p) => v.add(p), new THREE.Vector3()).multiplyScalar(1 / pts.length);
    if (!this.followAt) {
      this.followAt = c.clone();
      this.resetView();
      return;
    }
    const step = c.clone().sub(this.followAt).multiplyScalar(0.18);
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
    const r = FINGER_MM * P[9].distanceTo(P[0]) / 90, up = new THREE.Vector3(0, 1, 0);
    const segR: Record<number, number> = {};
    SEGMENTS.forEach(([a, b, t], k) => {
      segR[a] = Math.max(segR[a] || 0, t); segR[b] = Math.max(segR[b] || 0, t * 0.92);
      const m = H.segs[k], A = P[a], dir = new THREE.Vector3().subVectors(P[b], A), len = dir.length();
      m.position.copy(A).addScaledVector(dir, 0.5);
      if (len > 1e-6) m.quaternion.setFromUnitVectors(up, dir.normalize());
      m.scale.set(r * t, len, r * t);
    });
    H.joints.forEach((m, j) => { m.position.copy(P[j]); m.scale.setScalar(r * (segR[j] || 1)); });
    const n = new THREE.Vector3().subVectors(P[5], P[0]).cross(new THREE.Vector3().subVectors(P[17], P[0])).normalize();
    const c = PALM.reduce((v, j) => v.add(P[j]), new THREE.Vector3()).multiplyScalar(1 / PALM.length);
    const pts = PALM.flatMap(j => {
      const q = P[j].clone().lerp(c, -0.05);
      return [q.clone().addScaledVector(n, r * 1.1), q.clone().addScaledVector(n, -r * 1.1), q];
    });
    H.palm.geometry.dispose();
    const hull = new ConvexGeometry(pts);
    hull.deleteAttribute('normal'); hull.deleteAttribute('uv');
    H.palm.geometry = mergeVertices(hull, 1e-3);
    H.palm.geometry.computeVertexNormals();
  }
}
