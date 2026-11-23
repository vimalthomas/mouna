import {
  AfterViewInit, Component, ElementRef, EventEmitter, Input, NgZone, OnChanges, OnDestroy, Output, SimpleChanges, ViewChild, inject,
} from '@angular/core';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { environment } from '../../../../environments/environment';

/** One frame of a sign's skin data (scripts/build_sign_skin.py). */
interface SkinFrame {
  /** hands: 21 points in video pixels (x, y, relative depth), or null */
  h: ([number, number, number][] | null)[];
  /** each hand's 3D shape in millimetres (same axes as the video) */
  w: ([number, number, number][] | null)[];
  /** upper-body pose points 0-24 (video pixels + depth) */
  p: [number, number, number][] | null;
  /** face mesh: 478 points (video pixels + depth) */
  f: [number, number, number][] | null;
}
interface SkinData { width: number; height: number; fps: number; n_frames: number; frames: SkinFrame[]; }

// finger segments (parent, child, relative thickness); palm points
const SEGMENTS: [number, number, number][] = [[1, 2, 1.05], [2, 3, 0.95], [3, 4, 0.85], [5, 6, 0.92], [6, 7, 0.82], [7, 8, 0.74],
  [9, 10, 0.95], [10, 11, 0.84], [11, 12, 0.75], [13, 14, 0.9], [14, 15, 0.8], [15, 16, 0.72], [17, 18, 0.8], [18, 19, 0.72], [19, 20, 0.65]];
const PALM = [0, 1, 2, 5, 9, 13, 17];
const FINGER_MM = 10; // finger radius; an adult finger is ~18-22 mm across
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

/**
 * v3: the sign's face and hands as a shaded 3D surface, following the
 * example video frame by frame (promoted from the fidelity lab). The face is
 * the tracker's own 478-point mesh; the hands are tapered finger segments on
 * a solid palm, shaped from each hand's metric 3D landmarks and placed where
 * the hand is in the video. Emits `available` = false when a sign has no
 * skin data, so the caller can fall back to the older 3D hands.
 */
@Component({
  selector: 'app-sign-skin',
  standalone: true,
  template: `
    <div class="relative rounded-xl overflow-hidden border border-border/50" style="aspect-ratio: 4/3; background: radial-gradient(circle at 50% 35%, #2c2a38, #121118);">
      <canvas #canvasRef class="w-full h-full block cursor-grab active:cursor-grabbing" style="touch-action: none"></canvas>
      <span class="absolute top-2 left-2 text-[11px] font-medium text-white/80 bg-black/35 px-2 py-0.5 rounded-md pointer-events-none">Face and hands (3D)</span>
      @if (state === 'loading') {
        <div class="absolute inset-0 flex items-center justify-center text-white/60 text-xs">Loading 3D…</div>
      }
      <button (click)="resetView()" class="absolute bottom-2 right-2 text-[11px] font-semibold px-2.5 py-1 rounded-full bg-white/90 text-[#2b2433]">Front view</button>
    </div>
    <div class="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 text-[11px] text-ink-muted">
      <span>Follows the video · drag to turn · scroll to zoom</span>
      <label class="flex items-center gap-1 cursor-pointer"><input type="checkbox" [checked]="showFace" (change)="showFace = !showFace"> face</label>
      <label class="flex items-center gap-1 cursor-pointer"><input type="checkbox" [checked]="showBody" (change)="showBody = !showBody"> arms and body</label>
    </div>
  `,
})
export class SignSkinComponent implements AfterViewInit, OnChanges, OnDestroy {
  /** Sign whose skin to show. */
  @Input() word = '';
  /** Element containing the example <video> to follow (frame-synced). */
  @Input() videoHost: HTMLElement | null = null;
  @Output() available = new EventEmitter<boolean>();
  @ViewChild('canvasRef') private canvasRef!: ElementRef<HTMLCanvasElement>;

  state: 'loading' | 'ready' | 'none' = 'loading';
  showFace = true;
  showBody = true;

  private zone = inject(NgZone);
  private renderer: THREE.WebGLRenderer | null = null;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(30, 4 / 3, 1, 100000);
  private controls: OrbitControls | null = null;
  private raf = 0;
  private resizeObs: ResizeObserver | null = null;
  private data: SkinData | null = null;
  private scales = [1, 1];
  private faceDepth = 0;
  private frameBox = { cx: 0, cy: 0, size: 100 };
  private startTime = performance.now();
  private loadToken = 0;

  private skinMat = new THREE.MeshPhysicalMaterial({ color: new THREE.Color('#e3b394'), roughness: 0.5, metalness: 0,
    sheen: 0.5, sheenColor: new THREE.Color('#ffd2b8'), sheenRoughness: 0.6 });
  private shirtMat = new THREE.MeshPhysicalMaterial({ color: new THREE.Color('#3a3650'), roughness: 0.9, metalness: 0, sheen: 0.25,
    transparent: true, opacity: 0.88 });
  private hairMat = new THREE.MeshPhysicalMaterial({ color: new THREE.Color('#2a2330'), roughness: 0.75, metalness: 0, sheen: 0.6, sheenColor: new THREE.Color('#6b5a7a') });
  private body: { arms: THREE.Mesh[]; joints: THREE.Mesh[]; torso: THREE.Mesh; neck: THREE.Mesh; traps: THREE.Mesh[]; head: THREE.Mesh; g: THREE.Group } | null = null;
  private faceGeo = new THREE.BufferGeometry();
  private faceMesh: THREE.Mesh | null = null;
  private hands: { g: THREE.Group; segs: THREE.Mesh[]; joints: THREE.Mesh[]; palm: THREE.Mesh }[] = [];

  ngAfterViewInit(): void {
    this.zone.runOutsideAngular(() => this.init());
  }

  ngOnChanges(ch: SimpleChanges): void {
    if (ch['word'] && this.renderer) this.load();
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
    const b = this.frameBox, dist = b.size * 2.1;
    this.camera.position.set(b.cx, b.cy, dist);
    this.camera.near = dist / 50; this.camera.far = dist * 20; this.camera.updateProjectionMatrix();
    this.controls?.target.set(b.cx, b.cy, 0);
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
    this.body = this.makeBody();

    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(canvas);
    this.resize();
    await this.load();
    const loop = () => { this.raf = requestAnimationFrame(loop); this.render(); };
    loop();
  }

  private async load(): Promise<void> {
    const token = ++this.loadToken;
    this.data = null;
    this.setState('loading');
    try {
      const r = await fetch(`${environment.apiBaseUrl}/signs/${encodeURIComponent(this.word)}/skin`);
      if (!r.ok) throw new Error(String(r.status));
      const d: SkinData = await r.json();
      if (token !== this.loadToken) return;
      this.data = d;
      this.scales = [0, 1].map(k => this.handScale(d, k));
      const fw = d.frames.find(f => f.f)?.f;
      this.faceDepth = fw ? (Math.max(...fw.map(p => p[0])) - Math.min(...fw.map(p => p[0]))) * 0.45 : 0;
      this.frame(d);
      this.startTime = performance.now();
      this.setState('ready');
    } catch {
      if (token === this.loadToken) this.setState('none');
    }
  }

  private setState(s: 'loading' | 'ready' | 'none'): void {
    this.zone.run(() => {
      this.state = s;
      if (s !== 'loading') this.available.emit(s === 'ready');
    });
  }

  private makeHand() {
    const g = new THREE.Group();
    const cyl = new THREE.CylinderGeometry(1, 1, 1, 18, 1, true), sph = new THREE.SphereGeometry(1, 18, 14);
    const segs = SEGMENTS.map(() => { const m = new THREE.Mesh(cyl, this.skinMat); g.add(m); return m; });
    const joints = Array.from({ length: 21 }, () => { const m = new THREE.Mesh(sph, this.skinMat); g.add(m); return m; });
    const palm = new THREE.Mesh(new THREE.BufferGeometry(), this.skinMat); g.add(palm);
    this.scene.add(g);
    return { g, segs, joints, palm };
  }

  private makeBody() {
    const g = new THREE.Group();
    const cyl = new THREE.CylinderGeometry(1, 1, 1, 20, 1, true), sph = new THREE.SphereGeometry(1, 20, 16);
    const arms = Array.from({ length: 4 }, () => { const m = new THREE.Mesh(cyl, this.shirtMat); g.add(m); return m; });   // upper L/R, fore L/R
    const joints = Array.from({ length: 6 }, () => { const m = new THREE.Mesh(sph, this.shirtMat); g.add(m); return m; }); // shoulders, elbows, wrists
    const torso = new THREE.Mesh(sph, this.shirtMat); g.add(torso);
    const neck = new THREE.Mesh(cyl, this.skinMat); g.add(neck);
    const traps = [0, 1].map(() => { const m = new THREE.Mesh(cyl, this.shirtMat); g.add(m); return m; });  // shoulders sloping up to the neck
    const head = new THREE.Mesh(sph, this.hairMat); g.add(head);  // back of the head behind the face mesh
    this.scene.add(g);
    return { arms, joints, torso, neck, traps, head, g };
  }

  /** mm -> video pixels for a hand: its typical size on video / in 3D. */
  private handScale(d: SkinData, k: number): number {
    const r: number[] = [];
    for (const f of d.frames) {
      const h = f.h[k], w = f.w?.[k];
      if (!h || !w) continue;
      const a = Math.hypot(h[9][0] - h[0][0], h[9][1] - h[0][1]);
      const b = Math.hypot(w[9][0] - w[0][0], w[9][1] - w[0][1], w[9][2] - w[0][2]);
      if (b > 1) r.push(a / b);
    }
    r.sort((x, y) => x - y);
    return r.length ? r[Math.floor(r.length / 2)] : 1;
  }

  private frame(d: SkinData): void {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    const add = (p: number[]) => { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); };
    for (const f of d.frames) { f.f?.forEach(add); f.h.forEach(h => h?.forEach(add)); }
    if (!isFinite(x0)) { x0 = 0; y0 = 0; x1 = d.width; y1 = d.height; }
    this.frameBox = { cx: (x0 + x1) / 2, cy: -(y0 + y1) / 2, size: Math.max(x1 - x0, y1 - y0) };
    this.resetView();
  }

  private resize(): void {
    if (!this.renderer) return;
    const c = this.canvasRef.nativeElement, w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  /** Frame to show: the example video's current frame, or our own clock. */
  private frameIndex(d: SkinData): number {
    const fixed = new URLSearchParams(location.search).get('skinFrame');
    if (fixed !== null) return Math.min(d.n_frames - 1, Math.max(0, +fixed));
    const v = this.videoHost?.querySelector('video');
    const t = v && v.readyState >= 1 ? v.currentTime : ((performance.now() - this.startTime) / 1000) % (d.n_frames / d.fps);
    return Math.min(d.n_frames - 1, Math.max(0, Math.round(t * d.fps)));
  }

  private render(): void {
    if (!this.renderer) return;
    const d = this.data;
    if (d && this.faceMesh) {
      const f = d.frames[this.frameIndex(d)];
      const pose = f.p;
      const sw = pose ? Math.hypot(pose[11][0] - pose[12][0], pose[11][1] - pose[12][1]) || 1 : 1;
      // depth toward the camera (+z), shoulders at 0 (see POSE_DEPTH)
      const midZ = pose ? (pose[11][2] + pose[12][2]) / 2 : 0;
      const depth = (q: number[]) => -(q[2] - midZ) * POSE_DEPTH;
      this.faceMesh.visible = this.showFace && !!f.f;
      if (f.f) {
        const pos = this.faceGeo.attributes['position'] as THREE.BufferAttribute;
        const noseZ = pose ? Math.min(Math.max(depth(pose[0]), 0.15 * sw), 0.7 * sw) : -this.faceDepth, n1 = f.f[1][2] || 0;
        f.f.forEach((p, k) => pos.setXYZ(k, p[0], -p[1], noseZ - ((p[2] || 0) - n1)));
        pos.needsUpdate = true;
        this.faceGeo.computeVertexNormals();
      }
      const wristAt: (THREE.Vector3 | null)[] = [null, null];  // where each pose wrist (15, 16) meets a hand
      for (let k = 0; k < 2; k++) {
        const h = f.h[k];
        let wz = 0;
        if (pose && h) {
          const pk = Math.hypot(pose[15][0] - h[0][0], pose[15][1] - h[0][1]) < Math.hypot(pose[16][0] - h[0][0], pose[16][1] - h[0][1]) ? 15 : 16;
          // hands can't sit far behind the face or far out in front of it
          wz = Math.min(Math.max(depth(pose[pk]), -0.3 * sw), 1.6 * sw);  // a hand can't reach further than an arm
          if (Math.hypot(pose[pk][0] - h[0][0], pose[pk][1] - h[0][1]) < 0.6 * sw) wristAt[pk - 15] = new THREE.Vector3(h[0][0], -h[0][1], wz);
        }
        try { this.updateHand(this.hands[k], h, f.w?.[k] ?? null, this.scales[k], wz); }
        catch { this.hands[k].g.visible = false; }
      }
      this.updateBody(pose, wristAt, sw, f.f, depth);
    } else {
      this.hands.forEach(h => (h.g.visible = false));
      if (this.faceMesh) this.faceMesh.visible = false;
      if (this.body) this.body.g.visible = false;
    }
    this.controls?.update();
    this.renderer.render(this.scene, this.camera);
  }

  /** Shoulders, upper arms and forearms (as a shirt), torso and neck; each
   * forearm ends at its tracked hand's wrist so the hands sit on the arms. */
  private updateBody(pose: number[][] | null, wristAt: (THREE.Vector3 | null)[], sw: number, face: number[][] | null,
                     depth: (q: number[]) => number): void {
    const B = this.body;
    if (!B) return;
    if (!pose || !this.showBody) { B.g.visible = false; return; }
    B.g.visible = true;
    const v = (q: number[]) => new THREE.Vector3(q[0], -q[1], Math.min(Math.max(depth(q), -0.5 * sw), 1.6 * sw));
    const up = new THREE.Vector3(0, 1, 0);
    const S = [v(pose[11]), v(pose[12])], E = [v(pose[13]), v(pose[14])];
    const W = [wristAt[0] ?? v(pose[15]), wristAt[1] ?? v(pose[16])];
    const tube = (m: THREE.Mesh, a: THREE.Vector3, b: THREE.Vector3, r: number) => {
      const dir = new THREE.Vector3().subVectors(b, a), len = dir.length();
      m.position.copy(a).addScaledVector(dir, 0.5);
      if (len > 1e-6) m.quaternion.setFromUnitVectors(up, dir.normalize());
      m.scale.set(r, len, r);
    };
    const ru = 0.1 * sw, rf = 0.08 * sw;
    for (let k = 0; k < 2; k++) {
      tube(B.arms[k], S[k], E[k], ru);
      tube(B.arms[2 + k], E[k], W[k], rf);
      B.joints[k].position.copy(S[k]); B.joints[k].scale.setScalar(ru * 1.25);
      B.joints[2 + k].position.copy(E[k]); B.joints[2 + k].scale.setScalar(ru * 0.95);
      B.joints[4 + k].position.copy(W[k]); B.joints[4 + k].scale.setScalar(rf * 0.9);
    }
    const mid = S[0].clone().add(S[1]).multiplyScalar(0.5);
    const neckBase = mid.clone().add(new THREE.Vector3(0, 0.12 * sw, -0.1 * sw));
    B.torso.position.copy(mid).add(new THREE.Vector3(0, -0.45 * sw, -0.18 * sw));
    B.torso.scale.set(0.5 * sw, 0.6 * sw, 0.17 * sw);
    for (let k = 0; k < 2; k++) tube(B.traps[k], S[k], neckBase, 0.12 * sw);
    if (face) {
      // the face mesh has no back or hair: put a head behind it
      const xs = face.map(p => p[0]), ys = face.map(p => p[1]);
      const fw = Math.max(...xs) - Math.min(...xs), fh = Math.max(...ys) - Math.min(...ys);
      const top = new THREE.Vector3(face[10][0], -face[10][1], 0), chinV = new THREE.Vector3(face[152][0], -face[152][1], 0);
      const c = top.clone().lerp(chinV, 0.42);
      B.head.visible = this.showFace;
      // the face's sides (at the ears) sit ~0.7 face-widths behind the nose, so the head's centre goes about there
      B.head.position.set(c.x, c.y + 0.04 * fh, (this.faceMesh?.geometry.attributes['position'] as THREE.BufferAttribute).getZ(1) - 0.85 * fw);
      B.head.scale.set(0.56 * fw, 0.62 * fh, 0.55 * fw);
      tube(B.neck, neckBase, new THREE.Vector3(chinV.x, chinV.y + 0.15 * fh, B.head.position.z + 0.1 * fw), 0.15 * sw);
    } else {
      B.head.visible = false;
      tube(B.neck, neckBase, neckBase.clone().add(new THREE.Vector3(0, 0.3 * sw, 0)), 0.15 * sw);
    }
  }

  private updateHand(H: { g: THREE.Group; segs: THREE.Mesh[]; joints: THREE.Mesh[]; palm: THREE.Mesh },
                     img: number[][] | null, world: number[][] | null, scale: number, wristZ = 0): void {
    if (!img || !world) { H.g.visible = false; return; }
    H.g.visible = true;
    const w0 = world[0], i0 = img[0];
    const P = world.map(p => new THREE.Vector3(i0[0] + (p[0] - w0[0]) * scale, -(i0[1] + (p[1] - w0[1]) * scale), wristZ - (p[2] - w0[2]) * scale));
    const r = FINGER_MM * scale, up = new THREE.Vector3(0, 1, 0);
    const segR: Record<number, number> = {};
    SEGMENTS.forEach(([a, b, t], k) => {
      segR[a] = Math.max(segR[a] || 0, t); segR[b] = Math.max(segR[b] || 0, t * 0.92);
      const m = H.segs[k], A = P[a], dir = new THREE.Vector3().subVectors(P[b], A), len = dir.length();
      m.position.copy(A).addScaledVector(dir, 0.5);
      m.quaternion.setFromUnitVectors(up, dir.normalize());
      m.scale.set(r * t, len, r * t);
    });
    H.joints.forEach((m, j) => { m.position.copy(P[j]); m.scale.setScalar(r * (segR[j] || 1)); });
    const n = new THREE.Vector3().subVectors(P[5], P[0]).cross(new THREE.Vector3().subVectors(P[17], P[0])).normalize();
    const c = PALM.reduce((v, j) => v.add(P[j]), new THREE.Vector3()).multiplyScalar(1 / PALM.length);
    const pts = PALM.flatMap(j => {
      const q = P[j].clone().lerp(c, -0.12);
      return [q.clone().addScaledVector(n, r * 1.25), q.clone().addScaledVector(n, -r * 1.25), q];
    });
    H.palm.geometry.dispose();
    const hull = new ConvexGeometry(pts);
    hull.deleteAttribute('normal'); hull.deleteAttribute('uv');
    H.palm.geometry = mergeVertices(hull, 1e-3);
    H.palm.geometry.computeVertexNormals();
  }
}
