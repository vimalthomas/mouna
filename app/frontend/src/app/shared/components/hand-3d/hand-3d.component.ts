import { AfterViewInit, Component, ElementRef, Input, NgZone, OnChanges, OnDestroy, ViewChild, inject } from '@angular/core';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ParameterResult, SkeletonFrame } from '../../../core/models/sign.model';

type V3 = [number, number, number];

const BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20],
  // palm-edge bones so the slab's rim reads as rounded, not a cut edge
  [0, 5], [5, 9], [9, 13], [13, 17], [0, 17],
];
const PALM_OUTLINE = [0, 1, 5, 9, 13, 17];
const PALM_CENTER_IDX = [0, 5, 9, 13, 17];
// Joint radius in hand-local units (wrist->middle-knuckle = 1). Index by
// landmark: wrist/knuckles thickest, tapering to fingertips.
const JOINT_R = [0.16, 0.12, 0.105, 0.095, 0.085, 0.11, 0.095, 0.085, 0.075, 0.115, 0.1, 0.09, 0.078, 0.11, 0.095, 0.085, 0.074, 0.1, 0.085, 0.075, 0.066];
const PALM_HALF_THICKNESS = 0.09;
const LOOP_SECONDS = 3.4; // same loop length as the 2D skeleton panels at 1x

const SKIN = 0xd8b49a;
const SKIN_REF = 0xb9bfcc;
const PALM_SIDE = 0x2bb3a3;
const WARN = 0xf59e0b;

/**
 * Hands-only 3D: your dominant hand and the target's, side by side in ONE
 * scene with one orbit control, so rotating to inspect palm direction keeps
 * both in the same view. Uses hand-local MediaPipe xyz (z is consistent
 * within a single hand model; see mouna.coach.skeleton). Expects the
 * CANONICAL attempt (same handedness as the reference).
 *
 * Palm side: the "right_hand" arrays' palm normal is +cross(index-wrist,
 * pinky-wrist) in (x right, y up, z toward camera) coords, and the opposite
 * for "left_hand" -- measured on the 23/17 reference signs with that hand
 * (fingertips curl to that side in 96% / 88% of them), not assumed.
 */
@Component({
  selector: 'app-hand-3d',
  standalone: true,
  template: `
    <div class="relative rounded-xl overflow-hidden bg-surface-sunk border border-border/50" [style.aspect-ratio]="targetOnly ? '4/3' : '16/9'">
      <canvas #canvasRef class="w-full h-full block cursor-grab active:cursor-grabbing"></canvas>
      <div class="absolute top-2 left-0 right-0 flex justify-around pointer-events-none">
        @if (!targetOnly) { <span class="text-[11px] font-medium text-ink-muted bg-surface/85 px-2 py-0.5 rounded-md">Your hand</span> }
        <span class="text-[11px] font-medium text-ink-muted bg-surface/85 px-2 py-0.5 rounded-md">{{ targetOnly ? 'Hand shape (3D)' : 'Target hand' }}</span>
      </div>
      @if (!hasData) {
        <div class="absolute inset-0 flex items-center justify-center text-ink-faint text-xs px-4 text-center">No hand data</div>
      }
      <button (click)="resetView()" class="absolute bottom-2 right-2 text-[11px] font-semibold px-2.5 py-1 rounded-full bg-surface/90 border border-border/50 text-ink-muted">Reset view</button>
    </div>
    <div class="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 text-[11px] text-ink-muted">
      <span>Drag to rotate · scroll to zoom</span>
      <span class="flex items-center gap-1.5"><span class="inline-block w-2.5 h-2.5 rounded-sm" [style.background]="palmCss"></span> palm side</span>
      @if (mismatch('handshape')) { <span class="flex items-center gap-1.5"><span class="inline-block w-2.5 h-2.5 rounded-sm" style="background:#f59e0b"></span> handshape differs</span> }
      @if (mismatch('orientation')) { <span>palm color amber = palm direction differs</span> }
    </div>
  `,
})
export class Hand3dComponent implements AfterViewInit, OnChanges, OnDestroy {
  @Input() attemptFrames: SkeletonFrame[] | null = null;
  @Input() referenceFrames: SkeletonFrame[] | null = null;
  @Input() parameters: Record<string, ParameterResult> | null = null;
  @Input() speedMultiplier = 1;
  /** Prep mode: only the target hand, centered (no attempt yet). */
  @Input() targetOnly = false;

  @ViewChild('canvasRef') private canvasRef!: ElementRef<HTMLCanvasElement>;
  private zone = inject(NgZone);

  private renderer: THREE.WebGLRenderer | null = null;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(32, 16 / 9, 0.1, 100);
  private controls: OrbitControls | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private raf = 0;
  private lastTs = 0;
  private progress = 0;
  private mine: HandPair | null = null;
  private target: HandPair | null = null;
  private mineSeq: PairSeq = EMPTY_SEQ;
  private targetSeq: PairSeq = EMPTY_SEQ;

  get hasData(): boolean { return this.mineSeq.frames.length > 0 || this.targetSeq.frames.length > 0; }
  get palmCss(): string { return this.mismatch('orientation') ? '#f59e0b' : '#2bb3a3'; }
  mismatch(key: string): boolean { return this.parameters?.[key]?.match === false; }

  ngAfterViewInit(): void {
    const canvas = this.canvasRef.nativeElement;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(-2, 3, 5);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.45);
    fill.position.set(3, -1, -4);
    this.scene.add(fill);

    this.mine = new HandPair();
    this.target = new HandPair();
    this.mine.group.position.x = this.targetOnly ? 0 : -1.7;
    this.target.group.position.x = this.targetOnly ? 0 : 1.7;
    this.mine.group.visible = !this.targetOnly;
    this.fitScale();
    this.scene.add(this.mine.group, this.target.group);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enablePan = false;
    this.controls.enableDamping = true;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 16;
    this.resetView();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
    this.applyInputs();

    this.zone.runOutsideAngular(() => {
      const loop = (ts: number) => {
        const dt = this.lastTs ? (ts - this.lastTs) / 1000 : 0;
        this.lastTs = ts;
        this.progress = (this.progress + (dt * this.speedMultiplier) / LOOP_SECONDS) % 1;
        this.tick();
        this.raf = requestAnimationFrame(loop);
      };
      this.raf = requestAnimationFrame(loop);
    });
  }

  ngOnChanges(): void {
    // Sequences are computed here (before the view is checked) so the
    // template's hasData doesn't change mid-check.
    this.mineSeq = buildSeq(this.attemptFrames);
    this.targetSeq = buildSeq(this.referenceFrames);
    if (this.renderer) { this.applyInputs(); this.fitScale(); }
  }

  resetView(): void {
    this.camera.position.set(0, 0.2, this.targetOnly ? 5 : 7.5);
    this.controls?.target.set(0, 0, 0);
    this.controls?.update();
  }

  /** Two-handed signs are wider than one hand: shrink each side to fit its
   * slot (whole canvas in target-only mode, half of it when comparing). */
  private fitScale(): void {
    // Visible box at the default camera distance (fov 32): ~16:9 canvas.
    const roomW = this.targetOnly ? 3.6 : 3.0;
    const roomH = this.targetOnly ? 2.7 : 3.6;
    // Allow growing past 1:1 so a small pair still fills its panel.
    const fit = (q: PairSeq) => Math.min(1.8, roomW / Math.max(q.width, 0.01), roomH / Math.max(q.height, 0.01));
    if (this.mine) this.mine.group.scale.setScalar(fit(this.mineSeq));
    if (this.target) this.target.group.scale.setScalar(fit(this.targetSeq));
  }

  private applyInputs(): void {
    const handshapeBad = this.mismatch('handshape');
    const orientBad = this.mismatch('orientation');
    this.mine?.setColors(handshapeBad ? WARN : SKIN, orientBad ? WARN : PALM_SIDE);
    this.target?.setColors(SKIN_REF, PALM_SIDE);
  }

  private resize(): void {
    if (!this.renderer) return;
    const c = this.canvasRef.nativeElement;
    const w = c.clientWidth || 1;
    const h = c.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private tick(): void {
    if (!this.renderer) return;
    if (!this.targetOnly) this.mine?.update(sample(this.mineSeq, this.progress));
    this.target?.update(sample(this.targetSeq, this.progress));
    this.controls?.update();
    this.renderer.render(this.scene, this.camera);
  }

  ngOnDestroy(): void {
    cancelAnimationFrame(this.raf);
    this.resizeObserver?.disconnect();
    this.controls?.dispose();
    this.mine?.dispose();
    this.target?.dispose();
    this.renderer?.dispose();
  }
}

// ---- data helpers ----

/** One hand's pose; sign flips the palm normal for "left_hand" arrays (see
 * component doc). */
type HandPose = { pts: V3[]; sign: 1 | -1 };
type PairFrame = { right: HandPose | null; left: HandPose | null };
/** width: rough horizontal extent in scene units, for fitting the view. */
type PairSeq = { frames: PairFrame[]; two: boolean; width: number; height: number };
const EMPTY_SEQ: PairSeq = { frames: [], two: false, width: 2, height: 2 };

const toPose = (h: [number, number, number][], sign: 1 | -1): HandPose => ({ pts: h.map(p => toScene(p as V3)), sign });

/**
 * One-handed sign (or one hand mostly missing): the dominant hand alone,
 * centered on its palm -- as before. Two-handed sign (both hands in >=30%
 * of frames): both hands, each wrist placed at its 2D body-relative
 * position converted to hand units (hand-local 3D shape is exact; the
 * between-hands depth isn't reliable, so both sit at the same depth).
 */
function buildSeq(frames: SkeletonFrame[] | null): PairSeq {
  if (!frames || frames.length === 0) return EMPTY_SEQ;
  const n = frames.length;
  const r = frames.filter(f => f.right_hand_3d).length;
  const l = frames.filter(f => f.left_hand_3d).length;
  const two = r >= 0.3 * n && l >= 0.3 * n;

  if (!two) {
    const useRight = r >= l;
    const out: PairFrame[] = [];
    for (const f of frames) {
      const h = useRight ? f.right_hand_3d : f.left_hand_3d;
      if (h) out.push({ right: centerOnPalm(toPose(h, useRight ? 1 : -1)), left: null });
    }
    return { frames: out, two: false, width: 2.2, height: 2.4 };
  }

  // Hand length in body units (wrist -> middle knuckle, 2D) to convert
  // body-relative wrist offsets into the hand-local scale.
  const lens: number[] = [];
  for (const f of frames) for (const h of [f.right_hand, f.left_hand]) if (h) lens.push(Math.hypot(h[9][0] - h[0][0], h[9][1] - h[0][1]));
  lens.sort((a, b) => a - b);
  const L = lens.length ? Math.max(lens[Math.floor(lens.length / 2)], 0.05) : 0.25;

  const place = (h3: [number, number, number][] | null | undefined, h2: [number, number][] | null, sign: 1 | -1): HandPose | null => {
    if (!h3 || !h2) return null;
    const ox = h2[0][0] / L;
    const oy = -h2[0][1] / L;
    return { pts: h3.map(p => { const q = toScene(p as V3); return [q[0] + ox, q[1] + oy, q[2]] as V3; }), sign };
  };

  let lastR: HandPose | null = null;
  let lastL: HandPose | null = null;
  const out: PairFrame[] = frames.map(f => {
    lastR = place(f.right_hand_3d, f.right_hand, 1) ?? lastR;
    lastL = place(f.left_hand_3d, f.left_hand, -1) ?? lastL;
    return { right: lastR, left: lastL };
  });

  // Center each frame on its own pair bounding box: the video already shows
  // where the hands travel; this view is for shape, palm direction and the
  // two hands' relation to each other, so keep them centered and large.
  let maxW = 0, maxH = 0;
  const shifted = out.map(fr => {
    const hs = [fr.right, fr.left].filter((h): h is HandPose => !!h);
    if (!hs.length) return fr;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const h of hs) for (const p of h.pts) { minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]); minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]); }
    maxW = Math.max(maxW, maxX - minX); maxH = Math.max(maxH, maxY - minY);
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const shift = (h: HandPose | null) => h ? { ...h, pts: h.pts.map(p => [p[0] - cx, p[1] - cy, p[2]] as V3) } : null;
    return { right: shift(fr.right), left: shift(fr.left) };
  });
  return { frames: shifted, two: true, width: Math.max(2.2, maxW), height: Math.max(2.4, maxH) };
}

function centerOnPalm(h: HandPose): HandPose {
  const idx = PALM_CENTER_IDX;
  const c = idx.reduce((acc, i) => [acc[0] + h.pts[i][0], acc[1] + h.pts[i][1], acc[2] + h.pts[i][2]], [0, 0, 0]).map(v => v / idx.length);
  return { ...h, pts: h.pts.map(p => [p[0] - c[0], p[1] - c[1], p[2] - c[2]] as V3) };
}

/** MediaPipe (x right, y down, z away from camera) -> scene (x right, y up,
 * z toward viewer): a 180-degree rotation about x, so handedness is kept. */
function toScene(p: V3): V3 { return [p[0], -p[1], -p[2]]; }

function lerpPose(a: HandPose | null, b: HandPose | null, k: number): HandPose | null {
  if (!a || !b) return a ?? b;
  return { pts: a.pts.map((p, j) => [p[0] + (b.pts[j][0] - p[0]) * k, p[1] + (b.pts[j][1] - p[1]) * k, p[2] + (b.pts[j][2] - p[2]) * k] as V3), sign: a.sign };
}

function sample(seq: PairSeq, progress: number): PairFrame | null {
  const f = seq.frames;
  if (f.length === 0) return null;
  if (f.length === 1) return f[0];
  const x = progress * (f.length - 1);
  const i = Math.floor(x);
  const k = x - i;
  const a = f[i];
  const b = f[Math.min(i + 1, f.length - 1)];
  return { right: lerpPose(a.right, b.right, k), left: lerpPose(a.left, b.left, k) };
}

/** Both hands of one side (you or target). */
class HandPair {
  readonly group = new THREE.Group();
  private right = new HandRig();
  private left = new HandRig();
  constructor() { this.group.add(this.right.group, this.left.group); }
  setColors(skin: number, palm: number): void { this.right.setColors(skin, palm); this.left.setColors(skin, palm); }
  update(frame: PairFrame | null): void {
    this.right.update(frame?.right ?? null);
    this.left.update(frame?.left ?? null);
  }
  dispose(): void { this.right.dispose(); this.left.dispose(); }
}

// ---- rig: meshes are built once and only re-posed per frame ----

class HandRig {
  readonly group = new THREE.Group();
  private skinMat = new THREE.MeshStandardMaterial({ color: SKIN, roughness: 0.6, metalness: 0.0 });
  private palmMat = new THREE.MeshStandardMaterial({ color: PALM_SIDE, roughness: 0.6, side: THREE.DoubleSide });
  private joints: THREE.Mesh[] = [];
  private bones: THREE.Mesh[] = [];
  private palmFront: THREE.Mesh;
  private palmBack: THREE.Mesh;
  private sphereGeo = new THREE.SphereGeometry(1, 20, 14);
  private cylGeo = new THREE.CylinderGeometry(1, 1, 1, 16, 1, true);
  private readonly up = new THREE.Vector3(0, 1, 0);

  constructor() {
    for (let i = 0; i < 21; i++) {
      const m = new THREE.Mesh(this.sphereGeo, this.skinMat);
      m.scale.setScalar(JOINT_R[i]);
      this.joints.push(m);
      this.group.add(m);
    }
    for (let b = 0; b < BONES.length; b++) {
      const m = new THREE.Mesh(this.cylGeo, this.skinMat);
      this.bones.push(m);
      this.group.add(m);
    }
    this.palmFront = new THREE.Mesh(fanGeometry(), this.palmMat);
    this.palmBack = new THREE.Mesh(fanGeometry(), this.skinMat);
    this.group.add(this.palmFront, this.palmBack);
    this.group.visible = false;
  }

  setColors(skin: number, palm: number): void {
    this.skinMat.color.setHex(skin);
    this.palmMat.color.setHex(palm);
  }

  update(frame: { pts: V3[]; sign: 1 | -1 } | null): void {
    if (!frame) { this.group.visible = false; return; }
    this.group.visible = true;
    const v = frame.pts.map(p => new THREE.Vector3(p[0], p[1], p[2]));
    // Positions arrive already placed (palm-centered for one hand, body-
    // relative for a pair -- see buildSeq), so no per-hand recentering here.

    v.forEach((p, i) => this.joints[i].position.copy(p));
    BONES.forEach(([a, b], k) => {
      const dir = new THREE.Vector3().subVectors(v[b], v[a]);
      const len = dir.length() || 1e-3;
      const m = this.bones[k];
      m.position.copy(v[a]).addScaledVector(dir, 0.5);
      m.quaternion.setFromUnitVectors(this.up, dir.divideScalar(len));
      const r = Math.min(JOINT_R[a], JOINT_R[b]);
      m.scale.set(r, len, r);
    });

    const n = new THREE.Vector3().crossVectors(new THREE.Vector3().subVectors(v[5], v[0]), new THREE.Vector3().subVectors(v[17], v[0])).normalize().multiplyScalar(frame.sign);
    setFan(this.palmFront.geometry, PALM_OUTLINE.map(i => v[i].clone().addScaledVector(n, PALM_HALF_THICKNESS)));
    setFan(this.palmBack.geometry, PALM_OUTLINE.map(i => v[i].clone().addScaledVector(n, -PALM_HALF_THICKNESS)));
  }

  dispose(): void {
    this.sphereGeo.dispose();
    this.cylGeo.dispose();
    this.palmFront.geometry.dispose();
    this.palmBack.geometry.dispose();
    this.skinMat.dispose();
    this.palmMat.dispose();
  }
}

/** Triangle fan over the palm outline (+ its centroid), positions filled per frame. */
function fanGeometry(): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  const n = PALM_OUTLINE.length;
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array((n + 1) * 3), 3));
  const idx: number[] = [];
  for (let i = 0; i < n; i++) idx.push(n, i, (i + 1) % n);
  g.setIndex(idx);
  return g;
}

function setFan(g: THREE.BufferGeometry, outline: THREE.Vector3[]): void {
  const pos = g.getAttribute('position') as THREE.BufferAttribute;
  const c = new THREE.Vector3();
  outline.forEach((p, i) => { pos.setXYZ(i, p.x, p.y, p.z); c.add(p); });
  c.divideScalar(outline.length);
  pos.setXYZ(outline.length, c.x, c.y, c.z);
  pos.needsUpdate = true;
  g.computeVertexNormals();
  g.computeBoundingSphere();
}
