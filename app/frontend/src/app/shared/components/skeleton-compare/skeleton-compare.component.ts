import { AfterViewInit, Component, ElementRef, Input, OnChanges, OnDestroy, ViewChild } from '@angular/core';
import * as THREE from 'three';
import { ParameterResult, SkeletonFrame } from '../../../core/models/sign.model';

// MediaPipe Hands landmark topology -- fixed, not sign-specific. Fingers
// start at their OWN knuckle (MCP), not the wrist -- all 5 sharing the
// wrist as a start point made them fan out from one spot like claws. The
// wrist-to-knuckle gap is real palm, covered by the palm polygon instead.
const FINGER_CHAINS: Record<string, number[]> = {
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};
// Palm outline: wrist + the five knuckle bases, in order across the hand --
// traces the actual palm shape instead of a circle that didn't match it.
const PALM_OUTLINE = [0, 1, 5, 9, 13, 17];
// Face contours drawn as raised 3D features, not flat decals.
const FACE_CONTOUR_KEYS: (keyof NonNullable<SkeletonFrame['face']>)[] = [
  'left_eye', 'right_eye', 'lips_outer', 'lips_inner',
];

const HEAD_RADIUS = 0.34;
const FRAME_MS = 140;
// Fixed cosmetic torso thickness -- NOT a real tracked depth value (we only
// have 2D video, see this session's VRM retargeting post-mortem on why
// fabricating real depth from flat data broke things). This is just a
// rounding/bevel applied to the already-2D silhouette shape so it reads as
// a solid form under lighting, same honest-depth rule the whole scene follows.
const TORSO_DEPTH = 0.26;
const FACE_Z_OFFSET = TORSO_DEPTH / 2 + 0.03; // just in front of the torso's own front face, no z-fighting

@Component({
  selector: 'app-skeleton-compare',
  standalone: true,
  template: `
    <div class="relative rounded-xl overflow-hidden bg-surface-sunk border border-border/50" [style.aspect-ratio]="mode === 'body' ? '4/4.4' : '16/10'">
      <canvas #canvasRef class="w-full h-full block"></canvas>

      @if (label) {
        <div class="absolute top-2 left-2 text-[10px] font-semibold uppercase tracking-wide text-ink-faint bg-surface/80 px-2 py-0.5 rounded">{{ label }}</div>
      }
      @if (!frame) {
        <div class="absolute inset-0 flex items-center justify-center text-ink-faint text-xs px-4 text-center">No skeleton data</div>
      }
    </div>
  `,
})
export class SkeletonCompareComponent implements OnChanges, AfterViewInit, OnDestroy {
  @Input() frames: SkeletonFrame[] | null = null;
  @Input() parameters: Record<string, ParameterResult> | null = null;
  @Input() label: string | null = null;
  @Input() animate = true;
  @Input() mode: 'body' | 'hands' | 'face' = 'body';
  /** Base/idle color for this panel -- the only thing that should visually
   * distinguish one panel from another now that panels don't overlay each
   * other; mismatch coloring (amber) still overrides this per-segment. */
  @Input() color = '#3f5fc4';
  /** 0.5 = half speed, 1 = normal, 1.5+ = faster -- same control as the
   * Preview screen's video speed buttons, applied to the animation loop. */
  @Input() speedMultiplier = 1;

  @ViewChild('canvasRef') private canvasRef!: ElementRef<HTMLCanvasElement>;

  private readonly HAND_ZOOM_RADIUS = 1.1;
  private readonly FACE_ZOOM_RADIUS = 0.55;
  private readonly BODY_CENTER: [number, number] = [0, 0.2];
  private readonly BODY_RADIUS_Y = 2.8;

  readonly HEAD_RADIUS = HEAD_RADIUS;

  private progress = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  frame: SkeletonFrame | null = null;

  readonly warnColor = '#ab7a14';

  get bodyColor(): string { return this.color; }
  get locationColor(): string { return this.mismatch('location') ? this.warnColor : this.color; }
  get handshapeColor(): string { return this.mismatch('handshape') ? this.warnColor : this.color; }
  get orientationColor(): string { return this.mismatch('orientation') ? this.warnColor : this.color; }
  /** Reacts to a future `parameters['facial']` result the same way the
   * other three already do -- harmless no-op today since the facial NMS
   * check runs through a separate check_facial_step flow, not the main
   * `parameters` object, but wired here so turning that on later is a
   * one-line change elsewhere, not a template change. */
  get eyebrowColor(): string { return this.mismatch('facial') ? this.warnColor : this.color; }

  private mismatch(key: string): boolean {
    return this.parameters?.[key]?.match === false;
  }

  // ---- Three.js scene plumbing ----
  private renderer: THREE.WebGLRenderer | null = null;
  private scene: THREE.Scene | null = null;
  private camera: THREE.OrthographicCamera | null = null;
  private dynamicGroup: THREE.Group | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private viewInitialized = false;

  ngAfterViewInit(): void {
    this.initScene();
    this.viewInitialized = true;
    this.renderFrame();
  }

  private initScene(): void {
    const canvas = this.canvasRef.nativeElement;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.scene = new THREE.Scene();
    this.dynamicGroup = new THREE.Group();
    this.scene.add(this.dynamicGroup);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(-1.2, 1.6, 2.2);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xffffff, 0.3);
    rim.position.set(1.5, -0.6, 1.0);
    this.scene.add(rim);

    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
    this.camera.position.set(0, 0, 10);

    this.resizeObserver = new ResizeObserver(() => this.resizeToContainer());
    this.resizeObserver.observe(canvas);
    this.resizeToContainer();
  }

  private resizeToContainer(): void {
    if (!this.renderer || !this.canvasRef) return;
    const canvas = this.canvasRef.nativeElement;
    const w = canvas.clientWidth || 1;
    const h = canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.renderFrame();
  }

  // ---- coordinate helpers ----
  // SVG-space point (y grows downward) -> Three.js scene point (y grows
  // upward). z stays 0 for anything we don't have a real tracked depth
  // value for -- see TORSO_DEPTH comment on why we don't fabricate one.
  private toVec3(p: [number, number], z = 0): THREE.Vector3 {
    return new THREE.Vector3(p[0], -p[1], z);
  }

  /** Centers on the actual face-oval centroid each frame when face data
   * exists (more accurate than the nose-derived headCenter estimate used
   * for the plain-circle fallback), falling back to headCenter for frames/
   * words without facial reference data so the zoom doesn't jump wildly
   * between "has face" and "doesn't" frames in the same clip. */
  private faceZoomCenter(): [number, number] | null {
    if (!this.frame) return null;
    const oval = this.frame.face?.face_oval;
    if (oval && oval.length > 0) {
      const x = oval.reduce((s, p) => s + p[0], 0) / oval.length;
      const y = oval.reduce((s, p) => s + p[1], 0) / oval.length;
      return [x, y];
    }
    return this.headCenter(this.frame);
  }

  /** Tracks wherever the hand(s) actually are each frame, so "hands" mode
   * stays framed on the hand regardless of where the sign places it. */
  private handsCenter(): [number, number] | null {
    if (!this.frame) return null;
    const points = [this.frame.left_hand?.[0], this.frame.right_hand?.[0]].filter((p): p is [number, number] => !!p);
    if (points.length === 0) return null;
    const x = points.reduce((s, p) => s + p[0], 0) / points.length;
    const y = points.reduce((s, p) => s + p[1], 0) / points.length;
    return [x, y];
  }

  headCenter(frame: SkeletonFrame): [number, number] {
    const nose = frame.pose['nose'];
    return [nose[0], nose[1] - HEAD_RADIUS * 0.4]; // nudge up -- nose sits low in the face, not at head center
  }

  /** Updates the orthographic camera's framing box each frame, matching the
   * old SVG viewBox behavior 1:1 (same centers/radii), just expressed as
   * an ortho camera box instead of a viewBox string. */
  private updateCameraFraming(): void {
    if (!this.camera || !this.canvasRef) return;
    const aspect = (this.canvasRef.nativeElement.clientWidth || 1) / (this.canvasRef.nativeElement.clientHeight || 1);

    let centerSvg: [number, number] | null = this.BODY_CENTER;
    let rY = this.BODY_RADIUS_Y;
    if (this.mode === 'hands') { centerSvg = this.handsCenter() ?? this.BODY_CENTER; rY = this.HAND_ZOOM_RADIUS; }
    if (this.mode === 'face') { centerSvg = this.faceZoomCenter() ?? this.BODY_CENTER; rY = this.FACE_ZOOM_RADIUS; }

    const rX = rY * aspect; // fills the container's real aspect exactly, same idea the old hands/face viewBox used (there, hardcoded at 1.6; here, read from the actual canvas)
    const centerX = centerSvg[0];
    const centerY = -centerSvg[1]; // flipped into scene space

    this.camera.left = -rX;
    this.camera.right = rX;
    this.camera.top = rY;
    this.camera.bottom = -rY;
    this.camera.position.set(centerX, centerY, 10);
    this.camera.lookAt(centerX, centerY, 0);
    this.camera.updateProjectionMatrix();
  }

  // ---- 3D geometry builders ----
  private capsuleMesh(a: [number, number], b: [number, number], rA: number, rB: number, material: THREE.Material): THREE.Group {
    const group = new THREE.Group();
    const va = this.toVec3(a);
    const vb = this.toVec3(b);
    const dir = new THREE.Vector3().subVectors(vb, va);
    const len = dir.length() || 0.001;
    const cyl = new THREE.Mesh(new THREE.CylinderGeometry(rB, rA, len, 10, 1), material);
    const mid = new THREE.Vector3().addVectors(va, vb).multiplyScalar(0.5);
    cyl.position.copy(mid);
    cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
    group.add(cyl);
    const joint = new THREE.Mesh(new THREE.SphereGeometry(rA, 12, 10), material);
    joint.position.copy(va);
    group.add(joint);
    return group;
  }

  private fingerCapsules(hand: [number, number][], material: THREE.Material): THREE.Group {
    const group = new THREE.Group();
    const BASE_WIDTH = 0.055;
    const TAPER = 0.18;
    for (const chain of Object.values(FINGER_CHAINS)) {
      for (let i = 0; i < chain.length - 1; i++) {
        const wA = BASE_WIDTH * (1 - i * TAPER);
        const wB = BASE_WIDTH * (1 - (i + 1) * TAPER);
        group.add(this.capsuleMesh(hand[chain[i]], hand[chain[i + 1]], wA, wB, material));
      }
    }
    return group;
  }

  private armCapsules(frame: SkeletonFrame, side: 'l' | 'r', material: THREE.Material): THREE.Group {
    const group = new THREE.Group();
    const shoulder = frame.pose[`${side}_shoulder`];
    const elbow = frame.pose[`${side}_elbow`];
    const wrist = frame.pose[`${side}_wrist`];
    group.add(this.capsuleMesh(shoulder, elbow, 0.14, 0.09, material));
    group.add(this.capsuleMesh(elbow, wrist, 0.09, 0.06, material));
    return group;
  }

  private palmMesh(hand: [number, number][], material: THREE.Material): THREE.Mesh {
    const pts = PALM_OUTLINE.map(i => new THREE.Vector2(hand[i][0], -hand[i][1]));
    const shape = new THREE.Shape(pts);
    return new THREE.Mesh(new THREE.ShapeGeometry(shape), material);
  }

  /** Same hips -> shoulders -> head-bump silhouette as before, now given
   * real thickness via extrusion + a bevel so it catches light as a solid
   * form instead of a flat tinted shape. */
  private bodySilhouetteMesh(frame: SkeletonFrame, material: THREE.Material): THREE.Mesh {
    const nose = frame.pose['nose'];
    const headTop: [number, number] = [nose[0], nose[1] - HEAD_RADIUS * 1.6];
    const headLeft: [number, number] = [nose[0] - HEAD_RADIUS, nose[1] - HEAD_RADIUS * 0.4];
    const headRight: [number, number] = [nose[0] + HEAD_RADIUS, nose[1] - HEAD_RADIUS * 0.4];
    const points: [number, number][] = [
      frame.pose['l_hip'], frame.pose['l_shoulder'], headLeft, headTop, headRight, frame.pose['r_shoulder'], frame.pose['r_hip'],
    ];
    const toScene = (p: [number, number]): THREE.Vector2 => new THREE.Vector2(p[0], -p[1]);
    const mid = (a: THREE.Vector2, b: THREE.Vector2) => new THREE.Vector2((a.x + b.x) / 2, (a.y + b.y) / 2);
    const scenePts = points.map(toScene);
    const shape = new THREE.Shape();
    const start = mid(scenePts[scenePts.length - 1], scenePts[0]);
    shape.moveTo(start.x, start.y);
    for (let i = 0; i < scenePts.length; i++) {
      const cur = scenePts[i];
      const next = scenePts[(i + 1) % scenePts.length];
      const m = mid(cur, next);
      shape.quadraticCurveTo(cur.x, cur.y, m.x, m.y);
    }
    const geo = new THREE.ExtrudeGeometry(shape, {
      depth: TORSO_DEPTH, bevelEnabled: true, bevelThickness: 0.05, bevelSize: 0.05, bevelSegments: 3, curveSegments: 12,
    });
    geo.translate(0, 0, -TORSO_DEPTH / 2); // center on z=0 so the front face sits at +TORSO_DEPTH/2
    return new THREE.Mesh(geo, material);
  }

  /** Face contours as thin raised tube-chains sitting just in front of the
   * torso's own front face -- real 3D features, not a flat decal. */
  private faceFeatureMesh(points: [number, number][], radius: number, material: THREE.Material): THREE.Group {
    const group = new THREE.Group();
    for (let i = 0; i < points.length; i++) {
      const a = points[i];
      const b = points[(i + 1) % points.length];
      group.add(this.capsuleMesh(a, b, radius, radius, material));
    }
    group.position.z = FACE_Z_OFFSET; // raised just in front of the torso's own front face, no z-fighting
    return group;
  }

  private orientationArrow(origin: [number, number], dir: [number, number], material: THREE.Material): THREE.Group {
    const tip: [number, number] = [origin[0] + dir[0] * 0.7, origin[1] + dir[1] * 0.7];
    return this.capsuleMesh(origin, tip, 0.045, 0.02, material);
  }

  private disposeGroup(group: THREE.Group): void {
    // Materials are shared across many meshes per frame (one per logical
    // part, not per mesh), so track and dispose each unique one once --
    // disposing per-mesh would double-dispose the shared ones.
    const disposedMaterials = new Set<THREE.Material>();
    group.traverse(obj => {
      if (obj instanceof THREE.Mesh) {
        obj.geometry.dispose();
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const m of mats) {
          if (!disposedMaterials.has(m)) { m.dispose(); disposedMaterials.add(m); }
        }
      }
    });
  }

  private material(colorHex: string): THREE.Material {
    return new THREE.MeshStandardMaterial({ color: new THREE.Color(colorHex), roughness: 0.55, metalness: 0.05 });
  }

  private renderFrame(): void {
    if (!this.renderer || !this.scene || !this.camera || !this.dynamicGroup) return;

    this.disposeGroup(this.dynamicGroup);
    this.scene.remove(this.dynamicGroup);
    this.dynamicGroup = new THREE.Group();
    this.scene.add(this.dynamicGroup);

    this.updateCameraFraming();

    const frame = this.frame;
    if (frame) {
      const bodyMat = this.material(this.bodyColor);
      const locationMat = this.material(this.locationColor);
      const handshapeMat = this.material(this.handshapeColor);
      const orientationMat = this.material(this.orientationColor);
      const eyebrowMat = this.material(this.eyebrowColor);

      if (this.mode === 'body') {
        const torso = this.bodySilhouetteMesh(frame, bodyMat);
        (torso.material as THREE.MeshStandardMaterial).transparent = true;
        (torso.material as THREE.MeshStandardMaterial).opacity = 0.9;
        this.dynamicGroup.add(torso);
        this.dynamicGroup.add(this.armCapsules(frame, 'l', locationMat));
        this.dynamicGroup.add(this.armCapsules(frame, 'r', locationMat));
      }

      if ((this.mode === 'body' || this.mode === 'face') && frame.face) {
        for (const key of FACE_CONTOUR_KEYS) {
          const pts = frame.face[key] as [number, number][] | undefined;
          if (pts && pts.length > 1) this.dynamicGroup.add(this.faceFeatureMesh(pts, 0.012, bodyMat));
        }
        if (frame.face.left_eyebrow?.length) this.dynamicGroup.add(this.faceFeatureMesh(frame.face.left_eyebrow, 0.016, eyebrowMat));
        if (frame.face.right_eyebrow?.length) this.dynamicGroup.add(this.faceFeatureMesh(frame.face.right_eyebrow, 0.016, eyebrowMat));
      }

      if (this.mode !== 'face') {
        if (frame.left_hand) {
          this.dynamicGroup.add(this.palmMesh(frame.left_hand, handshapeMat.clone()));
          this.dynamicGroup.add(this.fingerCapsules(frame.left_hand, handshapeMat));
        }
        if (frame.right_hand) {
          this.dynamicGroup.add(this.palmMesh(frame.right_hand, handshapeMat.clone()));
          this.dynamicGroup.add(this.fingerCapsules(frame.right_hand, handshapeMat));
        }
        if (frame.right_hand && frame.right_orientation) {
          this.dynamicGroup.add(this.orientationArrow(frame.right_hand[0], frame.right_orientation, orientationMat));
        }
        if (frame.left_hand && frame.left_orientation) {
          this.dynamicGroup.add(this.orientationArrow(frame.left_hand[0], frame.left_orientation, orientationMat));
        }
      }
    }

    this.renderer.render(this.scene, this.camera);
  }

  ngOnChanges(): void {
    if (this.timer) clearInterval(this.timer);
    this.progress = 0;
    this.updateFrame();
    const hasData = this.frames && this.frames.length > 0;
    if (hasData && this.animate) {
      this.timer = setInterval(() => {
        this.progress = (this.progress + 1 / 24) % 1; // ~24 steps per loop, independent of array length
        this.updateFrame();
      }, FRAME_MS / Math.max(this.speedMultiplier, 0.1));
    }
    if (this.viewInitialized) this.renderFrame();
  }

  private updateFrame(): void {
    this.frame = this.frameAt(this.frames);
    if (this.viewInitialized) this.renderFrame();
  }

  private frameAt(seq: SkeletonFrame[] | null | undefined): SkeletonFrame | null {
    if (!seq || seq.length === 0) return null;
    const idx = Math.min(seq.length - 1, Math.floor(this.progress * seq.length));
    return seq[idx];
  }

  ngOnDestroy(): void {
    if (this.timer) clearInterval(this.timer);
    this.resizeObserver?.disconnect();
    if (this.dynamicGroup) this.disposeGroup(this.dynamicGroup);
    this.renderer?.dispose();
  }
}
