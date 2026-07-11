/**
 * Three.js scene: camera, renderer, orbit controls, and the globe's two
 * visual layers (a translucent base sphere for depth cues, and the DGGS
 * cell layer carrying the data layer's colors). Cells are real polygons
 * built from `grid_geometry`'s per-cell boundary vertices (`cell-geometry.ts`),
 * triangulated as a fan around each cell's centroid and colored by a real
 * field from `query_field` (`field-layer.ts`). If a world's geometry ever
 * comes back without boundary data, this falls back to the previous
 * billboarded-disc point cloud so the globe still renders something.
 */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { viridis } from "./palette";
import type { FieldLayer } from "./field-layer";
import type { CellPoint } from "./cell-geometry";

const GLOBE_RADIUS = 1;
const BACKGROUND_COLOR = 0x05070d;
/** Coarse worlds (e.g. 122 cells at resolution 0) get bigger discs so the
 *  sphere isn't mostly empty; fine ones (tens of thousands of cells) get
 *  smaller ones so they don't overlap into a solid blob. */
function pointSizeFor(cellCount: number): number {
  if (cellCount <= 0) return 0.03;
  return Math.min(0.09, Math.max(0.006, 0.9 / Math.sqrt(cellCount)));
}

export class GlobeScene {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private cellPoints: THREE.Points | null = null;
  private cellPolygons: THREE.Mesh | null = null;
  private readonly discTexture: THREE.Texture;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.discTexture = makeDiscTexture();
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.scene.background = new THREE.Color(BACKGROUND_COLOR);

    this.camera = new THREE.PerspectiveCamera(
      50,
      canvas.clientWidth / canvas.clientHeight,
      0.1,
      100,
    );
    this.camera.position.set(0, 0, 3);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 1.3;
    this.controls.maxDistance = 8;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    this.scene.add(this.baseSphere());
    this.scene.add(this.graticule());

    window.addEventListener("resize", () => this.handleResize());
    this.handleResize();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  /** A dim, translucent sphere giving the point cloud a solid body to sit on. */
  private baseSphere(): THREE.Mesh {
    const geometry = new THREE.SphereGeometry(GLOBE_RADIUS * 0.995, 48, 32);
    const material = new THREE.MeshBasicMaterial({
      color: 0x0a1626,
      transparent: true,
      opacity: 0.85,
    });
    return new THREE.Mesh(geometry, material);
  }

  /** Plain lat/lon reference lines (standard cartography, not DGGS-specific). */
  private graticule(): THREE.LineSegments {
    const positions: number[] = [];
    const steps = 64;
    for (let latDeg = -60; latDeg <= 60; latDeg += 30) {
      for (let i = 0; i < steps; i += 1) {
        positions.push(
          ...latLonToXyz(latDeg, (360 * i) / steps),
          ...latLonToXyz(latDeg, (360 * (i + 1)) / steps),
        );
      }
    }
    for (let lonDeg = 0; lonDeg < 360; lonDeg += 30) {
      for (let i = 0; i < steps; i += 1) {
        positions.push(
          ...latLonToXyz(-90 + (180 * i) / steps, lonDeg),
          ...latLonToXyz(-90 + (180 * (i + 1)) / steps, lonDeg),
        );
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({ color: 0x1c2c44, transparent: true, opacity: 0.6 });
    return new THREE.LineSegments(geometry, material);
  }

  /**
   * (Re)build the cell layer from real cell geometry, coloring each cell by
   * `layer` via Viridis. Renders true boundary polygons when every point
   * carries at least a triangle's worth of boundary vertices; otherwise
   * falls back to the centroid point cloud.
   */
  setCells(points: CellPoint[], layer: FieldLayer): void {
    this.clearCells();
    if (points.length > 0 && points.every((point) => point.boundary.length >= 3)) {
      this.cellPolygons = buildCellPolygons(points, layer);
      this.scene.add(this.cellPolygons);
    } else {
      this.cellPoints = buildCellPointCloud(points, layer, this.discTexture);
      this.scene.add(this.cellPoints);
    }
  }

  private clearCells(): void {
    if (this.cellPoints) {
      this.scene.remove(this.cellPoints);
      this.cellPoints.geometry.dispose();
      (this.cellPoints.material as THREE.Material).dispose();
      this.cellPoints = null;
    }
    if (this.cellPolygons) {
      this.scene.remove(this.cellPolygons);
      this.cellPolygons.geometry.dispose();
      (this.cellPolygons.material as THREE.Material).dispose();
      this.cellPolygons = null;
    }
  }

  private handleResize(): void {
    const { clientWidth, clientHeight } = this.canvas;
    this.camera.aspect = clientWidth / clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(clientWidth, clientHeight, false);
  }

  private tick(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

/**
 * Build one filled polygon per cell from its ordered boundary vertices,
 * triangulated as a fan from the centroid to each consecutive pair of
 * boundary vertices (valid because DGGS cell boundaries are convex around
 * their centroid). All cells are merged into a single indexed
 * `THREE.BufferGeometry` so the whole layer is one draw call regardless of
 * cell count. `DoubleSide` sidesteps having to reconcile each cell's
 * boundary winding order with its outward sphere normal.
 */
function buildCellPolygons(points: CellPoint[], layer: FieldLayer): THREE.Mesh {
  const positions: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const span = layer.max - layer.min || 1;

  points.forEach((point, i) => {
    const t = (layer.values[i]! - layer.min) / span;
    const { r, g, b } = viridis(t);
    const centroidIndex = positions.length / 3;

    positions.push(point.x * GLOBE_RADIUS, point.y * GLOBE_RADIUS, point.z * GLOBE_RADIUS);
    colors.push(r, g, b);
    for (const vertex of point.boundary) {
      positions.push(vertex.x * GLOBE_RADIUS, vertex.y * GLOBE_RADIUS, vertex.z * GLOBE_RADIUS);
      colors.push(r, g, b);
    }

    const vertexCount = point.boundary.length;
    for (let k = 0; k < vertexCount; k += 1) {
      indices.push(centroidIndex, centroidIndex + 1 + k, centroidIndex + 1 + ((k + 1) % vertexCount));
    }
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  const material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  return new THREE.Mesh(geometry, material);
}

/** The pre-#62 fallback: one billboarded disc per cell centroid, for worlds whose geometry lacks boundaries. */
function buildCellPointCloud(points: CellPoint[], layer: FieldLayer, discTexture: THREE.Texture): THREE.Points {
  const positions = new Float32Array(points.length * 3);
  const colors = new Float32Array(points.length * 3);
  const span = layer.max - layer.min || 1;
  points.forEach((point, i) => {
    positions[i * 3] = point.x * GLOBE_RADIUS;
    positions[i * 3 + 1] = point.y * GLOBE_RADIUS;
    positions[i * 3 + 2] = point.z * GLOBE_RADIUS;
    const t = (layer.values[i]! - layer.min) / span;
    const { r, g, b } = viridis(t);
    colors[i * 3] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({
    size: pointSizeFor(points.length),
    vertexColors: true,
    map: discTexture,
    alphaTest: 0.5,
    transparent: true,
  });
  return new THREE.Points(geometry, material);
}

/**
 * A small radial-gradient canvas texture so each cell renders as a soft
 * billboarded disc rather than a hard square point sprite in the fallback
 * point cloud above.
 */
function makeDiscTexture(): THREE.Texture {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.8, "rgba(255,255,255,1)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function latLonToXyz(latDeg: number, lonDeg: number): [number, number, number] {
  const lat = (latDeg * Math.PI) / 180;
  const lon = (lonDeg * Math.PI) / 180;
  const r = GLOBE_RADIUS * 1.001;
  return [r * Math.cos(lat) * Math.cos(lon), r * Math.sin(lat), r * Math.cos(lat) * Math.sin(lon)];
}
