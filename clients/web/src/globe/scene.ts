/**
 * Three.js scene: camera, renderer, orbit controls, and the globe's two
 * visual layers (a translucent base sphere for depth cues, and the DGGS
 * cell-point cloud carrying the data layer's colors). See
 * `placeholder-sphere.ts` and `field-layer.ts` for why the cell geometry
 * and field values are currently a stand-in rather than server data.
 */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { viridis } from "./palette";
import type { FieldLayer } from "./field-layer";
import type { SpherePoint } from "./placeholder-sphere";

const GLOBE_RADIUS = 1;
const POINT_SIZE = 0.02;
const BACKGROUND_COLOR = 0x05070d;

export class GlobeScene {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private cellPoints: THREE.Points | null = null;

  constructor(private readonly canvas: HTMLCanvasElement) {
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

  /** (Re)build the cell-point cloud, coloring each point by `layer` via Viridis. */
  setCells(points: SpherePoint[], layer: FieldLayer): void {
    if (this.cellPoints) {
      this.scene.remove(this.cellPoints);
      this.cellPoints.geometry.dispose();
      (this.cellPoints.material as THREE.Material).dispose();
    }

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
    const material = new THREE.PointsMaterial({ size: POINT_SIZE, vertexColors: true });
    this.cellPoints = new THREE.Points(geometry, material);
    this.scene.add(this.cellPoints);
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

function latLonToXyz(latDeg: number, lonDeg: number): [number, number, number] {
  const lat = (latDeg * Math.PI) / 180;
  const lon = (lonDeg * Math.PI) / 180;
  const r = GLOBE_RADIUS * 1.001;
  return [r * Math.cos(lat) * Math.cos(lon), r * Math.sin(lat), r * Math.cos(lat) * Math.sin(lon)];
}
