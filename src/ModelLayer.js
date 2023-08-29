import mapboxgl from "mapbox-gl";
import {
  Scene,
  Camera,
  AmbientLight,
  WebGLRenderer,
  HemisphereLight,
  DirectionalLight,
  Matrix4,
  Vector3,
} from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";

class ModelLayer {
  constructor(options) {
    this.modelUrl = options.url;
    this.modelOrigin = options.origin;
    this.modelAltitude = options.altitude;
    this.onLoad = options.onLoad;
    this.id = options.id;
    this.type = "custom";
    this.renderingMode = "3d";
    const rotate = [Math.PI / 2, options.rotateY, 0];
    const scale = options.scale * 1e-8;
    const mercator = mapboxgl.MercatorCoordinate.fromLngLat(
      this.modelOrigin,
      this.modelAltitude
    );

    this.modelTransform = {
      translateX: mercator.x,
      translateY: mercator.y,
      translateZ: mercator.z,
      rotateX: rotate[0],
      rotateY: rotate[1],
      rotateZ: rotate[2],
      scale,
    };
  }

  onAdd(map, gl) {
    this.scaleY = 0;

    this.map = map;
    this.camera = new Camera();
    this.scene = new Scene();

    // const ambientLight = new AmbientLight(0xffffff, 1);
    // this.scene.add(ambientLight);
    const hemisphereLight = new HemisphereLight(0xffffff, 0x080820, 0.5);
    this.scene.add(hemisphereLight);

    const loader = new GLTFLoader();
    loader.load(this.modelUrl, (gltf) => {
      this.modelHeight = gltf.scene.scale.z;
      gltf.scene.scale.y = 0;
      this.model = gltf.scene;
      gltf.scene.traverse((child) => {
        console.log(child);
        if (child.isMesh) {
          //child.material.color.set(0xff0000);
          //child.material.needsUpdate = true;
        }
      });

      this.scene.add(gltf.scene);
      if (this.onLoad) this.onLoad();
    });

    this.renderer = new WebGLRenderer({
      canvas: map.getCanvas(),
      context: gl,
      antialias: true,
    });

    this.renderer.autoClear = false;
    this.renderer.gammaOutput = true;
    this.renderer.gammaFactor = 2;
  }

  render(gl, matrix) {
    if (this.model && this.scaleY < this.modelHeight) {
      this.scaleY += this.modelHeight / 8; // Adjust this value to control the animation speed
      this.model.scale.y = this.scaleY;
    }
    const modelTransform = this.modelTransform;
    const rotationX = new Matrix4().makeRotationAxis(
      new Vector3(1, 0, 0),
      modelTransform.rotateX
    );
    const rotationY = new Matrix4().makeRotationAxis(
      new Vector3(0, 1, 0),
      modelTransform.rotateY
    );
    const rotationZ = new Matrix4().makeRotationAxis(
      new Vector3(0, 0, 1),
      modelTransform.rotateZ
    );

    const m = new Matrix4().fromArray(matrix);
    const l = new Matrix4()
      .makeTranslation(
        modelTransform.translateX,
        modelTransform.translateY,
        modelTransform.translateZ
      )
      .scale(
        new Vector3(
          modelTransform.scale,
          -modelTransform.scale,
          modelTransform.scale
        )
      )
      .multiply(rotationX)
      .multiply(rotationY)
      .multiply(rotationZ);

    this.camera.projectionMatrix.elements = matrix;
    this.camera.projectionMatrix = m.multiply(l);
    this.renderer.state.reset();
    this.renderer.render(this.scene, this.camera);
    this.map.triggerRepaint();
  }
}

export default ModelLayer;
