import React, { Suspense, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, ContactShadows, useGLTF } from '@react-three/drei';
import * as THREE from 'three';

function getRiskColor(risk) {
  if (risk < 0.5) return '#22c55e';
  if (risk === 0.5) return '#eab308';
  return '#ef4444';
}

function HumanMesh({ riskProbability }) {
  const { scene } = useGLTF('/models/human-base.glb');
  const materialsRef = useRef([]);

  const { model, minY, maxY } = useMemo(() => {
    materialsRef.current = [];

    const cloned = scene.clone(true);

    // Calculate the original model bounds.
    cloned.updateMatrixWorld(true);

    const originalBounds = new THREE.Box3().setFromObject(cloned);
    const originalSize = originalBounds.getSize(new THREE.Vector3());
    const originalCenter = originalBounds.getCenter(new THREE.Vector3());

    // Normalize the model to a known height and center.
    const targetHeight = 3.4;
    const scale = targetHeight / Math.max(originalSize.y, 0.001);

    cloned.scale.setScalar(scale);
    cloned.position.set(
      -originalCenter.x * scale,
      -originalCenter.y * scale,
      -originalCenter.z * scale
    );

    cloned.updateMatrixWorld(true);

    // Calculate the actual normalized world-space boundaries.
    const normalizedBounds = new THREE.Box3().setFromObject(cloned);
    const normalizedMinY = normalizedBounds.min.y;
    const normalizedMaxY = normalizedBounds.max.y;

    cloned.traverse((child) => {
      if (!child.isMesh) return;

      child.geometry = child.geometry.clone();

      const originalMaterials = Array.isArray(child.material)
        ? child.material
        : [child.material];

      const newMaterials = originalMaterials.map((original) => {
        const mat = original.clone();

        mat.color = new THREE.Color('#b9d9ed');
        mat.metalness = 0.15;
        mat.roughness = 0.42;
        mat.transparent = true;
        mat.opacity = 0.9;
        mat.depthWrite = false;

        mat.onBeforeCompile = (shader) => {
          shader.uniforms.uFillLevel = { value: normalizedMinY };
          shader.uniforms.uMinY = { value: normalizedMinY };
          shader.uniforms.uMaxY = { value: normalizedMaxY };
          shader.uniforms.uRiskColor = {
            value: new THREE.Color('#22c55e'),
          };

          shader.vertexShader = shader.vertexShader.replace(
            '#include <common>',
            '#include <common>\nvarying float vHumanWorldY;'
          );

          shader.vertexShader = shader.vertexShader.replace(
            '#include <begin_vertex>',
            `#include <begin_vertex>
             vec4 humanWorldPosition = modelMatrix * vec4(transformed, 1.0);
             vHumanWorldY = humanWorldPosition.y;`
          );

          shader.fragmentShader = shader.fragmentShader.replace(
            '#include <common>',
            `#include <common>
             varying float vHumanWorldY;
             uniform float uFillLevel;
             uniform float uMinY;
             uniform float uMaxY;
             uniform vec3 uRiskColor;`
          );

          shader.fragmentShader = shader.fragmentShader.replace(
            '#include <color_fragment>',
            `#include <color_fragment>

             float edgeWidth = max((uMaxY - uMinY) * 0.008, 0.002);

             float fillMask = 1.0 - smoothstep(
               uFillLevel - edgeWidth,
               uFillLevel + edgeWidth,
               vHumanWorldY
             );

             float fillEdge = 1.0 - smoothstep(
               0.0,
               edgeWidth * 3.0,
               abs(vHumanWorldY - uFillLevel)
             );

             vec3 hologramColor = vec3(0.40, 0.68, 0.86);

             diffuseColor.rgb = mix(
               hologramColor,
               uRiskColor,
               fillMask
             );

             diffuseColor.rgb += fillEdge * 0.25;`
          );

          mat.userData.shader = shader;
        };

        mat.customProgramCacheKey = () => 'clintwin-human-risk-v2';

        materialsRef.current.push(mat);
        return mat;
      });

      child.material = Array.isArray(child.material)
        ? newMaterials
        : newMaterials[0];
    });

    return {
      model: cloned,
      minY: normalizedMinY,
      maxY: normalizedMaxY,
    };
  }, [scene]);

  const risk = THREE.MathUtils.clamp(
    Number(riskProbability) || 0,
    0,
    1
  );

  const fillLevel = minY + risk * (maxY - minY);

  // Use displayed precision for consistent percentage and color.
  const displayedRisk = Number((risk * 100).toFixed(1)) / 100;
  const color = getRiskColor(displayedRisk);

  useFrame(() => {
    materialsRef.current.forEach((mat) => {
      const shader = mat.userData.shader;
      if (!shader) return;

      shader.uniforms.uFillLevel.value = fillLevel;
      shader.uniforms.uRiskColor.value.set(color);
    });
  });

  return <primitive object={model} />;
}

function Scene({ riskProbability }) {
  const displayedRisk =
    Number((Number(riskProbability) * 100).toFixed(1)) / 100;

  const color = getRiskColor(displayedRisk);

  return (
    <>
      <ambientLight intensity={1.1} />

      <directionalLight
        position={[3, 5, 5]}
        intensity={2.2}
      />

      <directionalLight
        position={[-4, 2, -3]}
        intensity={1.3}
        color="#60a5fa"
      />

      <pointLight
        position={[0, 0, 2]}
        intensity={1.2}
        color={color}
      />

      <Suspense fallback={null}>
        <HumanMesh riskProbability={riskProbability} />
      </Suspense>

      <ContactShadows
        position={[0, -1.75, 0]}
        opacity={0.25}
        scale={3}
        blur={2.5}
      />

      <OrbitControls
        enablePan={false}
        minDistance={3}
        maxDistance={7}
        minPolarAngle={Math.PI / 3}
        maxPolarAngle={Math.PI / 1.6}
        target={[0, 0, 0]}
      />
    </>
  );
}

export default function HumanRiskModel3D({
  riskProbability = 0,
  modelName = 'BiLSTM',
}) {
  const risk = THREE.MathUtils.clamp(
    Number(riskProbability) || 0,
    0,
    1
  );

  const percentage = (risk * 100).toFixed(1);
  const displayedRisk = Number(percentage) / 100;
  const color = getRiskColor(displayedRisk);

  const riskLabel =
    displayedRisk < 0.5
      ? 'Lower risk'
      : displayedRisk === 0.5
        ? 'Moderate risk'
        : 'Higher risk';

  return (
    <div className="rounded-xl bg-slate-950/40 border border-slate-800/70 p-3">
      <div className="flex items-center justify-between px-2 pt-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">
            Human Digital Twin
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            {modelName} prediction
          </p>
        </div>

        <div className="text-right">
          <div className="text-3xl font-bold" style={{ color }}>
            {percentage}%
          </div>
          <div className="text-xs mt-1" style={{ color }}>
            {riskLabel}
          </div>
        </div>
      </div>

      <div className="h-[440px] w-full">
        <Canvas
          camera={{ position: [0, 0, 5], fov: 35 }}
          dpr={[1, 1.5]}
          gl={{ antialias: true, alpha: true }}
        >
          <Scene riskProbability={risk} />
        </Canvas>
      </div>

      <div className="flex justify-center gap-5 text-xs text-slate-400 pb-2">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-green-500" />
          Below 50%
        </span>

        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-yellow-500" />
          50%
        </span>

        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-red-500" />
          Above 50%
        </span>
      </div>

      <p className="text-center text-[10px] text-slate-500 mt-2">
        Illustrative model-risk visualization · Not a clinical assessment
      </p>
    </div>
  );
}

useGLTF.preload('/models/human-base.glb');
