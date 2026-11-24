import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';

function getRiskColor(risk) {
  if (risk < 0.5) return '#22c55e';
  if (risk <= 0.5) return '#eab308';
  return '#ef4444';
}

function HumanBody({ riskProbability }) {
  const materialRef = useRef();

  const risk = Math.min(1, Math.max(0, Number(riskProbability) || 0));
  const fillLevel = -1.65 + risk * 3.3;
  const riskColor = getRiskColor(risk);

  const material = useMemo(() => {
    const mat = new THREE.MeshStandardMaterial({
      color: '#a8c2d8',
      roughness: 0.35,
      metalness: 0.15,
    });

    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uFillLevel = { value: -1.65 };
      shader.uniforms.uRiskColor = {
        value: new THREE.Color('#22c55e'),
      };

      shader.vertexShader = shader.vertexShader.replace(
        '#include <common>',
        `#include <common>
         varying float vHumanWorldY;`
      );

      shader.vertexShader = shader.vertexShader.replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
         vec4 humanWorldPosition = modelMatrix * vec4(position, 1.0);
         vHumanWorldY = humanWorldPosition.y;`
      );

      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <common>',
        `#include <common>
         varying float vHumanWorldY;
         uniform float uFillLevel;
         uniform vec3 uRiskColor;`
      );

      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <color_fragment>',
        `#include <color_fragment>
         float fillMask = smoothstep(
           uFillLevel - 0.025,
           uFillLevel + 0.025,
           vHumanWorldY
         );
         float fillEdge = 1.0 - smoothstep(
           0.0,
           0.055,
           abs(vHumanWorldY - uFillLevel)
         );
         diffuseColor.rgb = mix(
           uRiskColor,
           vec3(0.48, 0.62, 0.75),
           fillMask
         );
         diffuseColor.rgb += fillEdge * 0.45;`
      );

      mat.userData.shader = shader;
    };

    mat.customProgramCacheKey = () => 'human-risk-fill-v1';

    return mat;
  }, []);

  useFrame(() => {
    if (material.userData.shader) {
      material.userData.shader.uniforms.uFillLevel.value = fillLevel;
      material.userData.shader.uniforms.uRiskColor.value.set(riskColor);
    }
  });

  const shared = {
    material,
    castShadow: true,
    receiveShadow: true,
  };

  return (
    <group ref={materialRef} position={[0, 0, 0]}>
      {/* Head */}
      <mesh {...shared} position={[0, 1.48, 0]}>
        <sphereGeometry args={[0.22, 32, 32]} />
      </mesh>

      {/* Neck */}
      <mesh {...shared} position={[0, 1.18, 0]}>
        <cylinderGeometry args={[0.09, 0.11, 0.25, 24]} />
      </mesh>

      {/* Chest and torso */}
      <mesh {...shared} position={[0, 0.68, 0]}>
        <capsuleGeometry args={[0.27, 0.65, 8, 24]} />
      </mesh>

      {/* Pelvis */}
      <mesh {...shared} position={[0, 0.05, 0]}>
        <sphereGeometry args={[0.29, 32, 24]} />
      </mesh>

      {/* Left and right arms */}
      {[-1, 1].map((side) => (
        <React.Fragment key={side}>
          <mesh
            {...shared}
            position={[side * 0.39, 0.72, 0]}
            rotation={[0, 0, -side * 0.12]}
          >
            <capsuleGeometry args={[0.105, 0.55, 8, 20]} />
          </mesh>

          <mesh
            {...shared}
            position={[side * 0.48, 0.12, 0]}
          >
            <capsuleGeometry args={[0.085, 0.52, 8, 20]} />
          </mesh>

          <mesh {...shared} position={[side * 0.48, -0.22, 0]}>
            <sphereGeometry args={[0.09, 20, 16]} />
          </mesh>
        </React.Fragment>
      ))}

      {/* Left and right legs */}
      {[-1, 1].map((side) => (
        <React.Fragment key={side}>
          <mesh
            {...shared}
            position={[side * 0.15, -0.57, 0]}
          >
            <capsuleGeometry args={[0.145, 0.66, 8, 24]} />
          </mesh>

          <mesh
            {...shared}
            position={[side * 0.15, -1.25, 0]}
          >
            <capsuleGeometry args={[0.105, 0.52, 8, 24]} />
          </mesh>

          <mesh
            {...shared}
            position={[side * 0.15, -1.62, 0.07]}
          >
            <sphereGeometry args={[0.13, 24, 16]} />
          </mesh>
        </React.Fragment>
      ))}
    </group>
  );
}

export default function HumanRiskModel3D({
  riskProbability = 0,
  modelName = 'BiLSTM',
}) {
  const risk = Math.min(1, Math.max(0, Number(riskProbability) || 0));
  const percentage = (risk * 100).toFixed(1);
  const color = getRiskColor(risk);

  const riskLabel =
    risk < 0.5 ? 'Lower risk' :
    risk <= 0.5 ? 'Moderate risk' :
    'Higher risk';

  return (
    <div className="glass-panel rounded-xl p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">
            3D Risk Visualization
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            {modelName} model output
          </p>
        </div>

        <div className="text-right">
          <div
            className="text-2xl font-bold"
            style={{ color }}
          >
            {percentage}%
          </div>
          <div className="text-xs" style={{ color }}>
            {riskLabel}
          </div>
        </div>
      </div>

      <div className="h-[380px] w-full mt-2">
        <Canvas
          camera={{ position: [0, 0.15, 5.2], fov: 38 }}
          dpr={[1, 1.5]}
          gl={{ antialias: true, alpha: true }}
        >
          <ambientLight intensity={0.8} />
          <directionalLight
            position={[3, 4, 5]}
            intensity={2}
            castShadow
          />
          <pointLight
            position={[-3, 1, 2]}
            intensity={1.2}
            color={color}
          />

          <HumanBody riskProbability={risk} />

          <ContactShadows
            position={[0, -1.8, 0]}
            opacity={0.3}
            scale={3}
            blur={2}
          />

          <OrbitControls
            enablePan={false}
            minDistance={3.5}
            maxDistance={7}
            minPolarAngle={Math.PI / 3}
            maxPolarAngle={Math.PI / 1.7}
          />
        </Canvas>
      </div>

      <div className="flex justify-center gap-5 text-xs text-slate-400">
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

      <p className="text-[10px] text-slate-500 text-center mt-3">
        Illustrative visualization of model probability. Not an anatomical
        representation or clinical assessment.
      </p>
    </div>
  );
}


