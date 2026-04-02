'use client';

import React, { useRef, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Text } from '@react-three/drei';
import * as THREE from 'three';
import URDFLoader from 'urdf-loader';
import { useKinematicsEngine, KinematicRobot, EncounterCallback } from '../hooks/useKinematicsEngine';
import { animationStore } from '../hooks/animationStore';

/* ─── Spatial Environment Components ─── */

function FloorZone({ position, color, length, label }: { position: [number, number, number], color: string, length: number, label: string }) {
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={position} receiveShadow>
        <planeGeometry args={[8, length]} />
        <meshStandardMaterial color={color} roughness={0.8} transparent opacity={0.18} />
      </mesh>
      {/* Zone boundary lines */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, position[1] + 0.01, position[2] + length / 2]}>
        <planeGeometry args={[8, 0.06]} />
        <meshBasicMaterial color={color} transparent opacity={0.6} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, position[1] + 0.01, position[2] - length / 2]}>
        <planeGeometry args={[8, 0.06]} />
        <meshBasicMaterial color={color} transparent opacity={0.6} />
      </mesh>
      {/* Zone label on the floor */}
      <Text
        position={[3.5, position[1] + 0.02, position[2]]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.35}
        color={color}
        anchorX="center"
        anchorY="middle"
        fillOpacity={0.7}
      >
        {label}
      </Text>
    </group>
  );
}

function LabeledDoor({ position, label, isRight }: { position: [number, number, number], label: string, isRight?: boolean }) {
  const rotationY = isRight ? -Math.PI / 2 : Math.PI / 2;
  const labelX = isRight ? -0.11 : 0.11;
  return (
    <group position={position}>
      <mesh position={[0, 1, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.2, 2, 1.5]} />
        <meshStandardMaterial color="#1e293b" metalness={0.5} roughness={0.5} />
      </mesh>
      <Text 
        position={[labelX, 1.5, 0]} 
        rotation={[0, rotationY, 0]} 
        fontSize={0.18} 
        color="#22d3ee" 
        anchorX="center" 
        anchorY="middle"
        maxWidth={1.2}
      >
        {label}
      </Text>
    </group>
  );
}

/* ─── Coordinate Axis Grid ─── */
function SpatialGrid() {
  return (
    <group>
      {[-8, -6, -4, -2, 0, 2, 4, 6, 8].map((z) => (
        <group key={z}>
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[-3.8, -0.38, z]}>
            <planeGeometry args={[0.4, 0.04]} />
            <meshBasicMaterial color="#22d3ee" transparent opacity={0.3} />
          </mesh>
          <Text
            position={[-3.8, -0.37, z + 0.3]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={0.15}
            color="#22d3ee"
            anchorX="center"
            fillOpacity={0.4}
          >
            {`Z=${z}`}
          </Text>
        </group>
      ))}
    </group>
  );
}

/* ─── URDF Robot Loader ─── */

interface URDFProps {
  url: string;
  position?: [number, number, number];
  scale?: number;
  color?: number;
  isFastbot?: boolean;
  robotId: string;
  onEncounter?: EncounterCallback;
}

/**
 * Applies premium materials and CBF wireframe shells to all meshes in a robot.
 * Called AFTER all async mesh loads complete (critical for STL-based models like G1).
 */
function applyMaterials(urdfRobot: THREE.Object3D, color: number) {
  const armorMat = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.9, roughness: 0.1 });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x556677, metalness: 0.6, roughness: 0.5 });
  const cyanMat = new THREE.MeshStandardMaterial({ color: 0x06b6d4, metalness: 0.6, roughness: 0.2, emissive: 0x06b6d4, emissiveIntensity: 0.2 });

  // Collect meshes FIRST to avoid traverse-while-modifying
  const meshes: THREE.Mesh[] = [];
  urdfRobot.traverse((child: THREE.Object3D) => {
    if ((child as THREE.Mesh).isMesh) {
      meshes.push(child as THREE.Mesh);
    }
  });

  for (const mesh of meshes) {
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    
    const mat = mesh.material as THREE.Material;
    if (mat && mat.name) {
      if (['light_gray', 'white', 'yellow_beacon', 'chassis_color'].includes(mat.name)) {
        mesh.material = armorMat;
      } else if (['dark_gray', 'dark', 'dark_wheels', 'wheel_color', 'camera_color'].includes(mat.name)) {
        mesh.material = darkMat;
      } else if (['blue_accent', 'cyan_body', 'lidar_color', 'beacon_color'].includes(mat.name)) {
        mesh.material = cyanMat;
      } else if (mat.name === 'green_dir') {
        mesh.material = new THREE.MeshStandardMaterial({ color: 0x22c55e, metalness: 0.4, roughness: 0.6 });
      } else if (mat.name === 'screen_color') {
        mesh.material = new THREE.MeshStandardMaterial({ color: 0x0a0a12, metalness: 0.8, roughness: 0.2, emissive: 0x22d3ee, emissiveIntensity: 0.1 });
      } else {
        mesh.material = armorMat;
      }
    } else {
      // STL meshes come with default MeshPhongMaterial, no name
      mesh.material = armorMat;
    }
    
    // CBF holographic wireframe shell
    const wireframeShell = new THREE.Mesh(
      mesh.geometry, 
      new THREE.MeshBasicMaterial({
        color: color,
        wireframe: true,
        transparent: true,
        opacity: 0.12,
        depthWrite: false
      })
    );
    wireframeShell.scale.set(1.08, 1.08, 1.08);
    mesh.add(wireframeShell);
  }

  return meshes.length;
}

function URDFRobot({ url, position = [0, -0.5, 0], scale = 3.5, color = 0x06b6d4, isFastbot = false, robotId, onEncounter }: URDFProps) {
  const [robotState, setRobotState] = useState<KinematicRobot | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const robotRef = useRef<KinematicRobot | null>(null);
  const groupRef = useRef<THREE.Group>(null);

  useEffect(() => {
    let cancelled = false;

    console.log("URDFRobot loading:", url);
    try {
      // Custom LoadingManager to track when ALL resources (XML + STL meshes) finish
      const loadingManager = new THREE.LoadingManager();
      let urdfRobot: KinematicRobot | null = null;

      // This fires when ALL items (URDF XML + every STL mesh) have loaded
      loadingManager.onLoad = () => {
        if (cancelled || !urdfRobot) return;
        
        const meshCount = applyMaterials(urdfRobot, color);
        console.log(`All resources loaded [${url}]. Meshes styled: ${meshCount}`);
        
        urdfRobot.scale.set(scale, scale, scale);
        urdfRobot.position.set(...position);
        urdfRobot.rotation.set(-Math.PI / 2, 0, 0);
        
        robotRef.current = urdfRobot;
        setRobotState(urdfRobot);
      };

      loadingManager.onError = (itemUrl) => {
        console.error(`Failed to load resource: ${itemUrl}`);
      };

      const loader = new URDFLoader(loadingManager);
      loader.load(
        url, 
        (loadedRobot: unknown) => {
          if (cancelled) return;
          // XML parsed, mesh loading in progress — store ref, materials applied in onLoad
          urdfRobot = loadedRobot as KinematicRobot;
          console.log(`URDF XML parsed [${url}]. Waiting for meshes...`);
        }, 
        undefined, 
        (err: unknown) => {
          if (cancelled) return;
          console.error("URDF LOAD ERROR:", err);
          setErrorMsg((err as Error).message || String(err));
        }
      );
    } catch (err) {
      console.error("Failed to initialize urdf-loader:", err);
      setErrorMsg((err as Error).message || String(err));
    }

    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  const { applyActuators } = useKinematicsEngine(onEncounter);

  useFrame(() => {
    try {
      if (groupRef.current) {
        const robotData = animationStore.robots[robotId];
        const fsm = robotData?.fsm || 'Passive';
        const policy = robotData?.policy || '';
        const robotY = animationStore.robotY;

        const { zOffset, xOffset } = applyActuators(robotRef.current, isFastbot, fsm, policy, robotId);
        
        groupRef.current.position.z = zOffset;
        groupRef.current.position.x = position[0] + (xOffset || 0);
        groupRef.current.position.y = THREE.MathUtils.lerp(groupRef.current.position.y, (robotY - 0.95) * 2 + 0.1, 0.1);
      }
    } catch (e) {
      console.warn("WebGL Animation Error:", e);
    }
  });

  return (
    <group ref={groupRef}>
      {errorMsg ? (
        <Html center>
          <div className="bg-red-900/80 border border-red-500 p-4 text-white font-mono text-xs whitespace-pre-wrap max-w-sm backdrop-blur-md shadow-2xl">
            <span className="font-bold text-red-300 block mb-2">URDF PARSE ERROR</span>
            {errorMsg}
          </div>
        </Html>
      ) : null}
      {robotState ? <primitive object={robotState} /> : null}
    </group>
  );
}

/* ─── WebGL Error Boundary ─── */
class WebGLErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  componentDidCatch(error: Error) {
    console.warn('WebGL error caught by boundary:', error.message);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="w-full h-full flex items-center justify-center bg-slate-950/90">
          <div className="text-center space-y-3 p-8">
            <div className="text-4xl">🤖</div>
            <h3 className="text-slate-300 font-bold tracking-widest text-sm">WEBGL UNAVAILABLE</h3>
            <p className="text-slate-500 text-xs max-w-sm">3D viewer requires WebGL support. The fleet is still operational — telemetry and minimap remain active.</p>
            <div className="flex items-center justify-center gap-2 mt-4">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-emerald-400 text-xs font-mono">FLEET ONLINE</span>
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

/* ─── Main Viewer Component ─── */

const RobotViewer = React.memo(function RobotViewer({ onEncounter }: { onEncounter?: EncounterCallback }) {
  return (
    <div className="w-full h-full relative">
      <WebGLErrorBoundary>
        <Canvas
          shadows
          gl={{ alpha: true, antialias: true }}
          camera={{ position: [0, 1.2, 4], fov: 40 }}
          onCreated={({ gl }) => {
            const canvas = gl.domElement;
            canvas.addEventListener('webglcontextlost', (e) => {
              e.preventDefault();
              console.warn('WebGL context lost');
            });
          }}
        >
          {/* Removed solid fog so background image is visible clearly */}
          <ambientLight intensity={0.6} />
          
          <directionalLight position={[2, 5, 3]} intensity={1.5} color="#ffffff" castShadow />
          <spotLight position={[-3, 4, -2]} intensity={2.0} color="#22d3ee" angle={0.4} penumbra={1} />
          <spotLight position={[3, -1, 3]} intensity={1.0} color="#06b6d4" angle={0.6} penumbra={1} />

          {/* Fleet Roster */}
          <URDFRobot url="/robots/unitree_g1/g1_29dof.urdf" position={[-0.8, 0.5, 0]} scale={1.5} color={0x06b6d4} robotId="robot_0" onEncounter={onEncounter} />
          <URDFRobot url="/robots/fastbot/fastbot.urdf" position={[1.0, -0.4, 0]} scale={1.2} color={0x22c55e} isFastbot={true} robotId="robot_1" onEncounter={onEncounter} />
          
          {/* Spatial Memory Zones with labels */}
          <FloorZone position={[0, -0.39, 6.67]} color="#22c55e" length={6.66} label="GREEN ZONE (SAFE)" />
          <FloorZone position={[0, -0.39, 0]} color="#f59e0b" length={6.66} label="AMBER ZONE (CAUTION)" />
          <FloorZone position={[0, -0.39, -6.67]} color="#ef4444" length={6.66} label="RED ZONE (DANGER)" />

          {/* Coordinate grid for spatial reference */}
          <SpatialGrid />

          {/* Labeled Doors */}
          <LabeledDoor position={[-3, -0.4, 4]} label={"Systems Lab\n[Safe Zone]"} />
          <LabeledDoor position={[3, -0.4, 0]} label={"Unit 02\n[Caution]"} isRight />
          <LabeledDoor position={[-3, -0.4, -4]} label={"Server Terminal\n[Danger]"} />

          {/* Transparent ground plane just to catch shadows */}
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.4, 0]} receiveShadow>
            <planeGeometry args={[15, 25]} />
            <shadowMaterial opacity={0.4} />
          </mesh>

          <OrbitControls 
            enableZoom={true} 
            enablePan={true} 
            maxPolarAngle={Math.PI / 2 + 0.1} 
            minPolarAngle={0.2} 
            makeDefault 
            target={[0, 0.8, 0]}
            maxDistance={12}
            minDistance={2}
          />
        </Canvas>
      </WebGLErrorBoundary>
    </div>
  );
});

export default RobotViewer;

