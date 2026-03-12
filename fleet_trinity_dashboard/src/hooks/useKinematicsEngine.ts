import { useRef } from 'react';
import * as THREE from 'three';

// Represents the standard URDF Object implementation
export interface KinematicRobot extends THREE.Group {
  joints: Record<string, { setJointValue: (angle: number) => void }>;
}

export type EncounterCallback = (robotId: string, event: 'zone_entry' | 'door_passed', target: string, policy: string, action: string) => void;

const DOORS = [
  { id: 'Systems Lab', z: 4 },
  { id: 'Unit 02', z: 0 },
  { id: 'Server Terminal', z: -4 }
];

const getZone = (z: number) => {
  if (z >= 3.33) return 'Green Zone (Safe)';
  if (z >= -3.33) return 'Amber Zone (Caution)';
  return 'Red Zone (Danger)';
};

/**
 * The Construct-Inspired Kinematics Engine
 * Provides separated physics/actuator control logic for 
 * driving the Unitree G1 Humanoid and FastBot differentials.
 */
export function useKinematicsEngine(onEncounter?: EncounterCallback) {
  const accumulatedTime = useRef(0);
  const lastTime = useRef<number | null>(null);
  const lastState = useRef<Record<string, { zone: string; z: number }>>({});
  // Cooldown map to prevent duplicate encounter logs within 500ms
  const encounterCooldown = useRef<Record<string, number>>({});
  const COOLDOWN_MS = 500;

  const fireEncounter = (robotId: string, event: 'zone_entry' | 'door_passed', target: string, policy: string, action: string) => {
    const key = `${robotId}:${event}:${target}`;
    const now = Date.now();
    if (encounterCooldown.current[key] && now - encounterCooldown.current[key] < COOLDOWN_MS) {
      return; // Still within cooldown — skip duplicate
    }
    encounterCooldown.current[key] = now;
    onEncounter?.(robotId, event, target, policy, action);
  };

  const applyActuators = (
    robot: KinematicRobot | null, 
    isFastbot: boolean, 
    fsm: string,
    policy: string,
    robotId: string
  ) => {
    // Calculate global delta time for smooth interpolation
    const current = Date.now();
    if (lastTime.current === null) {
      lastTime.current = current;
    }
    const delta = current - lastTime.current;
    lastTime.current = current;
    
    const isPlaying = fsm !== 'Passive' && fsm !== 'Emergency';
    
    if (isPlaying) {
      accumulatedTime.current += delta;
    }

    // "Return to Dock" implies reversing back to base
    const dirMulti = policy === 'Return to Dock' ? -1 : 1;
    const time = accumulatedTime.current * 0.001 * dirMulti; // Slowed from 0.003 → 0.001
    const speed = isFastbot ? 1.2 : 0.7; // Slowed from 2.5/1.5 → 1.2/0.7
    
    // Corridor Translation (Z-Axis Forward Mobility)
    const zOffset = (time * speed) % 20; // 0 to 20
    const finalZ = dirMulti === 1 ? zOffset - 10 : 10 - zOffset;

    // Obstacle Evasion (CBF) weaving effect — gentler sway
    const xOffset = policy === 'Obstacle Evasion (CBF)' ? Math.sin(time * 0.8) * 1.2 : 0;

    const currentZone = getZone(finalZ);
    if (!lastState.current[robotId]) {
      lastState.current[robotId] = { zone: currentZone, z: finalZ };
    } else {
      const prev = lastState.current[robotId];
      if (prev.zone !== currentZone && isPlaying) {
        fireEncounter(robotId, 'zone_entry', currentZone, policy, `Entering ${currentZone}`);
      }
      for (const door of DOORS) {
        const crossedForward = prev.z < door.z && finalZ >= door.z;
        const crossedBackward = prev.z > door.z && finalZ <= door.z;
        const wrapped = Math.abs(finalZ - prev.z) > 10;
        if (!wrapped && (crossedForward || crossedBackward) && isPlaying) {
            const actionInfo = xOffset !== 0 ? 'evading obstacles via CBF constraints' : 'nominal translation';
            fireEncounter(robotId, 'door_passed', door.id, policy, actionInfo);
        }
      }
      prev.zone = currentZone;
      prev.z = finalZ;
    }

    if (robot && robot.joints && isPlaying) {
      if (isFastbot) {
        // FastBot Differential Drive
        const wheelSpeed = time * 5;
        if (robot.joints['l_wheel_joint']) robot.joints['l_wheel_joint'].setJointValue(wheelSpeed);
        if (robot.joints['r_wheel_joint']) robot.joints['r_wheel_joint'].setJointValue(wheelSpeed);
        
        // LiDAR Scanning Emulation
        if (robot.joints['lidar_joint']) robot.joints['lidar_joint'].setJointValue(time * -15);
      } else {
        // Unitree G1 Bipedal Walking Kinematics 
        // Syncs sine-wave patterns for upper and lower limbs
        const walkSpeed = time * 2;
        
        // Hips (Pitch)
        if (robot.joints['left_hip_pitch_joint']) robot.joints['left_hip_pitch_joint'].setJointValue(Math.sin(walkSpeed) * 0.4);
        if (robot.joints['right_hip_pitch_joint']) robot.joints['right_hip_pitch_joint'].setJointValue(-Math.sin(walkSpeed) * 0.4);
        
        // Knees (Pitch)
        if (robot.joints['left_knee_joint']) robot.joints['left_knee_joint'].setJointValue(Math.abs(Math.sin(walkSpeed + Math.PI/2)) * 0.6);
        if (robot.joints['right_knee_joint']) robot.joints['right_knee_joint'].setJointValue(Math.abs(Math.sin(walkSpeed - Math.PI/2)) * 0.6);
        
        // Ankles (Pitch compensation)
        if (robot.joints['left_ankle_pitch_joint']) robot.joints['left_ankle_pitch_joint'].setJointValue(-Math.sin(walkSpeed) * 0.2);
        if (robot.joints['right_ankle_pitch_joint']) robot.joints['right_ankle_pitch_joint'].setJointValue(Math.sin(walkSpeed) * 0.2);
        
        // Shoulders (Counter-swing pitch)
        if (robot.joints['left_shoulder_pitch_joint']) robot.joints['left_shoulder_pitch_joint'].setJointValue(-Math.sin(walkSpeed) * 0.3);
        if (robot.joints['right_shoulder_pitch_joint']) robot.joints['right_shoulder_pitch_joint'].setJointValue(Math.sin(walkSpeed) * 0.3);
        
        // Elbows (Resting state compensation)
        if (robot.joints['left_elbow_joint']) robot.joints['left_elbow_joint'].setJointValue(0.4 + Math.sin(walkSpeed) * 0.1);
        if (robot.joints['right_elbow_joint']) robot.joints['right_elbow_joint'].setJointValue(0.4 - Math.sin(walkSpeed) * 0.1);
      }
    }

    return { zOffset: finalZ, xOffset };
  };

  return { applyActuators };
}
