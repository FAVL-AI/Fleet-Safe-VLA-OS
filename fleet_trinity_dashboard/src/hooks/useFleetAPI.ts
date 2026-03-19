import { useState, useEffect, useCallback, useRef } from 'react';
import { animationStore } from './animationStore';

// Mirroring the Python API definitions
export interface RobotState {
  fsm: string;
  policy: string;
  position: [number, number, number];
  arm: string | null;
}

export interface PipelineStatus {
  record: string;
  hdf5: string;
  lerobot: string;
  train: string;
  deploy: string;
}

export interface FleetState {
  robots: Record<string, RobotState>;
  pipeline?: PipelineStatus;
  wsConnected: boolean;
}

const API_BASE_URL = 'http://localhost:8000/api';
const WS_BASE_URL = 'ws://localhost:8000/ws';

export function useFleetAPI() {
  const [fleetState, setFleetState] = useState<FleetState>({
    robots: {
      'robot_0': { fsm: 'Passive', policy: 'HospitalPatrol', position: [0,0,0], arm: null },
      'robot_1': { fsm: 'Passive', policy: 'HospitalPatrol', position: [0,0,0], arm: null }
    },
    wsConnected: false
  });

  const wsRef = useRef<WebSocket | null>(null);

  // Initialize WebSocket Connection — max 3 retries then give up
  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimeout: NodeJS.Timeout;
    let retries = 0;
    const MAX_RETRIES = 3;

    const connectWS = () => {
      if (retries >= MAX_RETRIES) {
        console.warn(`[Fleet API] Backend unreachable after ${MAX_RETRIES} attempts — running in standalone mode`);
        return;
      }
      try {
        ws = new WebSocket(`${WS_BASE_URL}/fleet`);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('[Fleet API] WebSocket Connected');
          retries = 0; // Reset on successful connect
          setFleetState(prev => ({ ...prev, wsConnected: true }));
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'init') {
              setFleetState(prev => ({ 
                ...prev, 
                robots: data.robots || prev.robots,
                pipeline: data.pipeline
              }));
            } else if (data.type === 'fsm') {
              setFleetState(prev => ({
                ...prev,
                robots: {
                  ...prev.robots,
                  [data.robot_id]: { ...prev.robots[data.robot_id], fsm: data.new }
                }
              }));
            } else if (data.type === 'policy') {
              setFleetState(prev => ({
                ...prev,
                robots: {
                  ...prev.robots,
                  [data.robot_id]: { ...prev.robots[data.robot_id], policy: data.policy }
                }
              }));
            }
          } catch (e) {
            console.error('[Fleet API] Error parsing WS message:', e);
          }
        };

        ws.onclose = () => {
          setFleetState(prev => ({ ...prev, wsConnected: false }));
          retries++;
          const delay = Math.min(3000 * Math.pow(2, retries), 15000);
          console.warn(`[Fleet API] WebSocket disconnected. Retry ${retries}/${MAX_RETRIES} in ${delay/1000}s`);
          reconnectTimeout = setTimeout(connectWS, delay);
        };
        
        ws.onerror = () => {
          // Silently handled — onclose will fire next
        };
      } catch {
        console.warn('[Fleet API] Failed to connect WebSocket — running standalone');
      }
    };

    connectWS();

    return () => {
      clearTimeout(reconnectTimeout);
      if (ws) ws.close();
    };
  }, []);

  // Sync robot states into the animation store for the 3D viewer
  useEffect(() => {
    animationStore.robots = fleetState.robots;
  }, [fleetState.robots]);

  // REST API Mutations — apply locally as fallback when backend is unreachable
  const setFSMState = useCallback(async (robotId: string, state: string) => {
    // Always apply locally so UI responds immediately
    setFleetState(prev => ({
      ...prev,
      robots: {
        ...prev.robots,
        [robotId]: { ...prev.robots[robotId], fsm: state }
      }
    }));

    try {
      const res = await fetch(`${API_BASE_URL}/fleet/${robotId}/fsm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state })
      });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return await res.json();
    } catch {
      console.warn(`[Fleet API] Backend unavailable — FSM state applied locally for ${robotId}:`, state);
      return null;
    }
  }, []);

  const setPolicy = useCallback(async (robotId: string, policy: string) => {
    // Always apply locally so UI responds immediately
    setFleetState(prev => ({
      ...prev,
      robots: {
        ...prev.robots,
        [robotId]: { ...prev.robots[robotId], policy }
      }
    }));

    try {
      const res = await fetch(`${API_BASE_URL}/fleet/${robotId}/policy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ policy })
      });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return await res.json();
    } catch {
      console.warn(`[Fleet API] Backend unavailable — policy applied locally for ${robotId}:`, policy);
      return null;
    }
  }, []);

  return {
    ...fleetState,
    setFSMState,
    setPolicy
  };
}
