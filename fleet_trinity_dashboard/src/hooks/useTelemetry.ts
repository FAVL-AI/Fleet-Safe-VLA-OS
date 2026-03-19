import { useState, useEffect, useRef } from 'react';
import { animationStore } from './animationStore';

// Generates an SVG path string from an array of values (0-100)
function generateSVGPath(data: number[]) {
  if (data.length === 0) return '';
  const stepX = 100 / (data.length - 1);
  // Map data (0-100) to Y (0-40)
  const mapY = (val: number) => 40 - (val / 100) * 40;
  
  let path = `M0,${mapY(data[0])}`;
  
  for (let i = 0; i < data.length - 1; i++) {
    const x1 = i * stepX + stepX / 2;
    const y1 = mapY(data[i]);
    const x2 = i * stepX + stepX / 2;
    const y2 = mapY(data[i + 1]);
    const endX = (i + 1) * stepX;
    const endY = mapY(data[i + 1]);
    
    // Smooth bezier curve connecting points
    path += ` C${x1},${y1} ${x2},${y2} ${endX},${endY}`;
  }
  return path;
}

export function useTelemetry() {
  const [cbfInterventions, setCbfInterventions] = useState<number[]>(Array(10).fill(50));
  const [safetyViolations, setSafetyViolations] = useState<number[]>(Array(10).fill(10));
  const [adherence, setAdherence] = useState<number[]>(Array(10).fill(85));
  const [efficiency, setEfficiency] = useState<number[]>(Array(10).fill(90));
  const [robotY, setRobotY] = useState(0);
  const [recalibrating, setRecalibrating] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/fleet');
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'telemetry') {
          setCbfInterventions(prev => [...prev.slice(1), data.cbf]);
          setSafetyViolations(prev => [...prev.slice(1), data.violations]);
          setAdherence(prev => [...prev.slice(1), data.adherence]);
          setEfficiency(prev => [...prev.slice(1), data.efficiency]);
          
          setRobotY(data.robotY);
          animationStore.robotY = data.robotY;
          if (data.recalibrating !== undefined) setRecalibrating(data.recalibrating);
        }
      } catch {
        // silent handle
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  return {
    paths: {
      cbf: generateSVGPath(cbfInterventions.map(v => v * 0.4)), // Scale visual height
      violations: generateSVGPath(safetyViolations.map(v => v * 2)),
      adherence: generateSVGPath(adherence.map(v => v * 0.5)),
      efficiency: generateSVGPath(efficiency.map(v => v * 0.5))
    },
    metrics: {
      cbfCurrent: Math.round(cbfInterventions[cbfInterventions.length - 1]),
      violationsCurrent: Math.round(safetyViolations[safetyViolations.length - 1] / 10),
      adherenceCurrent: Math.round(adherence[adherence.length - 1]),
      efficiencyCurrent: Math.round(efficiency[efficiency.length - 1])
    },
    kinematics: {
      robotY,
      recalibrating
    }
  };
}
