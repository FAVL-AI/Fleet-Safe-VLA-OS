'use client';

import React, { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { Mail, Bell, AlertTriangle, FileText, List, Clock, Settings, User, Info, Terminal, Play, Pause, Maximize, Minimize, PanelLeft, PanelRight } from 'lucide-react';
import { useTelemetry } from '@/hooks/useTelemetry';
import { useFleetAPI } from '@/hooks/useFleetAPI';

const RobotViewer = dynamic(() => import('@/components/RobotViewer'), { ssr: false });

/* ─── Live Animated Minimap with Zone-Aware Speed ─── */
const PATROL_WAYPOINTS_A = [
  { x: 40, y: 350, label: 'Dock' },
  { x: 100, y: 350, label: '' },
  { x: 100, y: 280, label: '' },
  { x: 140, y: 280, label: '' },
  { x: 140, y: 150, label: 'Unit 02' },
  { x: 100, y: 150, label: '' },
  { x: 100, y: 60, label: '' },
  { x: 160, y: 60, label: 'Pharmacy' },
  { x: 160, y: 150, label: '' },
  { x: 160, y: 220, label: 'Unit 05' },
];

const PATROL_WAYPOINTS_B = [
  { x: 160, y: 60, label: 'Pharmacy' },
  { x: 100, y: 60, label: '' },
  { x: 100, y: 150, label: '' },
  { x: 40, y: 150, label: 'Ward A' },
  { x: 40, y: 220, label: '' },
  { x: 100, y: 220, label: '' },
  { x: 100, y: 300, label: '' },
  { x: 40, y: 300, label: 'Ward B' },
  { x: 40, y: 350, label: 'Dock' },
  { x: 100, y: 350, label: '' },
];

// Slow Zones — ICU/wards/pharmacy require cautious approach
const SLOW_ZONES = [
  { cx: 40, cy: 125, r: 35, name: 'Unit 02 ICU', speed: 0.3 },
  { cx: 160, cy: 225, r: 35, name: 'Unit 05 ICU', speed: 0.3 },
  { cx: 40, cy: 150, r: 30, name: 'Ward A', speed: 0.35 },
  { cx: 40, cy: 300, r: 30, name: 'Ward B', speed: 0.35 },
  { cx: 160, cy: 60, r: 30, name: 'Pharmacy', speed: 0.4 },
];

function lerp2D(waypoints: typeof PATROL_WAYPOINTS_A, t: number) {
  const total = waypoints.length;
  const idx = t * (total - 1);
  const i = Math.floor(idx);
  const frac = idx - i;
  const a = waypoints[Math.min(i, total - 1)];
  const b = waypoints[Math.min(i + 1, total - 1)];
  return { x: a.x + (b.x - a.x) * frac, y: a.y + (b.y - a.y) * frac };
}

function getSpeedMultiplier(x: number, y: number): { speed: number; zone: string | null } {
  for (const z of SLOW_ZONES) {
    const dist = Math.sqrt((x - z.cx) ** 2 + (y - z.cy) ** 2);
    if (dist < z.r) return { speed: z.speed, zone: z.name };
  }
  return { speed: 1.0, zone: null };
}

interface SpeedLogEntry { time: string; robot: string; event: string; zone: string; speed: string }

const LiveMinimap = () => {
  // Non-uniform time accumulators for zone-aware speed
  const progressA = useRef(0);
  const progressB = useRef(0);
  const lastTimestamp = useRef(Date.now());
  const [posA, setPosA] = useState({ x: 40, y: 350 });
  const [posB, setPosB] = useState({ x: 160, y: 60 });
  const [speedA, setSpeedA] = useState({ speed: 1.0, zone: null as string | null });
  const [speedB, setSpeedB] = useState({ speed: 1.0, zone: null as string | null });
  const [speedLog, setSpeedLog] = useState<SpeedLogEntry[]>([]);
  const prevZoneA = useRef<string | null>(null);
  const prevZoneB = useRef<string | null>(null);
  const rafRef = useRef(0);

  useEffect(() => {
    const BASE_SPEED_A = 1 / 24; // full cycle fraction per second
    const BASE_SPEED_B = 1 / 30;

    const tick = () => {
      const now = Date.now();
      const dt = (now - lastTimestamp.current) * 0.001;
      lastTimestamp.current = now;

      // Get current speed multipliers
      const curPosA = lerp2D(PATROL_WAYPOINTS_A, progressA.current % 1);
      const curPosB = lerp2D(PATROL_WAYPOINTS_B, progressB.current % 1);
      const sA = getSpeedMultiplier(curPosA.x, curPosA.y);
      const sB = getSpeedMultiplier(curPosB.x, curPosB.y);

      // Advance with speed modulation
      progressA.current += dt * BASE_SPEED_A * sA.speed;
      progressB.current += dt * BASE_SPEED_B * sB.speed;

      // Keep in 0-1 range (loop)
      if (progressA.current >= 1) progressA.current -= 1;
      if (progressB.current >= 1) progressB.current -= 1;

      const newPosA = lerp2D(PATROL_WAYPOINTS_A, progressA.current);
      const newPosB = lerp2D(PATROL_WAYPOINTS_B, progressB.current);
      setPosA(newPosA);
      setPosB(newPosB);
      setSpeedA(sA);
      setSpeedB(sB);

      // Log zone transitions
      const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
      if (sA.zone !== prevZoneA.current) {
        if (sA.zone) {
          setSpeedLog(prev => [{ time: ts, robot: 'G1', event: 'SLOW', zone: sA.zone!, speed: `${Math.round(sA.speed * 100)}%` }, ...prev].slice(0, 8));
        } else if (prevZoneA.current) {
          setSpeedLog(prev => [{ time: ts, robot: 'G1', event: 'RESUME', zone: 'corridor', speed: '100%' }, ...prev].slice(0, 8));
        }
        prevZoneA.current = sA.zone;
      }
      if (sB.zone !== prevZoneB.current) {
        if (sB.zone) {
          setSpeedLog(prev => [{ time: ts, robot: 'FB', event: 'SLOW', zone: sB.zone!, speed: `${Math.round(sB.speed * 100)}%` }, ...prev].slice(0, 8));
        } else if (prevZoneB.current) {
          setSpeedLog(prev => [{ time: ts, robot: 'FB', event: 'RESUME', zone: 'corridor', speed: '100%' }, ...prev].slice(0, 8));
        }
        prevZoneB.current = sB.zone;
      }

      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  const pathA = PATROL_WAYPOINTS_A.map((wp, i) => `${i === 0 ? 'M' : 'L'}${wp.x},${wp.y}`).join(' ');
  const pathB = PATROL_WAYPOINTS_B.map((wp, i) => `${i === 0 ? 'M' : 'L'}${wp.x},${wp.y}`).join(' ');

  return (
    <div className="flex flex-col h-full">
      <svg viewBox="0 0 200 400" className="w-full flex-1 text-gray-500 font-mono text-[8px]" fill="none" style={{ minHeight: 0 }}>
        <defs>
          <radialGradient id="glow-cyan">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="glow-green">
            <stop offset="0%" stopColor="#22c55e" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#22c55e" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Floor Plan Outlines */}
        <path d="M20,20 L180,20 L180,380 L20,380 Z" stroke="#334155" strokeWidth="2" />
        <path d="M20,100 L80,100 M120,100 L180,100" stroke="#334155" strokeWidth="2" />
        <path d="M20,200 L80,200 M120,200 L180,200" stroke="#334155" strokeWidth="2" />
        <path d="M20,300 L80,300 M120,300 L180,300" stroke="#334155" strokeWidth="2" />
        <path d="M100,20 L100,60 M100,120 L100,180 M100,220 L100,280 M100,320 L100,380" stroke="#334155" strokeWidth="2" />

        {/* Slow Zone overlays — amber caution halos */}
        {SLOW_ZONES.map((z, i) => (
          <g key={`zone-${i}`}>
            <circle cx={z.cx} cy={z.cy} r={z.r} fill="#f59e0b" fillOpacity="0.06" stroke="#f59e0b" strokeWidth="0.8" strokeDasharray="3 3" opacity="0.5" />
            <text x={z.cx + z.r - 4} y={z.cy - z.r + 8} fill="#f59e0b" fontSize="5" opacity="0.6">⚠</text>
          </g>
        ))}

        {/* Rooms */}
        <rect x="25" y="110" width="30" height="30" fill="#22d3ee" fillOpacity="0.08" stroke="#22d3ee" strokeWidth="0.8" />
        <text x="30" y="160" fill="#22d3ee" fontSize="7">Unit 02</text>
        <rect x="145" y="210" width="30" height="30" fill="#22d3ee" fillOpacity="0.08" stroke="#22d3ee" strokeWidth="0.8" />
        <text x="145" y="260" fill="#22d3ee" fontSize="7">Unit 05</text>
        <text x="25" y="365" fill="#22c55e" fontSize="7">Dock</text>
        <text x="148" y="55" fill="#a855f7" fontSize="7">Pharmacy</text>
        <text x="25" y="148" fill="#f59e0b" fontSize="7">Ward A</text>
        <text x="25" y="310" fill="#f59e0b" fontSize="7">Ward B</text>

        {/* Patrol Routes */}
        <path d={pathA} stroke="#22d3ee" strokeWidth="1.5" strokeDasharray="4 4" opacity="0.4" />
        <path d={pathB} stroke="#22c55e" strokeWidth="1.5" strokeDasharray="4 4" opacity="0.4" />

        {/* Robot A — G1 */}
        <circle cx={posA.x} cy={posA.y} r="14" fill="url(#glow-cyan)" />
        <circle cx={posA.x} cy={posA.y} r="5" fill={speedA.zone ? '#f59e0b' : '#22d3ee'} stroke={speedA.zone ? '#92400e' : '#0e7490'} strokeWidth="1.2">
          <animate attributeName="r" values="4.5;5.5;4.5" dur={speedA.zone ? '2.5s' : '1.2s'} repeatCount="indefinite" />
        </circle>
        <text x={posA.x + 8} y={posA.y + 2} fill={speedA.zone ? '#f59e0b' : '#22d3ee'} fontSize="6" fontWeight="bold">
          G1{speedA.zone ? ' ⚠' : ''}
        </text>
        <text x={posA.x + 8} y={posA.y + 9} fill={speedA.zone ? '#f59e0b' : '#22d3ee'} fontSize="5" opacity="0.8">
          {(0.22 * speedA.speed).toFixed(2)} m/s
        </text>

        {/* Robot B — FastBot */}
        <circle cx={posB.x} cy={posB.y} r="14" fill="url(#glow-green)" />
        <rect x={posB.x - 4} y={posB.y - 4} width="8" height="8" rx="2" fill={speedB.zone ? '#f59e0b' : '#22c55e'} stroke={speedB.zone ? '#92400e' : '#166534'} strokeWidth="1.2" />
        <text x={posB.x + 8} y={posB.y + 2} fill={speedB.zone ? '#f59e0b' : '#22c55e'} fontSize="6" fontWeight="bold">
          FB{speedB.zone ? ' ⚠' : ''}
        </text>
        <text x={posB.x + 8} y={posB.y + 9} fill={speedB.zone ? '#f59e0b' : '#22c55e'} fontSize="5" opacity="0.8">
          {(0.18 * speedB.speed).toFixed(2)} m/s
        </text>

        {/* Waypoint dots */}
        {PATROL_WAYPOINTS_A.filter(wp => wp.label).map((wp, i) => (
          <circle key={`a-${i}`} cx={wp.x} cy={wp.y} r="2" fill="#334155" stroke="#22d3ee" strokeWidth="0.5" opacity="0.6" />
        ))}
      </svg>

      {/* Speed Event Log */}
      {speedLog.length > 0 && (
        <div className="mt-2 border-t border-slate-800 pt-1 overflow-hidden" style={{ maxHeight: '80px' }}>
          <span className="text-[9px] text-slate-500 tracking-widest font-bold block mb-1">SPEED LOG</span>
          <div className="space-y-0.5 overflow-y-auto" style={{ maxHeight: '60px' }}>
            {speedLog.map((entry, i) => (
              <div key={i} className="flex items-center gap-1 text-[9px] font-mono">
                <span className="text-slate-600">{entry.time}</span>
                <span className={entry.robot === 'G1' ? 'text-cyan-400' : 'text-green-400'}>{entry.robot}</span>
                <span className="text-slate-600">→</span>
                <span className={entry.event === 'SLOW' ? 'text-amber-400' : 'text-emerald-400'}>{entry.event}</span>
                <span className="text-slate-500">[{entry.zone}]</span>
                <span className={entry.event === 'SLOW' ? 'text-amber-500' : 'text-emerald-500'}>{entry.speed}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const LineChart = ({ color, points }: { color: 'cyan' | 'purple', points: string }) => (
  <svg viewBox="0 0 100 40" className="w-full h-24 mt-2 overflow-visible" preserveAspectRatio="none">
    <path 
      d={points} 
      fill="none" 
      className={color === 'cyan' ? 'graph-path-cyan' : 'graph-path-purple'} 
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    {/* Gradient Fill under line */}
    <path 
      d={`${points} L100,40 L0,40 Z`} 
      fill={`url(#grad-${color})`} 
      opacity="0.2"
    />
    <defs>
      <linearGradient id={`grad-cyan`} x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#22d3ee" />
        <stop offset="100%" stopColor="transparent" />
      </linearGradient>
      <linearGradient id={`grad-purple`} x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#a855f7" />
        <stop offset="100%" stopColor="transparent" />
      </linearGradient>
    </defs>
    
    {/* Axis Lines */}
    <line x1="0" y1="0" x2="0" y2="40" stroke="#334155" strokeWidth="0.5" />
    <line x1="0" y1="40" x2="100" y2="40" stroke="#334155" strokeWidth="0.5" />
    
    {/* Ticks text */}
    <text x="0" y="48" fill="#64748b" fontSize="4" fontFamily="monospace">12:00</text>
    <text x="30" y="48" fill="#64748b" fontSize="4" fontFamily="monospace">03:00</text>
    <text x="60" y="48" fill="#64748b" fontSize="4" fontFamily="monospace">18:00</text>
    <text x="85" y="48" fill="#64748b" fontSize="4" fontFamily="monospace">24:10</text>
  </svg>
);

export default function Home() {
  const telemetry = useTelemetry();
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [liveTime, setLiveTime] = useState('');
  const [bottomPanelHeight, setBottomPanelHeight] = useState(56);
  const isDraggingPanel = useRef(false);
  const dragStartY = useRef(0);
  const dragStartHeight = useRef(56);

  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setLiveTime(now.toLocaleTimeString('en-GB', { hour12: false }));
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  
  const { robots, setFSMState, setPolicy } = useFleetAPI();
  const activeRobot = robots['robot_0'] || { fsm: 'Passive', policy: 'HospitalPatrol' };
  const isAutonomousPlay = activeRobot.fsm !== 'Passive';
  const selectedPolicy = activeRobot.policy;

  const handleToggleExecution = () => {
    const newState = isAutonomousPlay ? 'Passive' : 'Patrol';
    setFSMState('robot_0', newState);
    setFSMState('robot_1', newState);
  };

  const handlePolicyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newPolicy = e.target.value;
    setPolicy('robot_0', newPolicy);
    setPolicy('robot_1', newPolicy);
  };

  const handleEncounter = React.useCallback(async (robotId: string, event: string, target: string, policy: string, action: string) => {
    try {
      await fetch(`http://localhost:8000/api/fleet/${robotId}/log`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ event, target, policy, action, timestamp: Date.now() })
      });
    } catch {
      // Backend unavailable — silently skip logging in standalone mode
    }
  }, []);

  return (
    <main className="min-h-screen w-full relative flex flex-col pt-0 text-[13px]">
      <div className="bg-aurora" />

      {/* Top Header */}
      <header className="glass-header w-full h-16 flex items-center justify-between px-6 z-50">
        <div className="flex items-center gap-4">
          <div className="grid grid-cols-2 gap-0.5 w-6 h-6">
            <div className="bg-[#22d3ee] rounded-sm"></div>
            <div className="bg-[#22d3ee] rounded-sm"></div>
            <div className="bg-[#22d3ee] rounded-sm"></div>
            <div className="bg-[#22d3ee] rounded-sm"></div>
          </div>
          <div className="flex gap-2 items-baseline">
            <h1 className="text-xl font-bold tracking-widest text-[#f8fafc]">F.L.E.E.T. COMMAND</h1>
            <span className="text-gray-500 font-light hidden md:inline">| UNITREE G1 OPERATIONS</span>
          </div>
        </div>

        <div className="flex items-center gap-5">
          <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center hover:bg-slate-700 transition cursor-pointer">
            <Mail size={16} className="text-slate-300" />
          </div>
          <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center relative hover:bg-slate-700 transition cursor-pointer">
            <Bell size={16} className="text-slate-300" />
            <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full flex items-center justify-center text-[10px] font-bold text-white">1</div>
          </div>
          <div className="flex items-center gap-3 ml-2 border-l border-slate-700 pl-4 cursor-pointer">
            <div className="w-8 h-8 rounded-full bg-slate-700 overflow-hidden">
              <User className="w-full h-full p-1 text-slate-400" />
            </div>
            <div className="flex flex-col text-xs">
              <span className="font-semibold text-white">User Profile</span>
              <span className="text-slate-400">Jan. 12, 2026</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Grid Layout */}
      <div className="flex-1 w-full p-6 flex flex-col xl:flex-row gap-6 h-[calc(100vh-4rem)] overflow-hidden">
        
        {/* LEFT COLUMN - Health & Graphs */}
        <div 
          className={`group/left flex flex-col gap-4 h-full shrink-0 transition-all duration-300 ease-in-out relative z-30 ${
            leftPanelOpen ? "w-full xl:w-[320px]" : "w-8 xl:w-8 hover:w-full hover:xl:w-[320px] cursor-pointer"
          }`}
          onClick={() => { if (!leftPanelOpen) setLeftPanelOpen(true); }}
        >
          {/* Collapsed Indicator */}
          <div className={`absolute inset-y-0 left-0 w-8 glass-panel flex flex-col items-center justify-center border-r border-[#22d3ee]/30 transition-opacity duration-300 ${leftPanelOpen ? 'opacity-0 pointer-events-none' : 'opacity-100 group-hover/left:opacity-0'}`}>
             <div className="text-[#22d3ee]/80 transform -rotate-90 whitespace-nowrap tracking-[0.3em] text-xs font-bold">
                SYSTEM HEALTH
             </div>
          </div>

          <div className={`w-full xl:w-[320px] flex flex-col gap-4 h-full transition-opacity duration-300 ${leftPanelOpen ? 'opacity-100' : 'opacity-0 group-hover/left:opacity-100 pointer-events-none group-hover/left:pointer-events-auto'}`}>
            {/* System Health */}
          <div className="glass-panel p-4 flex flex-col justify-center relative">
             <div className="flex justify-between items-end mb-2 pr-6">
               <span className="font-bold text-sm tracking-wide text-slate-200">SYSTEM HEALTH</span>
               <span className="text-[#22d3ee] font-mono font-bold text-sm">(99.2%)</span>
             </div>
             <PanelLeft size={16} className="absolute top-4 right-4 text-slate-500 cursor-pointer hover:text-white transition" onClick={(e) => { e.stopPropagation(); setLeftPanelOpen(false); }} />
             
             <div className="h-1 w-full bg-slate-800 rounded-full overflow-hidden mt-1">
               <div className="h-full w-[99.2%] bg-[#22d3ee] shadow-[0_0_10px_#22d3ee]"></div>
             </div>
          </div>

          <div className="glass-panel p-5 flex flex-col flex-1 gap-2 overflow-y-auto shrink-0">
             
             {/* CBF Panel */}
             <div className="flex flex-col mb-4">
                <div className="flex justify-between items-start border-b border-slate-800 pb-2 mb-2">
                  <div>
                    <h3 className="font-bold text-slate-200 text-sm">CONTROL BARRIER FUNCTION</h3>
                    <p className="text-xs text-slate-500">INTERVENTIONS (24h)</p>
                  </div>
                </div>
                <div className="flex justify-between items-baseline mb-2">
                  <span className="text-xs text-slate-400 tracking-wide">INTERVENTIONS (24h)</span>
                  <span className="text-[#22d3ee] font-mono text-lg font-bold">{telemetry.metrics.cbfCurrent + 200}</span>
                </div>
                <LineChart color="cyan" points={telemetry.paths.cbf} />
             </div>

             {/* Safety Violations */}
             <div className="flex flex-col mb-4">
                <div className="flex justify-between items-baseline mb-2">
                  <span className="text-xs text-slate-400 font-bold">SAFETY VIOLATIONS DETECTED</span>
                  <span className="text-[#22d3ee] font-mono text-lg font-bold">{telemetry.metrics.violationsCurrent}</span>
                </div>
                <LineChart color="cyan" points={telemetry.paths.violations} />
             </div>

             {/* Constraint Adherence */}
             <div className="flex flex-col mb-4">
                <div className="flex justify-between items-baseline mb-2">
                  <span className="text-xs text-slate-400 font-bold">CONSTRAINT ADHERENCE</span>
                  <span className="text-[#22d3ee] font-mono text-lg font-bold">{telemetry.metrics.adherenceCurrent}%</span>
                </div>
                <LineChart color="cyan" points={telemetry.paths.adherence} />
             </div>
             
             {/* Path Efficiency */}
             <div className="flex flex-col">
                <div className="flex justify-between items-baseline mb-2">
                  <span className="text-xs text-slate-400 font-bold">ROBOT PATH EFFICIENCY</span>
                  <span className="text-[#22d3ee] font-mono text-lg font-bold">{telemetry.metrics.efficiencyCurrent}%</span>
                </div>
                 <LineChart color="cyan" points={telemetry.paths.efficiency} />
              </div>
           </div>
          </div>
        </div>

        {/* CENTER COLUMN - Digital Twin Viewer */}
        <div className="flex-1 flex flex-col gap-4 h-full min-w-0">
          
          <div className="flex gap-4">
            <div className="glass-panel px-4 py-3 flex-1 flex flex-col justify-center">
              <div className="flex justify-between items-end mb-1">
                <span className="font-bold text-sm tracking-wide text-slate-200">ACTIVE ROBOTS</span>
                <span className="text-slate-400 font-mono text-xs">(14/15)</span>
              </div>
              <div className="h-1 w-full bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full w-[93%] bg-[#22d3ee]"></div>
              </div>
            </div>
            
            <div className="glass-panel px-4 py-3 flex items-center justify-center border border-red-500/30 bg-red-500/10 cursor-pointer">
              <AlertTriangle size={14} className="text-red-400 mr-2" />
              <span className="text-red-300 text-xs font-bold">Alert: Unit 02 Obstruction</span>
            </div>
            
            <div className="glass-panel px-4 py-3 flex items-center justify-center cursor-pointer hover:bg-slate-800/50 transition">
              <Info size={14} className="text-slate-400 mr-2" />
              <span className="text-slate-300 text-xs">Notifications</span>
            </div>
          </div>

          {/* Main 3D Panel */}
          <div 
            className="glass-panel w-full relative flex flex-col overflow-hidden min-h-[600px] h-full flex-1"
            style={{
              backgroundImage: 'linear-gradient(to bottom, rgba(15, 17, 26, 0.4) 0%, rgba(15, 17, 26, 0.9) 100%), url(/background_hospital.png)',
              backgroundSize: 'cover',
              backgroundPosition: 'center'
            }}
          >
            {/* Panel Expansion Controls atop the center column */}
            <div className="absolute top-4 right-4 z-20 flex gap-2">
              <button onClick={() => setLeftPanelOpen(!leftPanelOpen)} className="glass-panel p-2 hover:bg-slate-800 transition rounded shadow-lg backdrop-blur-md">
                 <PanelLeft size={16} className={leftPanelOpen ? "text-[#22d3ee]" : "text-slate-500"} />
              </button>
              <button onClick={() => setRightPanelOpen(!rightPanelOpen)} className="glass-panel p-2 hover:bg-slate-800 transition rounded shadow-lg backdrop-blur-md">
                 <PanelRight size={16} className={rightPanelOpen ? "text-[#22d3ee]" : "text-slate-500"} />
              </button>
              <button 
                onClick={() => {
                  const expand = leftPanelOpen || rightPanelOpen;
                  setLeftPanelOpen(!expand);
                  setRightPanelOpen(!expand);
                }} 
                className="glass-panel p-2 hover:bg-slate-800 transition rounded shadow-lg backdrop-blur-md"
              >
                 {leftPanelOpen || rightPanelOpen ? <Maximize size={16} className="text-slate-400" /> : <Minimize size={16} className="text-[#22d3ee]" />}
              </button>
            </div>

            <div className="absolute top-0 left-0 w-full p-4 flex justify-between items-start z-10 pointer-events-none">
              <h2 className="text-white font-bold tracking-widest text-sm drop-shadow-md">
                SAFETY CONSTRAINT <span className="text-slate-500 text-xs font-mono">(CBF ACTIVE)</span>
              </h2>
              <div className="glass-panel px-3 py-1.5 flex items-center gap-2 pointer-events-auto cursor-pointer hover:bg-slate-800/80">
                <FileText size={14} className="text-slate-400" />
                <span className="text-slate-300 text-xs font-mono">Unitree G1 log</span>
              </div>
            </div>

            <div className="absolute left-6 top-1/2 -translate-y-1/2 rotated-text z-10 opacity-60">
              <span className="text-2xl font-bold tracking-[0.2em] text-white">UNITREE G1 - UNIT 07 - ACTIVE</span>
            </div>

            <div className="absolute top-[25%] left-[25%] z-10 pointer-events-none">
               <div className="glass-panel px-2 py-1 flex items-center justify-center opacity-80 border-[#22d3ee]/40">
                 <span className="text-[#22d3ee] text-[10px] font-mono font-bold">SAFETY<br/>CONSTRAINT —<br/>(CBF ACTIVE)</span>
               </div>
               <div className="w-16 h-px bg-[#22d3ee]/80 mt-2 ml-4"></div>
            </div>

            <div className="absolute inset-0 w-full h-full z-0 overflow-hidden">
              <RobotViewer onEncounter={handleEncounter} />
            </div>

            {/* Autonomy Control Overlay */}
            <div className="absolute bottom-24 left-1/2 -translate-x-1/2 z-20 glass-panel p-2 flex items-center gap-4 bg-slate-900/80 backdrop-blur-md border border-[#22d3ee]/30 rounded-lg shadow-2xl">
              <div className="flex items-center gap-3 border-r border-slate-700 pr-5">
                 <Terminal size={14} className="text-[#22d3ee]" />
                 <span className="text-[10px] font-bold tracking-widest text-slate-400">ACTIVE POLICY</span>
                 <select 
                   value={selectedPolicy} 
                   onChange={handlePolicyChange}
                   className="bg-slate-950 text-[#22d3ee] text-xs font-mono p-1.5 rounded border border-slate-700 outline-none w-[200px]"
                 >
                   <option value="Corridor B Patrol">Corridor B Patrol</option>
                   <option value="Obstacle Evasion (CBF)">Obstacle Evasion (CBF)</option>
                   <option value="Return to Dock">Return to Dock</option>
                   <option value="Identify Target Person">Identify Target Person</option>
                 </select>
              </div>
              <button 
                onClick={handleToggleExecution}
                className={`flex items-center gap-2 px-5 py-2 rounded transition font-bold tracking-widest uppercase text-xs shadow-lg ${isAutonomousPlay ? 'bg-red-500/20 text-red-400 border border-red-500/50 hover:bg-red-500/30' : 'bg-[#22d3ee]/20 text-[#22d3ee] border border-[#22d3ee]/50 hover:bg-[#22d3ee]/30'}`}
              >
                {isAutonomousPlay ? <Pause size={14} /> : <Play size={14} />}
                {isAutonomousPlay ? 'HALT EXECUTION' : 'EXECUTE AUTONOMY'}
              </button>
            </div>

            <div 
              className="absolute bottom-0 left-0 w-full border-t border-slate-800/50 bg-slate-900/80 backdrop-blur-md z-10 select-none"
              style={{ height: `${bottomPanelHeight}px` }}
            >
              {/* Drag Handle */}
              <div 
                className="absolute top-0 left-0 w-full h-3 cursor-ns-resize group flex items-center justify-center hover:bg-[#22d3ee]/10 transition-colors"
                onMouseDown={(e) => {
                  e.preventDefault();
                  isDraggingPanel.current = true;
                  dragStartY.current = e.clientY;
                  dragStartHeight.current = bottomPanelHeight;
                  const onMove = (ev: MouseEvent) => {
                    if (!isDraggingPanel.current) return;
                    const delta = dragStartY.current - ev.clientY;
                    const newH = Math.max(56, Math.min(400, dragStartHeight.current + delta));
                    setBottomPanelHeight(newH);
                  };
                  const onUp = () => {
                    isDraggingPanel.current = false;
                    document.removeEventListener('mousemove', onMove);
                    document.removeEventListener('mouseup', onUp);
                  };
                  document.addEventListener('mousemove', onMove);
                  document.addEventListener('mouseup', onUp);
                }}
              >
                <div className="w-10 h-1 rounded-full bg-slate-600 group-hover:bg-[#22d3ee]/60 transition-colors" />
              </div>

              <div className="px-4 py-2 pt-4 h-full overflow-y-auto">
                {/* Row 1 — Primary Stats (compact) */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="flex items-center gap-2">
                    <span className="text-[#22d3ee] font-bold text-sm">ACTIVE</span>
                    <span className="text-slate-500 text-[10px]">PWR <span className="text-slate-300 font-bold">84%</span></span>
                  </div>
                  <div className="flex items-center gap-2 border-l border-slate-800 pl-4">
                    <span className="text-slate-500 text-[10px] tracking-widest">MISSION</span>
                    <span className="text-slate-200 font-mono text-sm">{selectedPolicy.toUpperCase()}</span>
                  </div>
                  <div className="flex items-center gap-2 border-l border-slate-800 pl-4">
                    <span className="text-slate-500 text-[10px] tracking-widest">UPDATED</span>
                    <span className="text-slate-200 font-mono text-sm">{liveTime}</span>
                  </div>
                </div>

                {/* Row 2 — Extended telemetry (visible when expanded) */}
                {bottomPanelHeight > 120 && (
                  <div className="grid grid-cols-4 gap-4 mt-4 pt-4 border-t border-slate-800/50">
                    <div className="flex flex-col">
                      <span className="text-slate-500 text-[10px] tracking-widest mb-1">FLEET UNITS</span>
                      <span className="text-[#22d3ee] font-mono font-bold">14 <span className="text-slate-500 font-normal">/ 15</span></span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-slate-500 text-[10px] tracking-widest mb-1">AVG SPEED</span>
                      <span className="text-[#22d3ee] font-mono font-bold">0.22 <span className="text-slate-500 font-normal text-[10px]">m/s</span></span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-slate-500 text-[10px] tracking-widest mb-1">CBF ACTIVE</span>
                      <span className="text-emerald-400 font-mono font-bold">YES <span className="text-slate-500 font-normal text-[10px]">constraints enforced</span></span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-slate-500 text-[10px] tracking-widest mb-1">SAFETY VIOLATIONS</span>
                      <span className="text-emerald-400 font-mono font-bold">{telemetry.metrics.violationsCurrent}</span>
                    </div>
                  </div>
                )}

                {/* Row 3 — Even more detail */}
                {bottomPanelHeight > 200 && (
                  <div className="grid grid-cols-4 gap-4 mt-4 pt-4 border-t border-slate-800/50">
                    <div className="flex flex-col">
                      <span className="text-slate-500 text-[10px] tracking-widest mb-1">PATH EFFICIENCY</span>
                      <span className="text-[#22d3ee] font-mono font-bold">{telemetry.metrics.efficiencyCurrent}%</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-slate-500 text-[10px] tracking-widest mb-1">CONSTRAINT ADHERENCE</span>
                      <span className="text-[#22d3ee] font-mono font-bold">{telemetry.metrics.adherenceCurrent}%</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-slate-500 text-[10px] tracking-widest mb-1">INTERVENTIONS (24h)</span>
                      <span className="text-amber-400 font-mono font-bold">{telemetry.metrics.cbfCurrent}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-slate-500 text-[10px] tracking-widest mb-1">ZONE SPEED MODE</span>
                      <span className="text-amber-400 font-mono font-bold">ADAPTIVE <span className="text-slate-500 font-normal text-[10px]">ICU/Ward slow</span></span>
                    </div>
                  </div>
                )}

                {/* Row 4 — Alert log */}
                {bottomPanelHeight > 300 && (
                  <div className="mt-4 pt-4 border-t border-slate-800/50">
                    <span className="text-slate-500 text-[10px] tracking-widest block mb-2">RECENT ALERTS</span>
                    <div className="space-y-1 text-xs font-mono">
                      <div className="flex gap-3 text-amber-400"><span className="text-slate-600">{liveTime}</span> Unit 02 — Obstruction detected, rerouting via Corridor B</div>
                      <div className="flex gap-3 text-slate-400"><span className="text-slate-600">{liveTime}</span> G1 entered ICU slow zone — speed reduced to 30%</div>
                      <div className="flex gap-3 text-emerald-400"><span className="text-slate-600">{liveTime}</span> FastBot completed Ward B patrol checkpoint</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN - Minimap */}
        <div 
          className={`group/right flex flex-col h-full shrink-0 transition-all duration-300 ease-in-out relative z-30 ${
            rightPanelOpen ? "w-full xl:w-[380px]" : "w-8 xl:w-8 hover:w-full hover:xl:w-[380px] cursor-pointer"
          }`}
          onClick={() => { if (!rightPanelOpen) setRightPanelOpen(true); }}
        >
          {/* Collapsed Indicator */}
          <div className={`absolute inset-y-0 right-0 w-8 glass-panel flex flex-col items-center justify-center border-l border-[#22d3ee]/30 transition-opacity duration-300 ${rightPanelOpen ? 'opacity-0 pointer-events-none' : 'opacity-100 group-hover/right:opacity-0'}`}>
             <div className="text-[#22d3ee]/80 transform rotate-90 whitespace-nowrap tracking-[0.3em] text-xs font-bold">
                MINIMAP
             </div>
          </div>

          <div className={`w-full xl:w-[380px] glass-panel p-4 flex flex-col h-full transition-opacity duration-300 ${rightPanelOpen ? 'opacity-100' : 'opacity-0 group-hover/right:opacity-100 pointer-events-none group-hover/right:pointer-events-auto'}`}>
            <div className="flex justify-between items-center border-b border-slate-800 pb-3 mb-3">
            <h2 className="font-bold tracking-widest text-sm text-slate-200">LIVE PATROL</h2>
            <PanelRight size={16} className="text-slate-500 cursor-pointer hover:text-white transition" onClick={(e) => { e.stopPropagation(); setRightPanelOpen(false); }} />
          </div>
          
          {/* Live FastBot Hospital Patrol — embedded (compact) */}
          <div className="w-full border border-[#22d3ee]/20 rounded-lg relative bg-slate-950/40 overflow-hidden" style={{ height: '200px' }}>
            <iframe 
              src="http://localhost:8080/fastbot_dashboard.html"
              className="w-full h-full border-0 rounded-lg"
              style={{ height: '200px' }}
              title="FastBot Hospital Patrol — Live"
            />
          </div>

          {/* Hospital Minimap — expanded */}
          <div className="mt-3 border-t border-slate-800 pt-3 flex-1 flex flex-col">
            <span className="text-[10px] text-slate-500 tracking-widest font-bold block mb-2">HOSPITAL MINIMAP — LEVEL 4</span>
            <div className="flex-1 w-full border border-slate-700/50 rounded-lg p-2 relative bg-slate-950/40">
              <LiveMinimap />
            </div>
          </div>
          
          <div className="flex items-center gap-4 mt-3 text-xs font-mono text-slate-400">
            <div className="flex items-center gap-2"><div className="w-3 h-3 bg-slate-400/20 border border-slate-500"></div> WALLS</div>
            <div className="flex items-center gap-2"><div className="w-3 h-3 bg-[#22d3ee]/20 border border-[#22d3ee]"></div> OBSTACLES</div>
          </div>

          <div className="border-t border-slate-800 mt-3 pt-3 flex justify-between items-center text-slate-400">
             <div 
               className="flex items-center gap-2 bg-[#22d3ee]/10 text-[#22d3ee] px-3 py-1.5 rounded text-[10px] font-bold tracking-widest cursor-pointer hover:bg-[#22d3ee]/20 transition border border-[#22d3ee]/30 shadow-[0_0_10px_rgba(34,211,238,0.1)]"
               onClick={() => window.open('http://localhost:8000/api/fleet/logs/download', '_blank')}
             >
               <FileText size={14} /> EXPORT LOGS
             </div>
             <div className="flex gap-3">
               <List size={16} className="cursor-pointer hover:text-white transition" />
               <Clock size={16} className="cursor-pointer hover:text-white transition" />
              <Settings size={16} className="cursor-pointer hover:text-white transition" />
             </div>
            </div>
          </div>
        </div>

      </div>
    </main>
  );
}
