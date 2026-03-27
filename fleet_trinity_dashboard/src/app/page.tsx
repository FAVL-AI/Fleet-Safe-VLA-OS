'use client';

import React, { useState } from 'react';
import dynamic from 'next/dynamic';
import { Mail, Bell, AlertTriangle, FileText, List, Clock, Settings, User, Info, Terminal, Play, Pause, Maximize, Minimize, PanelLeft, PanelRight, ChevronDown, ChevronUp } from 'lucide-react';
import { useTelemetry } from '@/hooks/useTelemetry';
import { useFleetAPI } from '@/hooks/useFleetAPI';

const RobotViewer = dynamic(() => import('@/components/RobotViewer'), { ssr: false });

const MinimapSVG = () => (
  <svg viewBox="0 0 400 600" className="w-full h-full text-gray-500 font-mono text-[8px]" fill="none">
    <style>
      {`
        .dash-anim { stroke-dasharray: 6 6; animation: dash 20s linear infinite; }
        .pulse-ring { animation: radar 3s ease-out infinite; }
        @keyframes dash { to { stroke-dashoffset: -400; } }
        @keyframes radar { 0% { r: 5px; stroke-width: 2px; opacity: 0.8; } 100% { r: 35px; stroke-width: 0.5px; opacity: 0; } }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
      `}
    </style>
    {/* Map Background grid */}
    <rect width="400" height="600" fill="#0f111a" opacity="0.3" />
    
    {/* ZONES */}
    {/* ICU - Top Left */}
    <rect x="20" y="20" width="160" height="120" fill="#ef4444" fillOpacity="0.05" stroke="#ef4444" strokeWidth="1" strokeDasharray="4 4" />
    <text x="30" y="40" fill="#ef4444" fontSize="12" fontWeight="bold" letterSpacing="2">ICU WARD (RESTRICTED)</text>
    <text x="30" y="55" fill="#ef4444" fontSize="9" opacity="0.8">MAX SPEED: 0.5M/S</text>

    {/* Pharmacy - Top Right */}
    <rect x="220" y="20" width="160" height="120" fill="#3b82f6" fillOpacity="0.08" stroke="#3b82f6" strokeWidth="1" />
    <text x="230" y="40" fill="#60a5fa" fontSize="12" fontWeight="bold" letterSpacing="2">PHARMACY DISPENSARY</text>
    <text x="230" y="55" fill="#60a5fa" fontSize="9" opacity="0.8">SECURE ZONE - ID REQ</text>
    
    {/* Main Corridors */}
    <rect x="180" y="20" width="40" height="420" fill="#334155" fillOpacity="0.2" />
    <rect x="20" y="140" width="360" height="40" fill="#334155" fillOpacity="0.2" />
    <rect x="20" y="280" width="360" height="40" fill="#334155" fillOpacity="0.2" />
    
    {/* Corridor Alpha Details (2 Lanes) */}
    <line x1="200" y1="20" x2="200" y2="440" stroke="#475569" strokeWidth="1" strokeDasharray="4 4" />
    <text x="188" y="280" fill="#94a3b8" fontSize="6" transform="rotate(-90 188 280)" letterSpacing="1">↓ SOUTHBOUND (1.5M/S)</text>
    <text x="215" y="280" fill="#94a3b8" fontSize="6" transform="rotate(-90 215 280)" letterSpacing="1">↑ NORTHBOUND (1.5M/S)</text>
    
    {/* General Wards and Rooms */}
    <rect x="20" y="180" width="160" height="100" fill="#22d3ee" fillOpacity="0.03" stroke="#334155" strokeWidth="1" />
    <text x="30" y="200" fill="#94a3b8" fontSize="11" fontWeight="bold">PATIENT WARD A</text>
    {/* Ward A Beds */}
    <g stroke="#334155" strokeWidth="1">
      <rect x="30" y="220" width="25" height="45" fill="#1e293b" rx="2" />
      <rect x="35" y="225" width="15" height="8" fill="#475569" rx="2" />
      <rect x="30" y="240" width="25" height="25" fill="#0284c7" fillOpacity="0.15" />
      
      <rect x="70" y="220" width="25" height="45" fill="#1e293b" rx="2" />
      <rect x="75" y="225" width="15" height="8" fill="#475569" rx="2" />
      <rect x="70" y="240" width="25" height="25" fill="#0284c7" fillOpacity="0.15" />

      <rect x="110" y="220" width="25" height="45" fill="#1e293b" rx="2" />
      <rect x="115" y="225" width="15" height="8" fill="#475569" rx="2" />
      <rect x="110" y="240" width="25" height="25" fill="#0284c7" fillOpacity="0.15" />
    </g>
    
    <rect x="220" y="180" width="160" height="100" fill="#22d3ee" fillOpacity="0.03" stroke="#334155" strokeWidth="1" />
    <text x="230" y="200" fill="#94a3b8" fontSize="11" fontWeight="bold">PATIENT WARD B</text>
    {/* Ward B Beds */}
    <g stroke="#334155" strokeWidth="1">
      <rect x="240" y="220" width="25" height="45" fill="#1e293b" rx="2" />
      <rect x="245" y="225" width="15" height="8" fill="#475569" rx="2" />
      <rect x="240" y="240" width="25" height="25" fill="#0284c7" fillOpacity="0.15" />

      <rect x="280" y="220" width="25" height="45" fill="#1e293b" rx="2" />
      <rect x="285" y="225" width="15" height="8" fill="#475569" rx="2" />
      <rect x="280" y="240" width="25" height="25" fill="#0284c7" fillOpacity="0.15" />

      <rect x="320" y="220" width="25" height="45" fill="#1e293b" rx="2" />
      <rect x="325" y="225" width="15" height="8" fill="#475569" rx="2" />
      <rect x="320" y="240" width="25" height="25" fill="#0284c7" fillOpacity="0.15" />
    </g>

    {/* Radiology / MRI - Bottom Left Room */}
    <rect x="20" y="320" width="160" height="120" fill="#8b5cf6" fillOpacity="0.05" stroke="#334155" strokeWidth="1" />
    <text x="30" y="340" fill="#a78bfa" fontSize="11" fontWeight="bold">RADIOLOGY / MRI</text>
    {/* MRI Machine */}
    <rect x="60" y="360" width="60" height="40" fill="#1e293b" stroke="#64748b" strokeWidth="2" rx="10" />
    <circle cx="90" cy="380" r="12" fill="#0f111a" stroke="#64748b" strokeWidth="2" />
    <rect x="30" y="375" width="30" height="10" fill="#334155" rx="2" />

    {/* Triage / Outpatient - Bottom Right Room */}
    <rect x="220" y="320" width="160" height="120" fill="#10b981" fillOpacity="0.05" stroke="#334155" strokeWidth="1" />
    <text x="230" y="340" fill="#34d399" fontSize="11" fontWeight="bold">TRIAGE / OUTPATIENT</text>
    {/* Triage Exam Tables */}
    <g stroke="#334155" strokeWidth="1" fill="#1e293b">
      <rect x="240" y="360" width="40" height="15" rx="2" />
      <rect x="240" y="390" width="40" height="15" rx="2" />
      <rect x="300" y="360" width="40" height="15" rx="2" />
      <rect x="300" y="390" width="40" height="15" rx="2" />
    </g>

    {/* Staff and Wheelchairs */}
    <g fill="#3b82f6">
      <circle cx="150" cy="240" r="4" />
      <text x="156" y="243" fontSize="7" fill="#60a5fa">Staff (RN)</text>
      
      <circle cx="260" cy="380" r="4" />
      <text x="266" y="383" fontSize="7" fill="#60a5fa">Doctor</text>
    </g>
    
    <g stroke="#94a3b8" strokeWidth="1.5" fill="none">
      {/* Wheelchairs */}
      <circle cx="40" cy="285" r="4" />
      <line x1="36" y1="285" x2="44" y2="285" />
      <text x="30" y="278" fontSize="6" fill="#94a3b8" stroke="none">Wheelchair</text>

      <circle cx="340" cy="420" r="4" />
      <line x1="336" y1="420" x2="344" y2="420" />
    </g>
    
    {/* Crowded Lobby - Bottom */}
    <rect x="20" y="440" width="360" height="140" fill="#facc15" fillOpacity="0.05" stroke="#facc15" strokeWidth="1" strokeDasharray="2 2" />
    <text x="30" y="460" fill="#facc15" fontSize="12" fontWeight="bold" letterSpacing="2">MAIN LOBBY (HIGH TRAFFIC)</text>
    <text x="30" y="475" fill="#facc15" fontSize="9" opacity="0.8">CBF OBSTACLE AVOIDANCE: MAXIMUM</text>

    {/* WALLS / STRUCTURES */}
    <path d="M20,20 L380,20 L380,580 L20,580 Z" stroke="#475569" strokeWidth="3" />
    <path d="M180,20 L180,140 M220,20 L220,140 M180,180 L180,280 M220,180 L220,280 M180,320 L180,440 M220,320 L220,440" stroke="#475569" strokeWidth="2" />
    <path d="M20,140 L180,140 M220,140 L380,140 M20,180 L180,180 M220,180 L380,180" stroke="#475569" strokeWidth="2" />
    <path d="M20,280 L180,280 M220,280 L380,280 M20,320 L180,320 M220,320 L380,320" stroke="#475569" strokeWidth="2" />
    <path d="M20,440 L380,440" stroke="#475569" strokeWidth="2" />

    {/* HUMANS / OBSTACLES (Orange/Red Dots) */}
    <g fill="#f97316">
      {/* Corridor Humans */}
      <circle cx="200" cy="110" r="5" />
      <text x="210" y="113" fontSize="8" fill="#f97316">Pedestrian (0.8m/s)</text>
      <circle cx="195" cy="122" r="4" />
      
      {/* Lobby Crowd */}
      <circle cx="80" cy="500" r="5" /><circle cx="100" cy="520" r="4" />
      <circle cx="130" cy="480" r="5" /><circle cx="150" cy="540" r="4" />
      <circle cx="280" cy="510" r="5" /><circle cx="320" cy="490" r="4" />
      <circle cx="300" cy="550" r="5" /><circle cx="250" cy="530" r="4" />
      <text x="160" y="510" fontSize="10" fill="#facc15" fontWeight="bold">DENSITY: HIGH</text>
    </g>

    {/* STATIC OBSTACLES (Medical Carts) */}
    <rect x="50" y="80" width="20" height="14" fill="#64748b" />
    <text x="45" y="70" fontSize="8" fill="#94a3b8">Med Cart</text>
    <rect x="240" y="240" width="20" height="14" fill="#64748b" />
    
    {/* CHARGING POINTS AND DOCKS */}
    <rect x="180" y="540" width="40" height="40" fill="#22d3ee" fillOpacity="0.1" stroke="#22d3ee" strokeWidth="1" strokeDasharray="2 2" />
    <text x="182" y="555" fill="#22d3ee" fontSize="8" fontWeight="bold">ROBOT BASE</text>
    <circle cx="200" cy="565" r="3" fill="#22d3ee" />
    
    {/* Main Charging Hub */}
    <rect x="340" y="540" width="40" height="40" fill="#facc15" fillOpacity="0.1" stroke="#facc15" strokeWidth="1.5" strokeDasharray="2 2" />
    <text x="342" y="550" fill="#facc15" fontSize="8" fontWeight="bold">MAIN</text>
    <text x="342" y="560" fill="#facc15" fontSize="8" fontWeight="bold">CHARGE</text>
    <polygon points="360,565 355,572 358,572 354,580 364,570 361,570 365,565" fill="#facc15" />

    {/* Secondary Charging Points */}
    <rect x="150" y="180" width="30" height="15" fill="#facc15" fillOpacity="0.1" stroke="#facc15" strokeWidth="1" />
    <text x="152" y="188" fill="#facc15" fontSize="6" fontWeight="bold">[⚡] CHARGE</text>

    <rect x="350" y="320" width="30" height="15" fill="#facc15" fillOpacity="0.1" stroke="#facc15" strokeWidth="1" />
    <text x="352" y="328" fill="#facc15" fontSize="6" fontWeight="bold">[⚡] CHARGE</text>

    {/* Side Corridor Charging Point */}
    <rect x="130" y="270" width="25" height="10" fill="#facc15" fillOpacity="0.1" stroke="#facc15" strokeWidth="1" />
    <text x="133" y="277" fill="#facc15" fontSize="5" fontWeight="bold">[⚡] DOCK</text>

    {/* Human Obstacles & Dynamic UI */}
    <g transform="translate(215, 250)">
       <circle cx="0" cy="0" r="5" fill="#f97316" className="pulse-ring" />
       
       {/* Warning Box */}
       <rect x="-65" y="-15" width="56" height="38" fill="#ef4444" fillOpacity="0.15" stroke="#ef4444" strokeWidth="0.5" rx="2" />
       <text x="-60" y="-5" fill="#fca5a5" fontSize="6" fontWeight="bold">! YIELDING</text>
       <text x="-60" y="3" fill="#fca5a5" fontSize="6">Dist: 1.2m (Limit: 1.0)</text>
       <text x="-60" y="11" fill="#fca5a5" fontSize="6">Traffic: UNIT 05</text>
       <text x="-60" y="19" fill="#fca5a5" fontSize="6">Status: WAITING</text>
    </g>

    {/* ROBOTS AND PATHS */}

    {/* Robot 1: Unitree G1 (Unit 02) - Complete Workflow Cycle (60s) */}
    {/* Travels strictly on Left (185) and Right (215) lanes close to walls */}
    <path id="path-u2" d="M215,560 L215,340 Q185,250 215,160 L215,70 L280,70 L285,70 L280,70 L215,70 L185,70 L185,220 L100,220 L105,220 L100,220 L185,220 L185,560 L360,560 L355,560 L360,560 L215,560" stroke="#22d3ee" strokeWidth="2.5" className="dash-anim" fill="none" />
    <path d="M215,340 Q185,250 215,160" stroke="#22d3ee" strokeWidth="2" strokeDasharray="3 3" strokeOpacity="0.8" fill="none" />
    <text x="220" y="200" fill="#22d3ee" fontSize="7">Clearance Granted (4.2s)</text>

    {/* Unit 02 body */}
    <g>
      <circle r="25" stroke="#22d3ee" strokeWidth="1" className="pulse-ring" />
      <circle r="8" fill="#22d3ee" />
      <rect x="15" y="-10" width="105" height="30" fill="#0f111a" fillOpacity="0.9" stroke="#22d3ee" strokeWidth="1" rx="4" />
      <text x="20" y="2" fill="#22d3ee" fontSize="10" fontWeight="bold">UNITREE G1 (02)</text>
      <text x="20" y="14" fill="#22d3ee" fontSize="8">Task: Full Workflow Sequence</text>
      <animateMotion dur="60s" repeatCount="indefinite">
        <mpath href="#path-u2" />
      </animateMotion>
    </g>

    {/* Robot 2: Delivery Bot (Unit 05) - Side Corridor Docking Cycle (30s loop) */}
    <path id="path-u5" d="M185,70 L185,290 L130,290 L140,275 L145,275 L140,275 L130,290 L185,290 L185,440 L215,440 L215,340 Q185,250 215,160 L215,70 L185,70" stroke="#a855f7" strokeWidth="2.5" className="dash-anim" fill="none" />
    <g>
      <circle r="25" stroke="#a855f7" strokeWidth="1" className="pulse-ring" />
      <circle r="8" fill="#a855f7" />
      <rect x="15" y="-6" width="70" height="20" fill="#0f111a" fillOpacity="0.8" stroke="#a855f7" strokeWidth="1" rx="3" />
      <text x="20" y="7" fill="#a855f7" fontSize="9" fontWeight="bold">UNIT 05 (FETCH)</text>
      <animateMotion dur="30s" repeatCount="indefinite">
        <mpath href="#path-u5" />
      </animateMotion>
    </g>

    {/* Robot 3: Disinfection (Unit 08) - Green */}
    <path id="path-u8" d="M120,520 L180,480 L200,440 L200,400" stroke="#22c55e" strokeWidth="2.5" className="dash-anim" fill="none" />
    <g>
      <circle r="25" stroke="#22c55e" strokeWidth="1" className="pulse-ring" />
      <circle r="8" fill="#22c55e" />
      <rect x="15" y="-10" width="100" height="26" fill="#0f111a" fillOpacity="0.8" stroke="#22c55e" strokeWidth="1" rx="3" />
      <text x="20" y="0" fill="#22c55e" fontSize="9" fontWeight="bold">UNIT 08 (UV)</text>
      <text x="20" y="10" fill="#22c55e" fontSize="8">Navigating Crowd</text>
      <animateMotion dur="25s" repeatCount="indefinite">
        <mpath href="#path-u8" />
      </animateMotion>
    </g>

  </svg>
);

const LiveLogs = () => {
  const [isOpen, setIsOpen] = useState(true);
  const [logs, setLogs] = useState<{time: string, unit: string, type: string, msg: string, zone: string}[]>([
    {time: "--:--:--", unit: "SYSTEM", type: "INFO", zone: "HQ", msg: "Digital Twin Workflow Initialized."},
  ]);

  React.useEffect(() => {
    const workflowEvents = [
      { unit: "UNIT 02 (UNITREE G1)", type: "TASK", zone: "ACTIVE BASE", msg: "Time in Zone: 5s. Instruction received: Pharmacy Med Pickup." },
      { unit: "UNIT 02 (UNITREE G1)", type: "INFO", zone: "CORRIDOR ALPHA", msg: "Nominal speed 1.5m/s. Hugging Right Wall Northbound." },
      { unit: "UNIT 05 (FASTBOT)", type: "INFO", zone: "SIDE CORRIDOR", msg: "Time in Zone: 12s. Tracking active: 3 units in 20m radius. No overcrowding detected." },
      { unit: "UNIT 02 (UNITREE G1)", type: "WARN", zone: "CORRIDOR ALPHA", msg: "Human obstacle ahead. Oncoming FastBot traffic restricting evasion. Halting." },
      { unit: "UNIT 02 (UNITREE G1)", type: "CBF_ACTIVE", zone: "CORRIDOR ALPHA", msg: "WAITING (4.2s elapsed): Yielding to Southbound FastBot to prevent collusion." },
      { unit: "UNIT 05 (FASTBOT)", type: "TASK", zone: "SIDE CORRIDOR", msg: "Battery Low (22%). Executing auto-docking sequence into side corridor port." },
      { unit: "UNIT 02 (UNITREE G1)", type: "INFO", zone: "CORRIDOR ALPHA", msg: "Southbound lane clear (4.2s waited). Re-routing via CBF. Safe Dist: 1.2m." },
      { unit: "UNIT 02 (UNITREE G1)", type: "INFO", zone: "PHARMACY", msg: "Arrived at Dispensary. Authenticating..." },
      { unit: "UNIT 05 (FASTBOT)", type: "INFO", zone: "SIDE CORRIDOR", msg: "Successfully docked. Self-charging initiated..." },
      { unit: "UNIT 02 (UNITREE G1)", type: "TASK", zone: "PHARMACY", msg: "Time in Zone: 8s. Medicine payload secured. Route set to WARD A." },
      { unit: "UNIT 02 (UNITREE G1)", type: "INFO", zone: "PATIENT WARD A", msg: "Time in Zone: 6s. Medicine safely delivered to Patient Ward A." },
      { unit: "UNIT 02 (UNITREE G1)", type: "WARN", zone: "PATIENT WARD A", msg: "Battery Level 18%. Mandatory return to Charging Dock." },
      { unit: "UNIT 02 (UNITREE G1)", type: "INFO", zone: "CORRIDOR ALPHA", msg: "Hugging Left Wall Southbound toward Charging Bay." },
      { unit: "UNIT 05 (FASTBOT)", type: "INFO", zone: "SIDE CORRIDOR", msg: "Time in Zone: 45m. Charge 80% complete. Undocking to resume sweeper patrol." },
      { unit: "UNIT 02 (UNITREE G1)", type: "TASK", zone: "CHARGING DOCK", msg: "Auto-charge initiated. Self-charge logging time to 100%: 45m." },
      { unit: "UNIT 02 (UNITREE G1)", type: "INFO", zone: "ACTIVE BASE", msg: "Time in Zone: 2m. Safely back to Active Base. Ready." }
    ];
    let i = 0;
    const interval = setInterval(() => {
      const ev = workflowEvents[i];
      const timeStr = new Date().toLocaleTimeString('en-US', {hour12:false});
      setLogs(prev => [...prev, {
        time: timeStr,
        unit: ev.unit, type: ev.type, zone: ev.zone, msg: ev.msg
      }].slice(-10));
      
      i = (i + 1) % workflowEvents.length;
    }, 5000); // 12 * 5s = 60s total loop
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={`mt-4 border border-slate-700/50 rounded-lg bg-slate-950/80 p-3 flex flex-col transition-all duration-300 ${isOpen ? 'min-h-[260px] flex-1 max-h-[400px]' : 'h-[44px] min-h-[44px] overflow-hidden flex-none'}`}>
      <div 
        className={`flex justify-between items-center cursor-pointer select-none ${isOpen ? 'mb-2 border-b border-slate-800 pb-2' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center gap-2">
          {isOpen ? <ChevronDown size={14} className="text-slate-400" /> : <ChevronUp size={14} className="text-slate-400" />}
          <h3 className="font-bold text-[10px] text-slate-300 tracking-widest uppercase">LIVE FLEET TELEMETRY LOGS</h3>
        </div>
        <div className="flex gap-1 items-center">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
          <span className="text-[8px] text-green-400 font-bold tracking-widest">RECORDING</span>
        </div>
      </div>
      <div className={`flex-1 overflow-y-auto flex flex-col gap-1.5 font-mono text-[9px] pr-2 ${isOpen ? 'opacity-100' : 'opacity-0 hidden'}`}>
        {logs.map((L, i) => (
          <div key={i} className="flex flex-col py-1 border-b border-slate-800/30 last:border-0 hover:bg-slate-800/50 transition">
            <div className="flex justify-between items-baseline mb-1">
              <div className="flex items-center gap-2">
                <span className="text-slate-500">[{L.time}]</span>
                <span className={`font-bold ${L.type === 'WARN' || L.type === 'CBF_ACTIVE' ? 'text-yellow-400' : 'text-[#22d3ee]'}`}>{L.unit}</span>
              </div>
              <span className="text-slate-400 text-[8px] border border-slate-700 px-1.5 py-0.5 rounded bg-slate-900 tracking-wider inline-block text-right">{L.zone}</span>
            </div>
            <div className={`pl-2 border-l-2 ${L.type === 'CBF_ACTIVE' ? 'border-red-500 text-red-300' : L.type === 'WARN' ? 'border-yellow-500 text-yellow-300' : 'border-[#22d3ee] text-slate-300'}`}>
              <span className="mr-1 opacity-50">&gt;</span> {L.msg}
            </div>
          </div>
        ))}
      </div>
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
  const [dseoMode, setDseoMode] = useState<'Normal' | 'Degraded' | 'Emergency'>('Normal');
  
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
    } catch (e) {
      console.error('Failed to log encounter:', e);
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
      <div className="flex-1 w-full p-6 flex flex-col xl:flex-row gap-6 min-h-[calc(100vh-4rem)] pb-12">
        
        {/* LEFT COLUMN - Health & Graphs */}
        <div 
          className={`group/left flex flex-col gap-4 h-full shrink-0 relative z-30 resize-y overflow-hidden min-h-[500px] max-h-[300vh] [&::-webkit-resizer]:bg-[#22d3ee]/50 ${
            leftPanelOpen ? "w-[320px]" : "w-8 hover:w-[320px] cursor-pointer"
          }`}
          onClick={() => { if (!leftPanelOpen) setLeftPanelOpen(true); }}
        >
          {/* Collapsed Indicator */}
          <div className={`absolute inset-y-0 left-0 w-8 glass-panel flex flex-col items-center justify-center border-r border-[#22d3ee]/30 transition-opacity duration-300 ${leftPanelOpen ? 'opacity-0 pointer-events-none' : 'opacity-100 group-hover/left:opacity-0'}`}>
             <div className="text-[#22d3ee]/80 transform -rotate-90 whitespace-nowrap tracking-[0.3em] text-xs font-bold">
                SYSTEM HEALTH
             </div>
          </div>

          <div className={`w-full flex flex-col gap-4 h-full transition-opacity duration-300 ${leftPanelOpen ? 'opacity-100' : 'opacity-0 group-hover/left:opacity-100 pointer-events-none group-hover/left:pointer-events-auto'}`}>
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
        <div className="flex-1 flex flex-col gap-4 h-full min-w-0 resize-y overflow-hidden min-h-[500px] max-h-[300vh] [&::-webkit-resizer]:bg-[#22d3ee]/50">
          
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
            className="glass-panel w-full relative flex flex-col overflow-hidden min-h-[600px] flex-1"
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

            <div className="absolute bottom-0 left-0 w-full border-t border-slate-800/50 p-4 grid grid-cols-3 gap-4 bg-slate-900/40 backdrop-blur-md z-10">
              <div className="flex flex-col">
                <span className="text-slate-500 text-xs tracking-widest mb-1">OPERATIONAL STATUS:</span>
                <div className="flex items-baseline gap-2">
                  <span className="text-[#22d3ee] font-bold text-lg">ACTIVE</span>
                  <span className="text-slate-400 text-[10px] uppercase flex flex-col"><span className="opacity-70">POWER</span><span className="font-bold">84%</span></span>
                </div>
              </div>
              <div className="flex flex-col border-l border-slate-800 pl-4">
                <span className="text-slate-500 text-xs tracking-widest mb-1">MISSION:</span>
                <span className="text-slate-200 font-mono text-lg tracking-wide">{selectedPolicy.toUpperCase()}</span>
              </div>
              <div className="flex flex-col border-l border-slate-800 pl-4">
                <span className="text-slate-500 text-xs tracking-widest mb-1">LAST UPDATE:</span>
                <span className="text-slate-200 font-mono text-lg tracking-wide">09:41:03</span>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN - Minimap */}
        <div 
          dir="rtl"
          className={`group/right flex flex-col h-full shrink-0 relative z-30 resize-y overflow-hidden min-h-[500px] max-h-[300vh] [&::-webkit-resizer]:bg-[#22d3ee]/50 ${
            rightPanelOpen ? "w-[320px]" : "w-8 hover:w-[320px] cursor-pointer"
          }`}
          onClick={() => { if (!rightPanelOpen) setRightPanelOpen(true); }}
        >
          {/* Collapsed Indicator */}
          <div dir="ltr" className={`absolute inset-y-0 right-0 w-8 glass-panel flex flex-col items-center justify-center border-l border-[#22d3ee]/30 transition-opacity duration-300 ${rightPanelOpen ? 'opacity-0 pointer-events-none' : 'opacity-100 group-hover/right:opacity-0'}`}>
             <div className="text-[#22d3ee]/80 transform rotate-90 whitespace-nowrap tracking-[0.3em] text-xs font-bold">
                MINIMAP
             </div>
          </div>

          <div dir="ltr" className={`w-full glass-panel p-5 flex flex-col h-full overflow-y-auto transition-opacity duration-300 ${rightPanelOpen ? 'opacity-100' : 'opacity-0 group-hover/right:opacity-100 pointer-events-none group-hover/right:pointer-events-auto'}`}>
            <div className="flex justify-between items-center border-b border-slate-800 pb-3 mb-4">
            <h2 className="font-bold tracking-widest text-sm text-slate-200">HOSPITAL MINIMAP</h2>
            <PanelRight size={16} className="text-slate-500 cursor-pointer hover:text-white transition" onClick={(e) => { e.stopPropagation(); setRightPanelOpen(false); }} />
          </div>
          
          <span className="text-xs text-slate-400 mb-4 block">LEVEL 4 - MULTI-ZONE OPERATIONS</span>
          
          <div className="w-full border border-slate-700/50 rounded-lg p-2 relative bg-slate-950/40 min-h-[400px]">
            <MinimapSVG />
          </div>
          
          <div className="flex items-center justify-between mt-4 text-xs font-mono text-slate-400 border-b border-slate-800/80 pb-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2"><div className="w-3 h-3 bg-slate-400/20 border border-slate-500"></div> WALLS</div>
              <div className="flex items-center gap-2"><div className="w-3 h-3 bg-[#f97316]"></div> HUMANS</div>
              <div className="flex items-center gap-2"><div className="w-3 h-3 bg-red-500/10 border border-red-500 border-dashed"></div> ICU</div>
            </div>
          </div>

          {/* Detailed Stream Logs Section */}
          <LiveLogs />

          {/* DSEO Preemption Control */}
          <div className="border-t border-slate-800/80 mt-6 pt-5">
            <h3 className="font-bold text-slate-300 text-[10px] tracking-widest mb-3 uppercase drop-shadow-sm">DSEO Preemption Override</h3>
            <div className="flex flex-col gap-2">
              <button 
                onClick={() => setDseoMode('Normal')}
                className={`px-3 py-2 rounded text-[10px] font-bold tracking-widest flex justify-between items-center transition border ${dseoMode === 'Normal' ? 'bg-[#22d3ee]/20 text-[#22d3ee] border-[#22d3ee]/50 shadow-[0_0_10px_rgba(34,211,238,0.15)]' : 'bg-slate-900/50 text-slate-400 border-slate-700/50 hover:bg-slate-800 hover:text-slate-200'}`}
              >
                <span>NORMAL OPERATION</span>
                {dseoMode === 'Normal' && <span className="w-2 h-2 rounded-full bg-[#22d3ee] animate-pulse"></span>}
              </button>
              <button 
                onClick={() => setDseoMode('Degraded')}
                className={`px-3 py-2 rounded text-[10px] font-bold tracking-widest flex justify-between items-center transition border ${dseoMode === 'Degraded' ? 'bg-[#facc15]/20 text-[#facc15] border-[#facc15]/50 shadow-[0_0_10px_rgba(250,204,21,0.15)]' : 'bg-slate-900/50 text-slate-400 border-slate-700/50 hover:bg-slate-800 hover:text-slate-200'}`}
              >
                <span>DEGRADED (HALF SPEED)</span>
                {dseoMode === 'Degraded' && <span className="w-2 h-2 rounded-full bg-[#facc15] animate-pulse"></span>}
              </button>
              <button 
                onClick={() => setDseoMode('Emergency')}
                className={`px-3 py-2 rounded text-[10px] font-bold tracking-widest flex justify-between items-center transition border ${dseoMode === 'Emergency' ? 'bg-[#ef4444]/20 text-[#ef4444] border-[#ef4444]/50 shadow-[0_0_10px_rgba(239,68,68,0.15)]' : 'bg-slate-900/50 text-slate-400 border-slate-700/50 hover:bg-slate-800 hover:text-slate-200'}`}
              >
                <span>EMERGENCY (E-STOP)</span>
                {dseoMode === 'Emergency' && <span className="w-2 h-2 rounded-full bg-[#ef4444] animate-pulse"></span>}
              </button>
            </div>
          </div>

          <div className="border-t border-slate-800 mt-5 pt-4 flex justify-between items-center text-slate-400">
             <div 
               className="flex items-center gap-2 bg-[#22d3ee]/10 text-[#22d3ee] px-3 py-1.5 rounded text-[10px] font-bold tracking-widest cursor-pointer hover:bg-[#22d3ee]/20 transition border border-[#22d3ee]/30 shadow-[0_0_10px_rgba(34,211,238,0.1)]"
               onClick={() => window.open('http://localhost:8000/api/fleet/logs/download', '_blank')}
             >
               <FileText size={14} /> EXPORT MISSION LOGS
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
