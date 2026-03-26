'use client';

import React, { useState } from 'react';
import dynamic from 'next/dynamic';
import { Mail, Bell, AlertTriangle, FileText, List, Clock, Settings, User, Info, Terminal, Play, Pause, Maximize, Minimize, PanelLeft, PanelRight } from 'lucide-react';
import { useTelemetry } from '@/hooks/useTelemetry';
import { useFleetAPI } from '@/hooks/useFleetAPI';

const RobotViewer = dynamic(() => import('@/components/RobotViewer'), { ssr: false });

const MinimapSVG = () => (
  <svg viewBox="0 0 200 400" className="w-full h-full text-gray-500 font-mono text-[8px]" fill="none">
    {/* Floor Plan Outlines */}
    <path d="M20,20 L180,20 L180,380 L20,380 Z" stroke="#334155" strokeWidth="2" />
    <path d="M20,100 L80,100 M120,100 L180,100" stroke="#334155" strokeWidth="2" />
    <path d="M20,200 L80,200 M120,200 L180,200" stroke="#334155" strokeWidth="2" />
    <path d="M20,300 L80,300 M120,300 L180,300" stroke="#334155" strokeWidth="2" />
    
    <path d="M100,20 L100,60 M100,120 L100,180 M100,220 L100,280 M100,320 L100,380" stroke="#334155" strokeWidth="2" />

    {/* Rooms */}
    <rect x="25" y="110" width="30" height="30" fill="#22d3ee" fillOpacity="0.1" stroke="#22d3ee" strokeWidth="1" />
    <text x="30" y="160" fill="#22d3ee">Unit 02</text>
    
    <rect x="145" y="210" width="30" height="30" fill="#22d3ee" fillOpacity="0.1" stroke="#22d3ee" strokeWidth="1" />
    <text x="145" y="260" fill="#22d3ee">Unit 05</text>

    {/* Pathing Line */}
    <path d="M150,350 L100,350 L100,280 L140,280 L140,150 L60,150" stroke="#22d3ee" strokeWidth="2" strokeDasharray="4 4" className="graph-path-cyan" />
    
    {/* Robot Icon */}
    <circle cx="60" cy="150" r="5" fill="#22d3ee" className="graph-path-cyan" />
    
    <circle cx="150" cy="350" r="3" fill="#22d3ee" />
  </svg>
);

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
          className={`group/right flex flex-col h-full shrink-0 transition-all duration-300 ease-in-out relative z-30 ${
            rightPanelOpen ? "w-full xl:w-[320px]" : "w-8 xl:w-8 hover:w-full hover:xl:w-[320px] cursor-pointer"
          }`}
          onClick={() => { if (!rightPanelOpen) setRightPanelOpen(true); }}
        >
          {/* Collapsed Indicator */}
          <div className={`absolute inset-y-0 right-0 w-8 glass-panel flex flex-col items-center justify-center border-l border-[#22d3ee]/30 transition-opacity duration-300 ${rightPanelOpen ? 'opacity-0 pointer-events-none' : 'opacity-100 group-hover/right:opacity-0'}`}>
             <div className="text-[#22d3ee]/80 transform rotate-90 whitespace-nowrap tracking-[0.3em] text-xs font-bold">
                MINIMAP
             </div>
          </div>

          <div className={`w-full xl:w-[320px] glass-panel p-5 flex flex-col h-full transition-opacity duration-300 ${rightPanelOpen ? 'opacity-100' : 'opacity-0 group-hover/right:opacity-100 pointer-events-none group-hover/right:pointer-events-auto'}`}>
            <div className="flex justify-between items-center border-b border-slate-800 pb-3 mb-4">
            <h2 className="font-bold tracking-widest text-sm text-slate-200">HOSPITAL MINIMAP</h2>
            <PanelRight size={16} className="text-slate-500 cursor-pointer hover:text-white transition" onClick={(e) => { e.stopPropagation(); setRightPanelOpen(false); }} />
          </div>
          
          <span className="text-xs text-slate-400 mb-4 block">LEVEL 4 - ICU/PATIENT WARDS</span>
          
          <div className="flex-1 w-full border border-slate-700/50 rounded-lg p-2 relative bg-slate-950/40">
            <MinimapSVG />
          </div>
          
          <div className="flex items-center gap-4 mt-4 text-xs font-mono text-slate-400">
            <div className="flex items-center gap-2"><div className="w-3 h-3 bg-slate-400/20 border border-slate-500"></div> WALLS</div>
            <div className="flex items-center gap-2"><div className="w-3 h-3 bg-[#22d3ee]/20 border border-[#22d3ee]"></div> OBSTACLES</div>
          </div>

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
