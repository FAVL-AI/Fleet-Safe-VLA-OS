import React, { useState, useEffect } from 'react';
import { Activity } from 'lucide-react';

export default function EngineMetrics() {
  const [metrics, setMetrics] = useState({ hz: 1000, latency: 250 });

  // Simulate real-time native engine variations
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics({
        hz: Math.floor(Math.random() * (1050 - 950 + 1) + 950),
        latency: Math.floor(Math.random() * (350 - 150 + 1) + 150),
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-slate-900/40 p-3 rounded border border-slate-700/50 backdrop-blur-md">
      <div className="flex items-center gap-2 mb-3">
        <Activity className="text-blue-400 w-4 h-4" />
        <h3 className="text-white text-xs font-bold tracking-widest uppercase">FleetSafe Engine (Rust)</h3>
        <span className="ml-auto flex items-center gap-1 text-[8px] text-green-400 font-mono bg-green-900/30 px-1 py-0.5 rounded border border-green-500/30">
          <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></span> ONLINE
        </span>
      </div>
      
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-slate-950/60 p-2 rounded border border-slate-800/80">
          <p className="text-slate-500 text-[9px] mb-1 font-mono uppercase">Control Loop Frequency</p>
          <p className="text-white text-sm font-black font-mono tracking-tight">{metrics.hz} <span className="text-slate-500 text-[10px] font-normal tracking-wide">Hz</span></p>
        </div>
        <div className="bg-slate-950/60 p-2 rounded border border-slate-800/80">
          <p className="text-slate-500 text-[9px] mb-1 font-mono uppercase">Zero-Copy Latency</p>
          <p className="text-blue-300 text-sm font-black font-mono tracking-tight">{metrics.latency} <span className="text-slate-500 text-[10px] font-normal tracking-wide">μs</span></p>
        </div>
      </div>
    </div>
  );
}
