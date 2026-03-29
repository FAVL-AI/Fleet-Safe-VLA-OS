import React, { useState, useEffect } from 'react';
import { Shield, ShieldAlert, HeartPulse } from 'lucide-react';

export default function SafetyKernelStatus() {
  const [safety, setSafety] = useState({ interventions: 50, violations: 10, adherence: 85.0 });

  // Simulate incoming interventions based on Safety Kernel activity
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate slow drift in interventions and adherence
      setSafety((prev) => {
        const rnd = Math.random();
        // 5% chance every 2 seconds to have an intervention
        const newInt = rnd < 0.05 ? prev.interventions + 1 : prev.interventions;
        // Compute adherence 80-100%
        const newAdh = Math.max(80, 100 - (newInt % 20));
        return {
          interventions: newInt,
          violations: prev.violations,
          adherence: newAdh
        };
      });
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-slate-900/40 p-3 rounded border border-slate-700/50 backdrop-blur-md">
      <div className="flex items-center gap-2 mb-3">
        <Shield className="text-emerald-400 w-4 h-4" />
        <h3 className="text-white text-xs font-bold tracking-widest uppercase">FleetSafe Kernel (Python)</h3>
      </div>
      
      <div className="grid grid-cols-2 gap-2 mb-2">
        <div className="bg-slate-950/60 p-2 rounded border border-slate-800/80">
          <p className="text-slate-500 text-[9px] mb-1 font-mono uppercase">VLA Iterative Interventions</p>
          <div className="flex items-end gap-2">
            <p className="text-yellow-400 text-sm font-black tracking-tight">{safety.interventions}</p>
            <HeartPulse className="w-3 h-3 text-yellow-500/50 mb-1" />
          </div>
        </div>

        <div className="bg-slate-950/60 p-2 rounded border border-slate-800/80">
          <p className="text-slate-500 text-[9px] mb-1 font-mono uppercase">CMDP Safety Adherence</p>
          <div className="flex items-end gap-2">
            <p className="text-emerald-400 text-sm font-black tracking-tight">{safety.adherence.toFixed(1)}%</p>
          </div>
        </div>
      </div>

      <div className="bg-rose-950/30 p-2 rounded border border-rose-900/50 flex justify-between items-center">
        <div className="flex gap-2 items-center">
            <ShieldAlert className="w-3 h-3 text-rose-500" />
            <span className="text-rose-400 text-[9px] font-mono uppercase font-bold tracking-widest">Catastrophic Violations</span>
        </div>
        <p className="text-rose-400 text-sm font-black font-mono">{safety.violations}</p>
      </div>
    </div>
  );
}
