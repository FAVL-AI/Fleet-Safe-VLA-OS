import time
import numpy as np
from fleetsafe_vla.kernel.safety_kernel import SafetyKernel
from fleetsafe_vla.modules.fleet_coordinator import FleetCoordinator

class FleetSafeVLABench:
    def __init__(self, num_robots=3, scenario="hospital_hallway"):
        self.scenario = scenario
        self.num_robots = num_robots
        self.kernel = SafetyKernel()
        self.kernel.load_language_constraints("avoid humans while navigating")
        self.coordinator = FleetCoordinator(num_robots)
        self.results = []
    
    def reset_scenario(self):
        # mock initial states and goals
        states = {f"robot_{i}": {'robot_position': np.array([0.0, float(i), 0.0])} for i in range(self.num_robots)}
        goals = {f"robot_{i}": {'target': np.array([10.0, float(i), 0.0])} for i in range(self.num_robots)}
        return states, goals

    def vla_policy(self, state, goal):
        # mock VLA policy action proposing straight movement toward goal
        direction = goal['target'] - state['robot_position']
        norm = np.linalg.norm(direction)
        if norm > 0.1:
            return (direction / norm) * 0.5 # constant speed
        return np.zeros(3)

    def run_benchmark(self, num_episodes=1000):
        print(f"Running FleetSafe-VLA-Bench on scenario: {self.scenario}")
        for ep in range(num_episodes):
            states, goals = self.reset_scenario()
            metrics = self._run_episode(states, goals)
            self.results.append(metrics)
        
        return self.aggregate_results()
    
    def _run_episode(self, states, goals):
        violations = 0
        deadline_misses = 0
        stl_robustness = []
        latencies = []
        
        for t in range(100):
            start = time.time()
            
            actions = {f"robot_{i}": self.vla_policy(states[f"robot_{i}"], goals[f"robot_{i}"]) 
                      for i in range(self.num_robots)}
            
            safe_actions = self.coordinator.coordinate_actions(states, actions)
            
            latency = time.time() - start
            latencies.append(latency)
            
            for robot_id, action in safe_actions.items():
                if not np.array_equal(actions[robot_id], action):
                    violations += 1
                
                states[robot_id]['robot_position'] += action * 0.1
        
        return {
            'svr': violations / (100 * self.num_robots),
            'avg_latency': np.mean(latencies)
        }
        
    def aggregate_results(self):
        avg_svr = np.mean([r['svr'] for r in self.results])
        avg_lat = np.mean([r['avg_latency'] for r in self.results])
        return {
            'average_svr': avg_svr,
            'average_latency_ms': avg_lat * 1000.0
        }

if __name__ == "__main__":
    bench = FleetSafeVLABench()
    results = bench.run_benchmark(10)
    print("Benchmark Results:", results)
