#!/usr/bin/env python3
"""
QUANTUM-RESISTANT CRYPTOGRAPHY FOR SECURE IOT COMMUNICATIONS
FINAL CASE STUDY EXECUTABLE - Complete Benchmarking Suite
"""

import time
import psutil
import os
import pandas as pd
import numpy as np
from oqs import KeyEncapsulation, Signature

class CaseStudyBenchmark:
    def __init__(self):
        self.results = []
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # IoT Device Profiles (Based on real specifications)
        self.iot_profiles = {
            "constrained_iot": {
                "processor": "ARM Cortex-M0+",
                "power_consumption": 0.05,  # 50mW
                "typical_ram": "64-256KB",
                "battery": "Coin cell (100-500mAh)",
                "use_cases": ["Sensors", "Wearables", "Smart Tags"]
            },
            "standard_iot": {
                "processor": "ARM Cortex-M4", 
                "power_consumption": 0.1,    # 100mW
                "typical_ram": "256KB-1MB",
                "battery": "Li-ion (500-2000mAh)",
                "use_cases": ["Smart Home", "Healthcare", "Asset Tracking"]
            },
            "gateway_iot": {
                "processor": "ARM Cortex-A series",
                "power_consumption": 0.5,    # 500mW
                "typical_ram": "1GB+",
                "power_source": "Mains/battery",
                "use_cases": ["Gateways", "Hubs", "Edge Computing"]
            }
        }
        
        # IoT Communication Scenarios
        self.iot_scenarios = {
            "environmental_sensor": {
                "operations_per_day": 1440,  # Every minute
                "data_size": 128,
                "battery_mah": 500,
                "security_criticality": "Medium"
            },
            "health_monitor": {
                "operations_per_day": 2880,  # Every 30 seconds
                "data_size": 256, 
                "battery_mah": 300,
                "security_criticality": "High"
            },
            "smart_lock": {
                "operations_per_day": 50,    # Few operations
                "data_size": 512,
                "battery_mah": 2000,
                "security_criticality": "Very High"
            },
            "industrial_sensor": {
                "operations_per_day": 86400, # Every second
                "data_size": 64,
                "power_source": "Mains",
                "security_criticality": "High"
            }
        }

    def print_header(self):
        """Print professional case study header"""
        print("\n" + "="*70)
        print("QUANTUM-RESISTANT CRYPTOGRAPHY FOR SECURE IOT COMMUNICATIONS")
        print("Case Study Implementation - Complete Benchmarking Suite")
        print("="*70)
        print(f"Execution Timestamp: {self.timestamp}")
        print(f"Platform: {os.uname().sysname} {os.uname().machine}")
        print("="*70)

    def get_memory_usage_kb(self):
        """Get current memory usage in KB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024

    def estimate_energy_consumption(self, operation_time_ms, device_profile):
        """Estimate energy consumption using real IoT power models"""
        power_watts = self.iot_profiles[device_profile]["power_consumption"]
        # Energy (Joules) = Power (Watts) × Time (seconds)
        energy_joules = (operation_time_ms / 1000) * power_watts
        return energy_joules * 1000  # Convert to milliJoules

    def calculate_battery_life(self, energy_mj_per_op, ops_per_day, battery_mah):
        """Calculate battery life impact for IoT devices"""
        # Convert battery capacity to Joules: mAh × 3.7V × 3.6 = Joules
        battery_joules = battery_mah * 3.7 * 3.6
        daily_energy_j = (energy_mj_per_op * ops_per_day) / 1000
        battery_life_days = battery_joules / daily_energy_j if daily_energy_j > 0 else float('inf')
        return battery_life_days

    def benchmark_kyber_complete(self, iterations=100):
        """Complete Kyber benchmarking for all security levels"""
        print("\n KYBER KEY ENCAPSULATION MECHANISM BENCHMARKING")
        print("-" * 50)
        
        kyber_algorithms = ["Kyber512", "Kyber768", "Kyber1024"]
        
        for kyber_alg in kyber_algorithms:
            print(f"\n Testing {kyber_alg}:")
            
            for device_profile in self.iot_profiles.keys():
                print(f"   Device: {device_profile}...")
                
                keygen_times, encap_times, decap_times = [], [], []
                memory_usage = []
                
                for i in range(iterations):
                    # Memory before
                    mem_before = self.get_memory_usage_kb()
                    
                    # Initialize Kyber
                    start_time = time.time()
                    kem = KeyEncapsulation(kyber_alg)
                    key_init_time = (time.time() - start_time) * 1000
                    
                    # Key Generation
                    start_time = time.time()
                    public_key = kem.generate_keypair()
                    keygen_time = (time.time() - start_time) * 1000
                    
                    # Encapsulation
                    start_time = time.time()
                    ciphertext, shared_secret_server = kem.encap_secret(public_key)
                    encap_time = (time.time() - start_time) * 1000
                    
                    # Decapsulation
                    start_time = time.time()
                    shared_secret_client = kem.decap_secret(ciphertext)
                    decap_time = (time.time() - start_time) * 1000
                    
                    # Memory after
                    mem_after = self.get_memory_usage_kb()
                    
                    # Verify cryptographic correctness
                    if shared_secret_server != shared_secret_client:
                        print(f"    ❌ CRYPTOGRAPHIC VERIFICATION FAILED!")
                        return False
                    
                    keygen_times.append(keygen_time)
                    encap_times.append(encap_time)
                    decap_times.append(decap_time)
                    memory_usage.append(mem_after - mem_before)
                    
                    kem.free()
                
                # Calculate statistics
                stats = {
                    'keygen_avg': np.mean(keygen_times),
                    'keygen_std': np.std(keygen_times),
                    'encap_avg': np.mean(encap_times),
                    'encap_std': np.std(encap_times),
                    'memory_avg': np.mean(memory_usage),
                    'energy_keygen': self.estimate_energy_consumption(
                        np.mean(keygen_times), device_profile
                    ),
                    'energy_encap': self.estimate_energy_consumption(
                        np.mean(encap_times), device_profile
                    )
                }
                
                # Store results for key generation
                self.results.append({
                    'algorithm': kyber_alg,
                    'algorithm_type': 'KEM',
                    'device_profile': device_profile,
                    'operation': 'key_generation',
                    'time_avg_ms': stats['keygen_avg'],
                    'time_std_ms': stats['keygen_std'],
                    'memory_kb': stats['memory_avg'],
                    'energy_mj': stats['energy_keygen'],
                    'security_level': kyber_alg.replace('Kyber', '')
                })
                
                # Store results for encapsulation
                self.results.append({
                    'algorithm': kyber_alg,
                    'algorithm_type': 'KEM', 
                    'device_profile': device_profile,
                    'operation': 'encapsulation',
                    'time_avg_ms': stats['encap_avg'],
                    'time_std_ms': stats['encap_std'],
                    'memory_kb': stats['memory_avg'],
                    'energy_mj': stats['energy_encap'],
                    'security_level': kyber_alg.replace('Kyber', '')
                })
                
                print(f"     KeyGen: {stats['keygen_avg']:.2f} ± {stats['keygen_std']:.2f} ms")
                print(f"     Encaps: {stats['encap_avg']:.2f} ± {stats['encap_std']:.2f} ms")
                print(f"     Energy: {stats['energy_keygen']:.2f} mJ")
                print(f"     Memory: {stats['memory_avg']:.1f} KB")
        
        print(" Kyber benchmarking completed successfully!")
        return True

    def benchmark_dilithium_complete(self, iterations=50):
        """Complete Dilithium benchmarking for all security levels"""
        print("\n DILITHIUM DIGITAL SIGNATURE SCHEME BENCHMARKING") 
        print("-" * 50)
        
        dilithium_algorithms = ["Dilithium2", "Dilithium3", "Dilithium5"]
        test_message = b"IoT device secure firmware update v2.5 - authentication required"
        
        for dilithium_alg in dilithium_algorithms:
            print(f"\n Testing {dilithium_alg}:")
            
            for device_profile in self.iot_profiles.keys():
                print(f"   Device: {device_profile}...")
                
                keygen_times, sign_times, verify_times = [], [], []
                memory_usage = []
                
                for i in range(iterations):
                    # Memory before
                    mem_before = self.get_memory_usage_kb()
                    
                    # Key Generation
                    start_time = time.time()
                    signer = Signature(dilithium_alg)
                    public_key = signer.generate_keypair()
                    keygen_time = (time.time() - start_time) * 1000
                    
                    # Signing
                    start_time = time.time()
                    signature = signer.sign(test_message)
                    sign_time = (time.time() - start_time) * 1000
                    
                    # Verification (new instance)
                    start_time = time.time()
                    verifier = Signature(dilithium_alg)
                    is_valid = verifier.verify(test_message, signature, public_key)
                    verify_time = (time.time() - start_time) * 1000
                    
                    # Memory after
                    mem_after = self.get_memory_usage_kb()
                    
                    # Verify cryptographic correctness
                    if not is_valid:
                        print(f"     SIGNATURE VERIFICATION FAILED!")
                        return False
                    
                    keygen_times.append(keygen_time)
                    sign_times.append(sign_time)
                    verify_times.append(verify_time)
                    memory_usage.append(mem_after - mem_before)
                    
                    signer.free()
                    verifier.free()
                
                # Calculate statistics
                stats = {
                    'keygen_avg': np.mean(keygen_times),
                    'keygen_std': np.std(keygen_times),
                    'sign_avg': np.mean(sign_times),
                    'sign_std': np.std(sign_times),
                    'verify_avg': np.mean(verify_times),
                    'verify_std': np.std(verify_times),
                    'memory_avg': np.mean(memory_usage)
                }
                
                # Store results for all operations
                operations = [
                    ('key_generation', stats['keygen_avg'], stats['keygen_std']),
                    ('signing', stats['sign_avg'], stats['sign_std']),
                    ('verification', stats['verify_avg'], stats['verify_std'])
                ]
                
                for op_name, op_time, op_std in operations:
                    self.results.append({
                        'algorithm': dilithium_alg,
                        'algorithm_type': 'Signature',
                        'device_profile': device_profile,
                        'operation': op_name,
                        'time_avg_ms': op_time,
                        'time_std_ms': op_std,
                        'memory_kb': stats['memory_avg'],
                        'energy_mj': self.estimate_energy_consumption(op_time, device_profile),
                        'security_level': dilithium_alg.replace('Dilithium', '')
                    })
                
                print(f"     KeyGen: {stats['keygen_avg']:.2f} ± {stats['keygen_std']:.2f} ms")
                print(f"     Sign:   {stats['sign_avg']:.2f} ± {stats['sign_std']:.2f} ms") 
                print(f"     Verify: {stats['verify_avg']:.2f} ± {stats['verify_std']:.2f} ms")
                print(f"     Memory: {stats['memory_avg']:.1f} KB")
        
        print(" Dilithium benchmarking completed successfully!")
        return True

    def analyze_iot_viability(self):
        """Analyze practical viability for real IoT scenarios"""
        print("\n IOT DEPLOYMENT VIABILITY ANALYSIS")
        print("-" * 50)
        
        viability_results = []
        
        for scenario_name, scenario_params in self.iot_scenarios.items():
            print(f"\n Analyzing {scenario_name.replace('_', ' ').title()}:")
            
            for algorithm in ["Kyber512", "Kyber768", "Dilithium2", "Dilithium3"]:
                # Get performance data for constrained devices
                algo_data = [r for r in self.results if 
                           r['algorithm'] == algorithm and 
                           r['device_profile'] == 'constrained_iot' and
                           r['operation'] == 'key_generation']
                
                if algo_data:
                    energy_per_op = algo_data[0]['energy_mj']
                    time_per_op = algo_data[0]['time_avg_ms']
                    memory_usage = algo_data[0]['memory_kb']
                    
                    # Calculate battery impact if battery-powered
                    if 'battery_mah' in scenario_params:
                        battery_life = self.calculate_battery_life(
                            energy_per_op, scenario_params['operations_per_day'], 
                            scenario_params['battery_mah']
                        )
                        
                        viability = {
                            'scenario': scenario_name,
                            'algorithm': algorithm,
                            'battery_life_days': battery_life,
                            'energy_per_op_mj': energy_per_op,
                            'time_per_op_ms': time_per_op,
                            'memory_kb': memory_usage,
                            'operations_per_day': scenario_params['operations_per_day'],
                            'security_criticality': scenario_params['security_criticality']
                        }
                        
                        viability_results.append(viability)
                        
                        # Print recommendation
                        if battery_life > 365:
                            recommendation = " EXCELLENT - Suitable for long-term deployment"
                        elif battery_life > 30:
                            recommendation = "  ACCEPTABLE - Regular battery replacement needed"
                        else:
                            recommendation = " NOT SUITABLE - Battery life too short"
                        
                        print(f"  {algorithm}:")
                        print(f"    Battery Life: {battery_life:.1f} days - {recommendation}")
                        print(f"    Energy/Op: {energy_per_op:.2f} mJ, Time: {time_per_op:.2f} ms")
        
        return viability_results

    def generate_executive_summary(self):
        """Generate key findings for case study report"""
        print("\n EXECUTIVE SUMMARY - KEY FINDINGS")
        print("=" * 50)
        
        # Find best performing algorithms for constrained IoT
        constrained_results = [r for r in self.results if r['device_profile'] == 'constrained_iot']
        
        if constrained_results:
            # Best KEM for constrained IoT
            kyber_results = [r for r in constrained_results if 'Kyber' in r['algorithm'] and r['operation'] == 'key_generation']
            best_kyber = min(kyber_results, key=lambda x: x['time_avg_ms'])
            
            # Best Signature for constrained IoT
            dilithium_results = [r for r in constrained_results if 'Dilithium' in r['algorithm'] and r['operation'] == 'key_generation']
            best_dilithium = min(dilithium_results, key=lambda x: x['time_avg_ms'])
            
            print(" RECOMMENDED ALGORITHMS FOR CONSTRAINED IOT:")
            print(f"    Key Exchange: {best_kyber['algorithm']}")
            print(f"     - Time: {best_kyber['time_avg_ms']:.2f} ms")
            print(f"     - Energy: {best_kyber['energy_mj']:.2f} mJ")
            print(f"     - Memory: {best_kyber['memory_kb']:.1f} KB")
            
            print(f"    Digital Signature: {best_dilithium['algorithm']}")
            print(f"     - Time: {best_dilithium['time_avg_ms']:.2f} ms") 
            print(f"     - Energy: {best_dilithium['energy_mj']:.2f} mJ")
            print(f"     - Memory: {best_dilithium['memory_kb']:.1f} KB")
        
        # Security vs Performance trade-off
        print("\n SECURITY-PERFORMANCE TRADE-OFF ANALYSIS:")
        security_levels = {}
        for algo in ["Kyber512", "Kyber768", "Kyber1024", "Dilithium2", "Dilithium3", "Dilithium5"]:
            algo_data = [r for r in constrained_results if r['algorithm'] == algo and r['operation'] == 'key_generation']
            if algo_data:
                security_levels[algo] = {
                    'time': algo_data[0]['time_avg_ms'],
                    'security': int(algo.replace('Kyber', '').replace('Dilithium', ''))
                }
                print(f"   {algo}: {algo_data[0]['time_avg_ms']:.2f} ms (Security Level: {security_levels[algo]['security']})")

    def save_all_results(self):
        """Save comprehensive results for case study report"""
        # Create results directory
        os.makedirs('../results', exist_ok=True)
        
        # Save main benchmark results
        df_main = pd.DataFrame(self.results)
        results_file = f'../results/benchmark_results_{self.timestamp}.csv'
        df_main.to_csv(results_file, index=False)
        
        # Save IoT viability analysis
        viability_df = pd.DataFrame(self.analyze_iot_viability())
        viability_file = f'../results/iot_viability_{self.timestamp}.csv'
        viability_df.to_csv(viability_file, index=False)
        
        # Save summary report
        summary_file = f'../results/executive_summary_{self.timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write("QUANTUM-RESISTANT CRYPTOGRAPHY FOR IOT - EXECUTIVE SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total measurements: {len(self.results)}\n")
            f.write(f"Algorithms tested: {df_main['algorithm'].nunique()}\n")
            f.write(f"Device profiles: {df_main['device_profile'].nunique()}\n\n")
            
            # Add key findings
            constrained_results = [r for r in self.results if r['device_profile'] == 'constrained_iot']
            if constrained_results:
                best_kyber = min([r for r in constrained_results if 'Kyber' in r['algorithm']], 
                               key=lambda x: x['time_avg_ms'])
                best_dilithium = min([r for r in constrained_results if 'Dilithium' in r['algorithm']], 
                                   key=lambda x: x['time_avg_ms'])
                
                f.write("KEY RECOMMENDATIONS:\n")
                f.write(f"- Best KEM for constrained IoT: {best_kyber['algorithm']}\n")
                f.write(f"- Best Signature for constrained IoT: {best_dilithium['algorithm']}\n")
                f.write(f"- Typical operation time: {best_kyber['time_avg_ms']:.2f} ms\n")
                f.write(f"- Energy per operation: {best_kyber['energy_mj']:.2f} mJ\n")
        
        print(f"\n RESULTS SAVED:")
        print(f"    Benchmark data: {results_file}")
        print(f"    Viability analysis: {viability_file}")
        print(f"    Executive summary: {summary_file}")
        
        return df_main, viability_df

    def run_complete_analysis(self):
        """Execute complete case study analysis"""
        self.print_header()
        
        print("\n STARTING COMPLETE CASE STUDY ANALYSIS...")
        
        # Run benchmarks
        print("\n1. RUNNING CRYPTOGRAPHIC BENCHMARKS...")
        kyber_success = self.benchmark_kyber_complete(iterations=50)
        dilithium_success = self.benchmark_dilithium_complete(iterations=30)
        
        if not (kyber_success and dilithium_success):
            print(" Benchmarking failed due to cryptographic errors!")
            return False
        
        # Save results
        print("\n2. SAVING AND ANALYZING RESULTS...")
        df_main, viability_df = self.save_all_results()
        
        # Generate summary
        print("\n3. GENERATING EXECUTIVE SUMMARY...")
        self.generate_executive_summary()
        
        print("\n CASE STUDY ANALYSIS COMPLETED SUCCESSFULLY!")
        print(" Use the generated files for your final report.")
        
        return True

def main():
    """Main execution function"""
    try:
        benchmark = CaseStudyBenchmark()
        success = benchmark.run_complete_analysis()
        
        if success:
            print("\n YOUR CASE STUDY IS READY FOR SUBMISSION!")
            print("Next steps:")
            print("1. Run the visualization script: python analysis/results_visualizer.py")
            print("2. Use the generated data and charts in your IEEE paper")
            print("3. Include the executive summary in your presentation")
        else:
            print("\n Analysis failed. Please check the error messages.")
            
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install pandas numpy matplotlib seaborn")
        print("  (OQS library should already be installed)")

if __name__ == "__main__":
    main()