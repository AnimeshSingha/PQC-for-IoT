#!/usr/bin/env python3
"""
CASE STUDY RESULTS VISUALIZATION
Generates professional charts and tables for final report
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

class CaseStudyVisualizer:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Create results directory
        os.makedirs('../results', exist_ok=True)
        
    def load_latest_data(self):
        """Load the most recent benchmark results"""
        try:
            # Find latest benchmark file
            result_files = [f for f in os.listdir('../results') if f.startswith('benchmark_results_')]
            if not result_files:
                print(" No benchmark results found. Please run master_benchmark.py first.")
                return None, None
            
            latest_benchmark = sorted(result_files)[-1]
            benchmark_path = f'../results/{latest_benchmark}'
            
            # Find latest viability file
            viability_files = [f for f in os.listdir('../results') if f.startswith('iot_viability_')]
            latest_viability = sorted(viability_files)[-1] if viability_files else None
            viability_path = f'../results/{latest_viability}' if latest_viability else None
            
            print(f" Loading: {latest_benchmark}")
            df_benchmark = pd.read_csv(benchmark_path)
            
            if viability_path:
                print(f" Loading: {latest_viability}")
                df_viability = pd.read_csv(viability_path)
            else:
                df_viability = None
                
            return df_benchmark, df_viability
            
        except Exception as e:
            print(f" Error loading data: {e}")
            return None, None

    def create_performance_comparison_chart(self, df):
        """Create comprehensive performance comparison charts"""
        print(" Generating Performance Comparison Charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Post-Quantum Cryptography Performance Analysis for IoT', fontsize=16, fontweight='bold')
        
        # Filter for constrained IoT devices (most relevant)
        constrained_data = df[df['device_profile'] == 'constrained_iot'].copy()
        
        # Chart 1: Kyber Performance Comparison
        kyber_data = constrained_data[constrained_data['algorithm_type'] == 'KEM'].copy()
        kyber_keygen = kyber_data[kyber_data['operation'] == 'key_generation'].copy()
        
        if not kyber_keygen.empty:
            algorithms = kyber_keygen['algorithm'].unique()
            times = [kyber_keygen[kyber_keygen['algorithm'] == algo]['time_avg_ms'].values[0] for algo in algorithms]
            energies = [kyber_keygen[kyber_keygen['algorithm'] == algo]['energy_mj'].values[0] for algo in algorithms]
            
            bars1 = axes[0,0].bar(algorithms, times, color='lightblue', alpha=0.7, label='Time (ms)')
            axes[0,0].set_title('Kyber Key Generation Performance\n(Constrained IoT Devices)', fontweight='bold')
            axes[0,0].set_ylabel('Time (milliseconds)')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                axes[0,0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.1f}ms', ha='center', va='bottom', fontweight='bold')
            
            # Energy consumption (secondary axis)
            ax2 = axes[0,0].twinx()
            line = ax2.plot(algorithms, energies, 'ro-', linewidth=2, markersize=8, label='Energy (mJ)')
            ax2.set_ylabel('Energy Consumption (mJ)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Combined legend
            lines1, labels1 = axes[0,0].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes[0,0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Chart 2: Dilithium Performance Comparison
        dilithium_data = constrained_data[constrained_data['algorithm_type'] == 'Signature'].copy()
        dilithium_keygen = dilithium_data[dilithium_data['operation'] == 'key_generation'].copy()
        
        if not dilithium_keygen.empty:
            algorithms = dilithium_keygen['algorithm'].unique()
            times = [dilithium_keygen[dilithium_keygen['algorithm'] == algo]['time_avg_ms'].values[0] for algo in algorithms]
            energies = [dilithium_keygen[dilithium_keygen['algorithm'] == algo]['energy_mj'].values[0] for algo in algorithms]
            
            bars2 = axes[0,1].bar(algorithms, times, color='lightcoral', alpha=0.7)
            axes[0,1].set_title('Dilithium Key Generation Performance\n(Constrained IoT Devices)', fontweight='bold')
            axes[0,1].set_ylabel('Time (milliseconds)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.1f}ms', ha='center', va='bottom', fontweight='bold')
            
            # Energy consumption (secondary axis)
            ax2 = axes[0,1].twinx()
            ax2.plot(algorithms, energies, 'go-', linewidth=2, markersize=8, label='Energy (mJ)')
            ax2.set_ylabel('Energy Consumption (mJ)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')

        # Chart 3: Memory Usage Comparison
        memory_data = constrained_data[constrained_data['operation'] == 'key_generation'].copy()
        
        if not memory_data.empty:
            algorithms = memory_data['algorithm'].unique()
            memory_usage = [memory_data[memory_data['algorithm'] == algo]['memory_kb'].mean() for algo in algorithms]
            
            bars3 = axes[1,0].bar(algorithms, memory_usage, color='lightgreen', alpha=0.7)
            axes[1,0].set_title('Memory Usage for Key Generation\n(Constrained IoT Devices)', fontweight='bold')
            axes[1,0].set_ylabel('Memory (KB)')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars3:
                height = bar.get_height()
                axes[1,0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.1f}KB', ha='center', va='bottom', fontweight='bold')

        # Chart 4: Security Level vs Performance Trade-off - FIXED WARNING
        security_data = constrained_data[constrained_data['operation'] == 'key_generation'].copy()
        
        if not security_data.empty:
            # Map security levels to numerical values - FIXED: Use .loc to avoid SettingWithCopyWarning
            security_map = {'512': 1, '768': 2, '1024': 3, '2': 1, '3': 2, '5': 3}
            security_data.loc[:, 'security_numeric'] = security_data['security_level'].map(security_map)
            
            # Create scatter plot
            for algo_type in security_data['algorithm_type'].unique():
                type_data = security_data[security_data['algorithm_type'] == algo_type].copy()
                scatter = axes[1,1].scatter(
                    type_data['security_numeric'], 
                    type_data['time_avg_ms'],
                    s=type_data['memory_kb'] * 2,  # Size represents memory usage
                    alpha=0.6,
                    label=algo_type
                )
                
                # Add algorithm labels
                for idx, row in type_data.iterrows():
                    axes[1,1].annotate(row['algorithm'], 
                                      (row['security_numeric'], row['time_avg_ms']),
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=8, alpha=0.8)
            
            axes[1,1].set_title('Security-Performance Trade-off Analysis\n(Size = Memory Usage)', fontweight='bold')
            axes[1,1].set_xlabel('Security Level (1=Lowest, 3=Highest)')
            axes[1,1].set_ylabel('Time (ms)')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'../results/performance_comparison_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        print(f" Saved: performance_comparison_{self.timestamp}.png")

    def create_iot_viability_chart(self, df_viability):
        """Create IoT deployment viability charts"""
        if df_viability is None or df_viability.empty:
            print(" No viability data available. Skipping viability charts.")
            return
            
        print(" Generating IoT Viability Charts...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('IoT Deployment Viability Analysis', fontsize=16, fontweight='bold')
        
        # Chart 1: Battery Life Analysis
        scenarios = df_viability['scenario'].unique()
        x = np.arange(len(scenarios))
        width = 0.2
        
        algorithms = df_viability['algorithm'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        
        for i, (algo, color) in enumerate(zip(algorithms, colors)):
            algo_data = df_viability[df_viability['algorithm'] == algo].copy()
            battery_life = []
            
            for scenario in scenarios:
                scenario_data = algo_data[algo_data['scenario'] == scenario].copy()
                if not scenario_data.empty:
                    battery_life.append(scenario_data['battery_life_days'].values[0])
                else:
                    battery_life.append(0)
            
            bars = ax1.bar(x + i * width, battery_life, width, label=algo, color=color, alpha=0.8)
            
            # Add value labels
            for bar, value in zip(bars, battery_life):
                if value > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{value:.0f}d', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax1.set_title('Battery Life Impact on IoT Scenarios', fontweight='bold')
        ax1.set_xlabel('IoT Scenario')
        ax1.set_ylabel('Battery Life (Days)')
        ax1.set_xticks(x + width * (len(algorithms)-1)/2)
        ax1.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add horizontal line for 1-year battery life
        ax1.axhline(y=365, color='red', linestyle='--', alpha=0.7, label='1 Year Target')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Chart 2: Energy Consumption per Operation
        energy_data = df_viability.groupby('algorithm')['energy_per_op_mj'].mean().sort_values()
        
        bars = ax2.bar(energy_data.index, energy_data.values, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(energy_data))),
                      alpha=0.7)
        
        ax2.set_title('Average Energy Consumption per Operation', fontweight='bold')
        ax2.set_ylabel('Energy (mJ)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}mJ', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'../results/iot_viability_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        print(f" Saved: iot_viability_{self.timestamp}.png")

    def create_device_comparison_chart(self, df):
        """Compare performance across different device types"""
        print(" Generating Device Comparison Chart...")
        
        # Focus on key generation operation
        keygen_data = df[df['operation'] == 'key_generation'].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Across IoT Device Types', fontsize=16, fontweight='bold')
        
        # Select key algorithms for comparison
        key_algorithms = ['Kyber512', 'Kyber768', 'Dilithium2', 'Dilithium3']
        
        for i, algorithm in enumerate(key_algorithms):
            ax = axes[i//2, i%2]
            algo_data = keygen_data[keygen_data['algorithm'] == algorithm].copy()
            
            if not algo_data.empty:
                device_profiles = algo_data['device_profile'].unique()
                times = [algo_data[algo_data['device_profile'] == device]['time_avg_ms'].values[0] 
                        for device in device_profiles]
                energies = [algo_data[algo_data['device_profile'] == device]['energy_mj'].values[0] 
                           for device in device_profiles]
                
                x = np.arange(len(device_profiles))
                width = 0.35
                
                # Time bars
                bars1 = ax.bar(x - width/2, times, width, label='Time (ms)', 
                              color='skyblue', alpha=0.7)
                # Energy bars
                bars2 = ax.bar(x + width/2, energies, width, label='Energy (mJ)', 
                              color='lightcoral', alpha=0.7)
                
                ax.set_title(f'{algorithm} Performance', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([p.replace('_', '\n').title() for p in device_profiles])
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}', ha='center', va='bottom', 
                               fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'../results/device_comparison_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        print(f" Saved: device_comparison_{self.timestamp}.png")

    def generate_summary_tables(self, df, df_viability):
        """Generate summary tables for the report"""
        print(" Generating Summary Tables...")
        
        # Table 1: Performance Summary for Constrained IoT
        constrained_data = df[df['device_profile'] == 'constrained_iot'].copy()
        keygen_data = constrained_data[constrained_data['operation'] == 'key_generation'].copy()
        
        summary_data = []
        for algo in keygen_data['algorithm'].unique():
            algo_data = keygen_data[keygen_data['algorithm'] == algo].copy()
            if not algo_data.empty:
                row = algo_data.iloc[0]
                summary_data.append({
                    'Algorithm': row['algorithm'],
                    'Type': row['algorithm_type'],
                    'Security Level': row['security_level'],
                    'Time (ms)': f"{row['time_avg_ms']:.2f}",
                    'Energy (mJ)': f"{row['energy_mj']:.2f}",
                    'Memory (KB)': f"{row['memory_kb']:.1f}"
                })
        
        # Create summary table
        summary_df = pd.DataFrame(summary_data)
        summary_table = summary_df.to_string(index=False)
        
        # Table 2: IoT Viability Recommendations
        if df_viability is not None and not df_viability.empty:
            viability_summary = []
            for scenario in df_viability['scenario'].unique():
                scenario_data = df_viability[df_viability['scenario'] == scenario].copy()
                best_algo = scenario_data.loc[scenario_data['battery_life_days'].idxmax()]
                
                viability_summary.append({
                    'Scenario': scenario.replace('_', ' ').title(),
                    'Recommended Algorithm': best_algo['algorithm'],
                    'Battery Life (days)': f"{best_algo['battery_life_days']:.0f}",
                    'Security Criticality': best_algo['security_criticality']
                })
            
            viability_df = pd.DataFrame(viability_summary)
            viability_table = viability_df.to_string(index=False)
        else:
            viability_table = "No viability data available"
        
        # Save tables to file
        tables_file = f'../results/summary_tables_{self.timestamp}.txt'
        with open(tables_file, 'w') as f:
            f.write("QUANTUM-RESISTANT CRYPTOGRAPHY FOR IOT - SUMMARY TABLES\n")
            f.write("="*70 + "\n\n")
            
            f.write("TABLE 1: PERFORMANCE SUMMARY (Constrained IoT Devices)\n")
            f.write("-" * 70 + "\n")
            f.write(summary_table + "\n\n")
            
            f.write("TABLE 2: IOT DEPLOYMENT RECOMMENDATIONS\n")
            f.write("-" * 70 + "\n")
            f.write(viability_table + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f" Saved: summary_tables_{self.timestamp}.txt")
        
        return summary_df

    def run_complete_visualization(self):
        """Execute complete visualization pipeline"""
        print(" STARTING CASE STUDY VISUALIZATION...")
        print("="*60)
        
        # Load data
        df_benchmark, df_viability = self.load_latest_data()
        if df_benchmark is None:
            return False
        
        print(f" Loaded {len(df_benchmark)} benchmark measurements")
        if df_viability is not None:
            print(f" Loaded {len(df_viability)} viability assessments")
        
        # Generate all visualizations
        self.create_performance_comparison_chart(df_benchmark)
        self.create_iot_viability_chart(df_viability)
        self.create_device_comparison_chart(df_benchmark)
        self.generate_summary_tables(df_benchmark, df_viability)
        
        print("\n VISUALIZATION COMPLETED SUCCESSFULLY!")
        print(" Generated Charts:")
        print("   - Performance comparison (4 comprehensive charts)")
        print("   - IoT viability analysis (battery life impact)") 
        print("   - Device type comparison (across IoT categories)")
        print("   - Summary tables (for report inclusion)")
        
        print("\n Your case study now has all visual materials needed!")
        return True

def main():
    """Main execution function"""
    try:
        visualizer = CaseStudyVisualizer()
        success = visualizer.run_complete_visualization()
        
        if success:
            print("\n ALL VISUALIZATION MATERIALS READY!")
            print("Next steps for your final submission:")
            print("1. Include the generated charts in your IEEE paper")
            print("2. Use the summary tables in your results section")
            print("3. Reference the findings in your conclusion")
            print("4. Prepare your presentation using these visuals")
        else:
            print("\n Visualization failed. Please run master_benchmark.py first.")
            
    except Exception as e:
        print(f"\n Visualization error: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install matplotlib seaborn pandas numpy")

if __name__ == "__main__":
    main()