## Case Study Execution Guide
Quantum-Resistant Cryptography for Secure IoT Communications

📋 Table of Contents

    1. Quick Start

    2. Project Overview

    3. Prerequisites

    4. Step-by-Step Execution

    5. Generated Output Files

    6. Results Interpretation

    7. Troubleshooting

    8. Academic Integration

## Quick Start
For Immediate Execution:

# 1. Navigate to project directory
cd quantum_iot_case_study

# 2. Run main benchmark (5-10 minutes)
cd benchmarks
python3 master_benchmark.py

# 3. Generate visualizations (1-2 minutes)
cd ../analysis
python3 results_visualizer.py

# 4. View results
cd ../results
ls -la

Total Time: 6-12 minutes
Output: Complete case study data and visualizations

## Project Overview
Case Study Title:

"Quantum-Resistant Cryptography for Secure IoT Communications"
Objective:

Evaluate post-quantum lattice-based schemes (Kyber, Dilithium) for IoT security with energy-latency benchmarking.
What This Implementation Provides:

    ** Real cryptographic performance measurements

    ** IoT energy consumption modeling

    ** Battery life impact analysis

    ** Security-performance trade-off evaluation

    ** Professional academic charts and tables

    ** Executive summary with recommendations

## Prerequisites
Required Software:

    1. WSL2 + Ubuntu (already set up)

    2. Python 3.8+ (already available)

    3. Required Python packages

## Install Dependencies:

# Activate your virtual environment (if using)
source myenv/bin/activate

# Install required packages
pip install pandas numpy matplotlib seaborn psutil

# Verify OQS library is installed
python3 -c "from oqs import KeyEncapsulation; print('OQS library OK')"

Expected Output:
text

OQS library OK


📁 Project Structure
text

quantum_iot_case_study/
├── 📊 benchmarks/
│   ├── master_benchmark.py          # MAIN EXECUTABLE
│   └── methodology_notes.py         # Research methodology
├── 📈 analysis/
│   └── results_visualizer.py        # Charts & analysis
├── 📋 results/                      # Auto-generated output
├── 📝 documentation/
│   └── execution_guide.md           # This file
└── 📚 paper/                        # Your IEEE paper

## Step-by-Step Execution
Step 1: Run Main Benchmark
bash

# Navigate to benchmarks directory
cd quantum_iot_case_study/benchmarks

# Execute main benchmark
python3 master_benchmark.py

What Happens During Execution:

    System Check - Verifies environment and dependencies

    Kyber Benchmarking - Tests Kyber512, Kyber768, Kyber1024

    Dilithium Benchmarking - Tests Dilithium2, Dilithium3, Dilithium5

    IoT Analysis - Calculates battery life impact

    Results Saving - Generates CSV files and summary

Expected Output Preview:
text

 STARTING COMPLETE CASE STUDY ANALYSIS...
 KYBER KEY ENCAPSULATION MECHANISM BENCHMARKING
 Testing Kyber512:
   Device: constrained_iot...
    ✅ KeyGen: 2.34 ± 0.15 ms
    ✅ Encaps: 1.89 ± 0.12 ms
    ✅ Energy: 0.12 mJ
    ✅ Memory: 15.2 KB
...
✅ CASE STUDY ANALYSIS COMPLETED SUCCESSFULLY!

Estimated Time: 5-10 minutes
Step 2: Generate Visualizations
bash

# Navigate to analysis directory
cd ../analysis

# Generate charts and tables
python3 results_visualizer.py

What Happens During Execution:

    Data Loading - Loads latest benchmark results

    Chart Generation - Creates 3 comprehensive charts

    Table Creation - Generates summary tables

    File Organization - Saves all visual materials

Expected Output Preview:
text

🚀 STARTING CASE STUDY VISUALIZATION...
* Generating Performance Comparison Charts...
* Saved: performance_comparison_20241115_143022.png
* Generating IoT Viability Charts...
* Saved: iot_viability_20241115_143022.png
* Generating Summary Tables...
* VISUALIZATION COMPLETED SUCCESSFULLY!

Estimated Time: 1-2 minutes
Step 3: View Methodology Documentation (Optional)

# Navigate to documentation
cd ../documentation

# View methodology notes
python3 methodology_notes.py

## Generated Output Files

After successful execution, check your results/ directory:
Data Files:

    benchmark_results_YYYYMMDD_HHMMSS.csv - Raw performance measurements

    iot_viability_YYYYMMDD_HHMMSS.csv - Battery life analysis

    executive_summary_YYYYMMDD_HHMMSS.txt - Key findings and recommendations

Visualization Charts:

    performance_comparison_YYYYMMDD_HHMMSS.png - 4-panel performance analysis

    iot_viability_YYYYMMDD_HHMMSS.png - Battery life and energy consumption

    device_comparison_YYYYMMDD_HHMMSS.png - Cross-device performance

Report Materials:

    summary_tables_YYYYMMDD_HHMMSS.txt - Formatted tables for paper inclusion

## Verify Output:


cd ../results
ls -la
# Should see 6+ files with today's timestamp

## Results Interpretation
Key Metrics to Understand:
1. Performance Metrics:

    Time (ms): Operation execution time (lower = better)

    Energy (mJ): Estimated energy consumption (lower = better)

    Memory (KB): RAM usage during operations (lower = better)

2. IoT Viability:

    Battery Life (days): Estimated device lifetime

    >365 days: Excellent for long-term deployment

    30-365 days: Acceptable with regular maintenance

    <30 days: Not suitable for battery-powered devices

3. Security Levels:

    Kyber512 / Dilithium2: Level 1 security

    Kyber768 / Dilithium3: Level 2 security

    Kyber1024 / Dilithium5: Level 3 security

Sample Findings You Might See:
text

## RECOMMENDED ALGORITHMS FOR CONSTRAINED IOT:
   * Key Exchange: Kyber512
     - Time: 2.34 ms
     - Energy: 0.12 mJ
     - Memory: 15.2 KB
   
   * Digital Signature: Dilithium2
     - Time: 5.67 ms
     - Energy: 0.28 mJ
     - Memory: 32.1 KB

## Troubleshooting
Common Issues and Solutions:
Issue 1: "Module not found" errors


# Solution: Install missing packages
pip install pandas numpy matplotlib seaborn psutil

Issue 2: OQS library not found
bash

# Solution: Verify OQS installation
python3 -c "from oqs import KeyEncapsulation; print('OK')"
# If fails, reinstall OQS from your liboqs-python-main directory

Issue 3: Permission denied

# Solution: Ensure proper file permissions
chmod +x benchmarks/master_benchmark.py
chmod +x analysis/results_visualizer.py

Issue 4: Memory errors

# Solution: Reduce iteration count
# Edit master_benchmark.py, change iterations=50 to iterations=20

Issue 5: No results generated

# Solution: Check execution order
# Must run master_benchmark.py BEFORE results_visualizer.py

Verification Checklist:

    Python 3.8+ installed

    All required packages installed

    OQS library working

    Sufficient disk space (>100MB free)

    Running in correct directory

Academic Integration
For Your IEEE Paper:
Methodology Section:
latex

\section{Methodology}
This study employs a simulation-based benchmarking approach to evaluate 
post-quantum cryptographic schemes for IoT environments. Performance 
metrics were measured on standard computing hardware and extrapolated 
to IoT scenarios using established power consumption models...

Results Section - Include:

    Performance comparison charts from performance_comparison_*.png

    IoT viability analysis from iot_viability_*.png

    Summary tables from summary_tables_*.txt

    Executive summary key findings

Recommended Paper Structure:

    Abstract - Summary of findings

    Introduction - IoT security + quantum threat

    Related Work - Literature review

    Methodology - Your benchmarking approach

    Results - Charts and tables from this implementation

    Conclusion - Recommendations based on executive summary

For Your Presentation:
Slide Content:

    Slide 1: Problem statement + case study objective

    Slide 2: Methodology overview

    Slide 3: Performance comparison charts

    Slide 4: IoT viability analysis

    Slide 5: Key recommendations

    Slide 6: Conclusion and future work

Demo Script:

"During our case study implementation, we benchmarked NIST-selected post-quantum algorithms and found that Kyber512 and Dilithium2 provide the best balance of security and performance for constrained IoT devices, with battery life exceeding one year in typical deployment scenarios."

⏱️ Execution Timeline
Estimated Time Breakdown:

    Environment Setup: 0 minutes (already done)

    Benchmark Execution: 5-10 minutes

    Visualization Generation: 1-2 minutes

    Results Review: 2-3 minutes

    Total: 8-15 minutes

Quick Status Check:
bash

# Run this to verify everything is ready
cd quantum_iot_case_study/benchmarks
python3 -c "
try:
    from oqs import KeyEncapsulation, Signature
    import pandas as pd
    import matplotlib.pyplot as plt
    print(' All dependencies OK')
    print(' Ready for case study execution')
except ImportError as e:
    print(f' Missing dependency: {e}')
"

## Final Submission Checklist
Before Submission:

    Successfully executed master_benchmark.py

    Successfully executed results_visualizer.py

    Reviewed all generated files in results/ directory

    Included charts in your IEEE paper

    Referenced findings in your conclusion

    Prepared presentation using generated materials

Files to Include in Submission:

    IEEE format paper (4-6 pages)

    Presentation slides (15 minutes)

    Generated charts and tables

    Executive summary

    This execution guide

💡 Pro Tips for Demonstration
Live Demo Commands:


# Show real-time execution
cd benchmarks && python3 master_benchmark.py

# Quick results showcase
cd ../results && ls -la

# Display generated charts
eog performance_comparison_*.png  # On Linux
# OR open on Windows: start performance_comparison_*.png

Key Talking Points:

    "We implemented real NIST-standardized post-quantum cryptography"

    "Our energy-latency benchmarking provides practical IoT deployment guidance"

    "The results show clear trade-offs between security levels and resource consumption"

    "Kyber512 and Dilithium2 offer the best balance for constrained IoT devices"

Need Help?
Immediate Support:

    Check troubleshooting section above

    Verify all prerequisites are installed

    Ensure correct execution order: Benchmark → Visualize

    Check file permissions and directory structure

Common Success Indicators:

    ✅ "CASE STUDY ANALYSIS COMPLETED SUCCESSFULLY!" message

    ✅ 6+ files generated in results/ directory

    ✅ Charts display properly when opened

    ✅ Executive summary contains specific recommendations

 You're Ready!

Congratulations! You now have a complete, executable case study implementation that provides:

    ✅ Real post-quantum cryptography benchmarking

    ✅ IoT-specific energy-latency analysis

    ✅ Professional academic visualizations

    ✅ Ready-to-use materials for your paper and presentation

Execute the code and you'll have everything needed for your final submission tomorrow! 🚀

Last Updated: November 2025
Case Study: Quantum-Resistant Cryptography for Secure IoT Communications
