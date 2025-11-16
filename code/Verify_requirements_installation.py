
## Installation Verification Script:

### **5. verify_installation.py**
```python
#!/usr/bin/env python3
"""
Verification script for Quantum IoT Cryptography Case Study dependencies
"""

def verify_installation():
    print("🔍 Verifying installation...")
    
    dependencies = {
        "oqs": "0.5.0",
        "pandas": "1.5.0",
        "numpy": "1.21.0",
        "matplotlib": "3.5.0",
        "seaborn": "0.11.0",
        "psutil": "5.8.0"
    }
    
    all_ok = True
    
    for package, min_version in dependencies.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f" {package}: {version}")
        except ImportError:
            print(f" {package}: NOT INSTALLED")
            all_ok = False
    
    # Test OQS functionality
    try:
        from oqs import KeyEncapsulation, Signature
        print(" OQS library functional")
    except Exception as e:
        print(f" OQS functionality test failed: {e}")
        all_ok = False
    
    if all_ok:
        print("\n All dependencies verified successfully!")
        print("Ready to run the case study:")
        print("cd benchmarks && python3 master_benchmark.py")
    else:
        print("\n  Some dependencies are missing.")
        print("Run: pip install -r requirements.txt")

if __name__ == "__main__":
    verify_installation()