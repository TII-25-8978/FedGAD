import sys
import os
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_check(name, status, message=""):
    """Print check result."""
    symbol = "✓" if status else "✗"
    status_text = "PASS" if status else "FAIL"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}[{symbol}] {name}: {status_text}{reset}")
    if message:
        print(f"    {message}")


def check_python_version():
    """Check Python version."""
    print_header("Python Version Check")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    is_valid = version.major == 3 and version.minor >= 8
    
    print_check(
        "Python Version",
        is_valid,
        f"Found: Python {version_str} (Required: Python 3.8+)"
    )
    
    return is_valid


def check_required_packages():
    """Check if required packages are installed."""
    print_header("Required Packages Check")
    
    packages = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "sklearn": "Scikit-learn",
        "pandas": "Pandas",
    }
    
    all_installed = True
    
    for module, name in packages.items():
        try:
            if module == "sklearn":
                import sklearn
                version = sklearn.__version__
            else:
                mod = __import__(module)
                version = mod.__version__
            
            print_check(name, True, f"Version: {version}")
        except ImportError:
            print_check(name, False, "Not installed")
            all_installed = False
    
    return all_installed


def check_pytorch_cuda():
    """Check PyTorch CUDA configuration."""
    print_header("PyTorch CUDA Check")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print_check("CUDA Available", cuda_available)
        
        if cuda_available:
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"    CUDA Version: {cuda_version}")
            print(f"    Device Count: {device_count}")
            print(f"    Device Name: {device_name}")
            print(f"    GPU Memory: {memory:.2f} GB")
        else:
            print("    Running on CPU (GPU not available or not configured)")
        
        return True
    except Exception as e:
        print_check("PyTorch CUDA", False, str(e))
        return False


def check_project_structure():
    """Check if required files exist."""
    print_header("Project Structure Check")
    
    required_files = [
        "main.py",
        "requirements.txt",
        "README.md",
    ]
    
    optional_files = [
        "dataloader.py",
        "model.py",
        "regularizers.py",
        "config.py",
    ]
    
    required_dirs = [
        "datasets",
    ]
    
    all_present = True
    
    print("\nRequired Files:")
    for file in required_files:
        exists = Path(file).exists()
        print_check(file, exists)
        if not exists:
            all_present = False
    
    print("\nOptional Files (needed for training):")
    for file in optional_files:
        exists = Path(file).exists()
        print_check(file, exists)
    
    print("\nRequired Directories:")
    for dir in required_dirs:
        exists = Path(dir).exists()
        print_check(dir, exists, f"Path: {os.path.abspath(dir)}")
        if not exists:
            all_present = False
    
    return all_present


def check_datasets():
    """Check if datasets are present."""
    print_header("Dataset Check")
    
    dataset_dir = Path("datasets")
    
    if not dataset_dir.exists():
        print_check("Dataset Directory", False, "Directory not found")
        return False
    
    datasets = {
        "ToN_IoT.csv": "ToN-IoT Dataset",
        "CSE_CIC_IDS.csv": "CSE-CIC-IDS2018 Dataset",
    }
    
    found_any = False
    
    for filename, name in datasets.items():
        filepath = dataset_dir / filename
        exists = filepath.exists()
        
        if exists:
            size = filepath.stat().st_size / (1024**2)  # MB
            print_check(name, True, f"Size: {size:.2f} MB")
            found_any = True
        else:
            print_check(name, False, "Not found")
    
    if not found_any:
        print("\n    ⚠ No datasets found. Please download and place datasets in datasets/ directory.")
    
    return found_any


def check_code_issues():
    """Check for known code issues."""
    print_header("Code Issues Check")
    
    try:
        with open("main.py", "r") as f:
            content = f.read()
        
        # Check for ablation_mode parameter
        has_ablation_param = "ablation_mode=" in content and "def train_federated_model" in content
        print_check(
            "ablation_mode parameter",
            has_ablation_param,
            "Missing parameter" if not has_ablation_param else "Parameter defined"
        )
        
        return has_ablation_param
    except FileNotFoundError:
        print_check("main.py", False, "File not found")
        return False


def test_basic_functionality():
    """Test basic PyTorch functionality."""
    print_header("Basic Functionality Test")
    
    try:
        import torch
        import torch.nn as nn
        
        # Test tensor creation
        x = torch.randn(10, 5)
        print_check("Tensor Creation", True)
        
        # Test basic operations
        y = x * 2
        print_check("Tensor Operations", True)
        
        # Test neural network
        model = nn.Linear(5, 2)
        output = model(x)
        print_check("Neural Network", True, f"Output shape: {output.shape}")
        
        # Test GPU if available
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            model_gpu = model.cuda()
            output_gpu = model_gpu(x_gpu)
            print_check("GPU Operations", True, "GPU execution successful")
        
        return True
    except Exception as e:
        print_check("Basic Functionality", False, str(e))
        return False


def print_summary(checks):
    """Print summary of all checks."""
    print_header("Summary")
    
    total = len(checks)
    passed = sum(checks.values())
    failed = total - passed
    
    print(f"\nTotal Checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ All checks passed! Your installation is ready.")
        print("\nNext steps:")
        print("  1. Download datasets to datasets/ directory")
        print("  2. Run: python main.py")
    else:
        print("\n✗ Some checks failed. Please review the output above.")
        print("\nSuggested actions:")
        if not checks.get("Python Version"):
            print("  - Install Python 3.8 or higher")
        if not checks.get("Required Packages"):
            print("  - Run: pip install -r requirements.txt")
        if not checks.get("Project Structure"):
            print("  - Ensure all required files are present")
        if not checks.get("Datasets"):
            print("  - Download and place datasets in datasets/ directory")
        if not checks.get("Code Issues"):
            print("  - Fix code issues in main.py (see QUICKSTART.md)")


def main():
    """Run all verification checks."""
    print("=" * 70)
    print(" FedGAD Installation Verification")
    print("=" * 70)
    print("\nThis script will verify your installation and configuration.")
    
    checks = {
        "Python Version": check_python_version(),
        "Required Packages": check_required_packages(),
        "PyTorch CUDA": check_pytorch_cuda(),
        "Project Structure": check_project_structure(),
        "Datasets": check_datasets(),
        "Code Issues": check_code_issues(),
        "Basic Functionality": test_basic_functionality(),
    }
    
    print_summary(checks)
    
    print("\n" + "=" * 70)
    print(" Verification Complete")
    print("=" * 70 + "\n")
    
    # Return exit code
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)