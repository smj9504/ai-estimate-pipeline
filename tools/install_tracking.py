#!/usr/bin/env python3
"""
Installation script for token tracking system dependencies
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command):
    """Run command and return success status"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def install_package(package):
    """Install a Python package using pip"""
    print(f"📦 Installing {package}...")
    success, stdout, stderr = run_command(f"pip install {package}")
    
    if success:
        print(f"✅ {package} installed successfully")
        return True
    else:
        print(f"❌ Failed to install {package}: {stderr}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "src/tracking", 
        "src/api",
        "web/templates"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"📁 Creating directory: {directory}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✅ Directory exists: {directory}")

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        "matplotlib",
        "pandas",
        "scipy"
    ]
    
    print("🔍 Checking dependencies...")
    
    failed_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"❌ {package} not found")
            if not install_package(package):
                failed_packages.append(package)
    
    return len(failed_packages) == 0

def update_requirements():
    """Update requirements.txt with new dependencies"""
    requirements_path = Path("requirements.txt")
    
    new_requirements = [
        "matplotlib>=3.5.0",
        "pandas>=1.3.0", 
        "scipy>=1.7.0"
    ]
    
    if requirements_path.exists():
        print("📝 Reading current requirements.txt...")
        with open(requirements_path, 'r') as f:
            current_reqs = f.read().strip()
        
        # Add new requirements if not already present
        updated_reqs = current_reqs
        for req in new_requirements:
            package_name = req.split('>=')[0]
            if package_name not in current_reqs:
                updated_reqs += f"\n{req}"
                print(f"➕ Adding {req} to requirements.txt")
        
        if updated_reqs != current_reqs:
            with open(requirements_path, 'w') as f:
                f.write(updated_reqs + '\n')
            print("✅ Requirements.txt updated")
        else:
            print("✅ Requirements.txt already up to date")
    else:
        print("📝 Creating requirements.txt...")
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(new_requirements) + '\n')
        print("✅ Requirements.txt created")

def test_tracking_system():
    """Test the tracking system installation"""
    print("🧪 Testing tracking system...")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        
        # Test TokenTracker
        from src.tracking.token_tracker import TokenTracker, TokenPricingManager
        print("✅ TokenTracker import successful")
        
        # Test basic functionality
        pricing_manager = TokenPricingManager()
        cost = pricing_manager.calculate_cost("gpt-4o-mini", 1000, 500)
        print(f"✅ Cost calculation test: ${cost:.6f}")
        
        # Test UsageReporter
        from src.tracking.usage_reporter import UsageReporter
        print("✅ UsageReporter import successful")
        
        # Test API endpoints
        from src.api.tracking_endpoints import tracking_router
        print("✅ API endpoints import successful")
        
        print("🎉 All tracking system components working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Tracking system test failed: {e}")
        return False

def show_usage_instructions():
    """Show usage instructions"""
    print("\n" + "="*80)
    print("🎉 TOKEN TRACKING SYSTEM INSTALLATION COMPLETE!")
    print("="*80)
    print()
    print("📚 Usage Instructions:")
    print()
    print("1. 🌐 Web Dashboard:")
    print("   Visit: http://localhost:8000/usage")
    print("   (Start server with: python run_server.py)")
    print()
    print("2. 🖥️  Command Line Interface:")
    print("   python -m src.tracking.cli stats              # Show current stats")
    print("   python -m src.tracking.cli report daily       # Daily report")
    print("   python -m src.tracking.cli pricing            # Show model pricing")
    print("   python -m src.tracking.cli export csv         # Export to CSV")
    print()
    print("3. 🔌 API Endpoints:")
    print("   GET /api/tracking/stats                        # Usage statistics")
    print("   GET /api/tracking/reports/daily                # Daily reports")
    print("   GET /api/tracking/dashboard/summary            # Dashboard data")
    print()
    print("4. 📊 Features:")
    print("   • Real-time token usage tracking")
    print("   • Cost calculation for all AI models")
    print("   • Comprehensive reporting and analytics")
    print("   • Data export (CSV/JSON)")
    print("   • Interactive web dashboard")
    print("   • Command-line tools")
    print()
    print("🚀 The tracking system will automatically monitor all AI model calls!")
    print("="*80)

def main():
    """Main installation function"""
    print("🚀 AI Estimate Pipeline - Token Tracking System Installer")
    print("="*60)
    
    # Step 1: Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Step 2: Check and install dependencies
    print("\n📦 Installing dependencies...")
    if not check_and_install_dependencies():
        print("\n❌ Some dependencies failed to install. Please install them manually:")
        print("pip install matplotlib pandas scipy")
        return False
    
    # Step 3: Update requirements.txt
    print("\n📝 Updating requirements...")
    update_requirements()
    
    # Step 4: Test the system
    print("\n🧪 Testing installation...")
    if not test_tracking_system():
        print("\n❌ Installation test failed. Please check the error messages above.")
        return False
    
    # Step 5: Show usage instructions
    show_usage_instructions()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Installation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed with error: {e}")
        sys.exit(1)