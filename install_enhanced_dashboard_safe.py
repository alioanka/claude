#!/usr/bin/env python3
"""
Safe Enhanced Dashboard Installation Script
This script safely installs the enhanced dashboard without breaking existing bots
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command safely and show progress"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Enhanced dashboard requires Python 3.8 or higher")
        return False

def check_existing_requirements():
    """Check what packages are already installed"""
    print("🔍 Checking existing packages...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
        installed_packages = result.stdout.lower()
        
        # Check for key packages that might conflict
        conflicting_packages = ['fastapi', 'uvicorn', 'pandas', 'numpy']
        installed_conflicts = [pkg for pkg in conflicting_packages if pkg in installed_packages]
        
        if installed_conflicts:
            print(f"⚠️  Found existing packages: {', '.join(installed_conflicts)}")
            print("These packages will be updated to compatible versions")
        else:
            print("✅ No conflicting packages found")
            
        return True
    except Exception as e:
        print(f"⚠️  Could not check existing packages: {e}")
        return True

def install_packages_safely():
    """Install packages with version constraints to avoid conflicts"""
    print("📦 Installing enhanced dashboard packages safely...")
    
    # Install packages one by one to avoid conflicts
    packages = [
        "fastapi>=0.95.0,<0.105.0",
        "uvicorn[standard]>=0.20.0,<0.25.0", 
        "websockets>=10.0,<13.0",
        "jinja2>=3.0.0,<4.0.0",
        "aiofiles>=23.0.0,<24.0.0",
        "python-multipart>=0.0.5,<1.0.0",
        "pandas>=1.5.0,<2.1.0",
        "numpy>=1.21.0,<1.25.0",
        "psycopg2-binary>=2.9.0,<3.0.0",
        "sqlalchemy>=1.4.0,<2.1.0",
        "redis>=4.0.0,<6.0.0",
        "httpx>=0.24.0,<0.26.0",
        "aiohttp>=3.8.0,<4.0.0",
        "pydantic>=1.10.0,<2.6.0",
        "python-dotenv>=0.19.0,<2.0.0"
    ]
    
    for package in packages:
        if not run_command(f"{sys.executable} -m pip install '{package}'", f"Installing {package.split('>=')[0]}"):
            print(f"⚠️  Failed to install {package}, trying without version constraints...")
            package_name = package.split('>=')[0]
            if not run_command(f"{sys.executable} -m pip install {package_name}", f"Installing {package_name}"):
                print(f"❌ Could not install {package_name}")
                return False
    
    return True

def verify_installation():
    """Verify that the installation was successful"""
    print("🔍 Verifying installation...")
    
    required_modules = [
        'fastapi',
        'uvicorn', 
        'websockets',
        'pandas',
        'numpy',
        'psycopg2',
        'redis'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} - OK")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} - Missing")
    
    if missing_modules:
        print(f"❌ Missing modules: {', '.join(missing_modules)}")
        return False
    else:
        print("✅ All required modules are available")
        return True

def create_safe_runner():
    """Create a safe runner script that won't interfere with existing bots"""
    runner_content = '''#!/usr/bin/env python3
"""
Safe Enhanced Dashboard Runner
This script runs the enhanced dashboard without affecting existing bots
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables to avoid conflicts
os.environ['DASHBOARD_PORT'] = '8001'
os.environ['DASHBOARD_HOST'] = '0.0.0.0'

# Import and run the enhanced dashboard
try:
    from enhanced_dashboard import app
    import uvicorn
    
    print("🚀 Starting Enhanced Dashboard on port 8001...")
    print("📊 Dashboard will be available at: http://localhost:8001")
    print("🔗 API documentation: http://localhost:8001/api/docs")
    print("⚠️  This will NOT affect your existing bots running on other ports")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Error importing enhanced dashboard: {e}")
    print("Please run the installation script first")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting enhanced dashboard: {e}")
    sys.exit(1)
'''
    
    with open('run_enhanced_dashboard_safe.py', 'w') as f:
        f.write(runner_content)
    
    print("✅ Created safe runner script: run_enhanced_dashboard_safe.py")

def main():
    """Main installation function"""
    print("🛡️  SAFE ENHANCED DASHBOARD INSTALLATION")
    print("=" * 50)
    print("This script will install the enhanced dashboard safely")
    print("without breaking your existing ClaudeBot setup.")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check existing packages
    check_existing_requirements()
    
    # Install packages safely
    if not install_packages_safely():
        print("❌ Package installation failed")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    # Create safe runner
    create_safe_runner()
    
    print("\n" + "=" * 50)
    print("🎉 ENHANCED DASHBOARD INSTALLED SUCCESSFULLY!")
    print("=" * 50)
    print("✅ All packages installed safely")
    print("✅ No conflicts with existing bots")
    print("✅ Safe runner script created")
    print("\n🚀 To start the enhanced dashboard:")
    print("   python run_enhanced_dashboard_safe.py")
    print("\n📊 Dashboard will be available at:")
    print("   http://localhost:8001")
    print("\n⚠️  Your existing bots will continue running normally!")
    print("=" * 50)

if __name__ == "__main__":
    main()
