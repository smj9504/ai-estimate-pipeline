# Scripts Directory

This directory contains all server execution and management scripts for the AI Estimate Pipeline.

## 📁 File Structure

### Server Execution Scripts
- **run.py** - Main unified execution script
- **run.bat** - Windows batch file for quick server startup
- **run_conda.bat** - Conda environment execution script

## 🚀 Usage

### Starting the Server

#### Option 1: Python Script (Recommended)
```bash
# From project root
python scripts/run.py

# With options
python scripts/run.py setup    # Check project setup
python scripts/run.py test     # Run basic tests
python scripts/run.py --conda  # Run with Conda environment
python scripts/run.py --help   # Show help
```

#### Option 2: Windows Batch File
```batch
# From project root
scripts\run.bat
```

#### Option 3: Conda Environment
```batch
# From project root
scripts\run_conda.bat
```

### Direct Server Access
After starting the server:
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/health

## 🔧 Configuration

The scripts automatically handle:
- Environment variable loading from `.env`
- Python path configuration
- Dependency checking
- Server startup with auto-reload

## 📝 Notes

- Ensure all required API keys are configured in `.env` file
- Python 3.8+ is required
- Install dependencies first: `pip install -r requirements.txt`