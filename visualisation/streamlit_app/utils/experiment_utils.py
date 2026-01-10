"""
Experiment utilities for Causal-AIRL Streamlit app.
=====================================================
Run experiments via subprocess, track progress, and refresh data.
Windows-compatible using Python subprocess instead of bash.
"""

import os
import sys
import subprocess
import json
import time
import threading
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def get_python_executable() -> str:
    """Get the Python executable path for running experiments."""
    # Try venv first
    venv_python = os.path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe')
    if os.path.exists(venv_python):
        return venv_python
    
    # Try current interpreter
    return sys.executable


def ensure_results_dir():
    """Ensure results directory exists."""
    results_dir = os.path.join(REPO_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)
    return results_dir


def run_single_experiment(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Run a single experiment via experiments.run_experiment.
    
    Args:
        config_path: Path to YAML config file (relative to repo root)
        overrides: Dictionary of config overrides
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with success status and output
    """
    python_exe = get_python_executable()
    ensure_results_dir()
    
    # Build command
    cmd = [python_exe, '-m', 'experiments.run_experiment', '--config', config_path]
    
    if overrides:
        for key, value in overrides.items():
            cmd.extend(['--override', f'{key}={value}'])
    
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Experiment timed out after {timeout}s',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }


def run_quick_baselines(
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_iters: int = 30,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run quick baseline experiments for demo.
    
    Args:
        progress_callback: Callback function(current, total, message)
        max_iters: Number of training iterations (lower = faster)
        seed: Random seed
        
    Returns:
        Dictionary with results summary
    """
    results = []
    methods = ['airl', 'causal_airl']
    total = len(methods)
    
    for i, method in enumerate(methods):
        if progress_callback:
            progress_callback(i, total, f'Running {method} baseline...')
        
        config_path = f'configs/gridworld_baseline_{method}.yaml'
        
        # Check if config exists
        full_config_path = os.path.join(REPO_ROOT, config_path)
        if not os.path.exists(full_config_path):
            # Use tiny config if available
            config_path = f'configs/gridworld_tiny_{method}.yaml'
        
        result = run_single_experiment(
            config_path=config_path,
            overrides={
                'train.seed': seed,
                'irl.max_iters': max_iters,
            },
            timeout=180
        )
        
        results.append({
            'method': method,
            'config': config_path,
            **result
        })
    
    if progress_callback:
        progress_callback(total, total, 'Baseline experiments complete!')
    
    return {
        'experiments': results,
        'success': all(r['success'] for r in results),
        'total': total,
        'completed': sum(1 for r in results if r['success'])
    }


def run_confounded_experiments(
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_iters: int = 30,
    seed: int = 42,
    z_values: List[int] = [0, 1]
) -> Dict[str, Any]:
    """
    Run confounded GridWorld experiments for cross-Z evaluation.
    
    Args:
        progress_callback: Callback function(current, total, message)
        max_iters: Number of training iterations
        seed: Random seed
        z_values: Confounder values to test
        
    Returns:
        Dictionary with results summary
    """
    results = []
    methods = ['airl', 'causal_airl']
    total = len(methods) * len(z_values)
    current = 0
    
    for method in methods:
        for z in z_values:
            if progress_callback:
                progress_callback(current, total, f'Running {method} z={z}...')
            
            config_path = f'configs/confounded_{method}_z{z}.yaml'
            
            # Check if config exists
            full_config_path = os.path.join(REPO_ROOT, config_path)
            if not os.path.exists(full_config_path):
                # Skip if config doesn't exist
                current += 1
                continue
            
            result = run_single_experiment(
                config_path=config_path,
                overrides={
                    'train.seed': seed,
                    'irl.max_iters': max_iters,
                },
                timeout=180
            )
            
            results.append({
                'method': method,
                'z': z,
                'config': config_path,
                **result
            })
            
            current += 1
    
    if progress_callback:
        progress_callback(total, total, 'Confounded experiments complete!')
    
    return {
        'experiments': results,
        'success': all(r['success'] for r in results if 'success' in r),
        'total': total,
        'completed': sum(1 for r in results if r.get('success', False))
    }


def run_full_experiment_suite(
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_iters: int = 50,
    seed: int = 42,
    include_confounded: bool = True
) -> Dict[str, Any]:
    """
    Run full experiment suite: baselines + confounded.
    
    Args:
        progress_callback: Callback function
        max_iters: Training iterations per experiment
        seed: Random seed
        include_confounded: Whether to include confounded experiments
        
    Returns:
        Combined results
    """
    all_results = {
        'baselines': None,
        'confounded': None,
        'total_time': 0,
        'success': False
    }
    
    start_time = time.time()
    
    # Baselines
    if progress_callback:
        progress_callback(0, 100, 'Starting baseline experiments...')
    
    all_results['baselines'] = run_quick_baselines(
        progress_callback=lambda c, t, m: progress_callback(int(c/t * 40), 100, m) if progress_callback else None,
        max_iters=max_iters,
        seed=seed
    )
    
    # Confounded
    if include_confounded:
        if progress_callback:
            progress_callback(40, 100, 'Starting confounded experiments...')
        
        all_results['confounded'] = run_confounded_experiments(
            progress_callback=lambda c, t, m: progress_callback(40 + int(c/t * 60), 100, m) if progress_callback else None,
            max_iters=max_iters,
            seed=seed
        )
    
    all_results['total_time'] = time.time() - start_time
    all_results['success'] = (
        all_results['baselines'] is not None and 
        all_results['baselines'].get('success', False)
    )
    
    if progress_callback:
        progress_callback(100, 100, 'All experiments complete!')
    
    return all_results


class ExperimentRunner:
    """
    Async experiment runner with progress tracking for Streamlit.
    """
    
    def __init__(self):
        self.is_running = False
        self.progress = 0
        self.status = ""
        self.results = None
        self.error = None
        self.thread = None
    
    def _progress_callback(self, current: int, total: int, message: str):
        if total > 0:
            self.progress = current / total
        self.status = message
    
    def run_async(
        self,
        experiment_type: str = 'baselines',
        max_iters: int = 30,
        seed: int = 42
    ):
        """Start experiments in background thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.progress = 0
        self.status = "Starting..."
        self.results = None
        self.error = None
        
        def run():
            try:
                if experiment_type == 'baselines':
                    self.results = run_quick_baselines(
                        progress_callback=self._progress_callback,
                        max_iters=max_iters,
                        seed=seed
                    )
                elif experiment_type == 'confounded':
                    self.results = run_confounded_experiments(
                        progress_callback=self._progress_callback,
                        max_iters=max_iters,
                        seed=seed
                    )
                elif experiment_type == 'full':
                    self.results = run_full_experiment_suite(
                        progress_callback=self._progress_callback,
                        max_iters=max_iters,
                        seed=seed
                    )
            except Exception as e:
                self.error = str(e)
            finally:
                self.is_running = False
        
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            'is_running': self.is_running,
            'progress': self.progress,
            'status': self.status,
            'results': self.results,
            'error': self.error
        }


def list_available_configs() -> List[str]:
    """List available experiment config files."""
    configs_dir = os.path.join(REPO_ROOT, 'configs')
    if not os.path.isdir(configs_dir):
        return []
    
    return [f for f in os.listdir(configs_dir) if f.endswith('.yaml')]


def check_experiment_prerequisites() -> Dict[str, bool]:
    """Check if experiment prerequisites are met."""
    checks = {
        'python_available': False,
        'configs_exist': False,
        'experiments_module': False,
        'repo_root_valid': os.path.isdir(REPO_ROOT)
    }
    
    # Check Python
    python_exe = get_python_executable()
    checks['python_available'] = os.path.exists(python_exe)
    
    # Check configs
    configs_dir = os.path.join(REPO_ROOT, 'configs')
    checks['configs_exist'] = os.path.isdir(configs_dir) and len(list_available_configs()) > 0
    
    # Check experiments module
    experiments_dir = os.path.join(REPO_ROOT, 'experiments')
    checks['experiments_module'] = os.path.isdir(experiments_dir)
    
    return checks
