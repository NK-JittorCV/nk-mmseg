#!/usr/bin/env python3
"""
Batch evaluation script for all DFormer model variants
Systematically evaluates all models on both datasets and generates comprehensive results
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime

# Model configurations and expected checkpoints
MODELS = {
    # DFormer series
    'DFormer-T': {
        'nyu_config': 'local_configs.NYUDepthv2.DFormer_Tiny',
        'nyu_checkpoint': 'checkpoints/trained/NYUv2_DFormer_Tiny.pth',
        'sunrgbd_config': 'local_configs.SUNRGBD.DFormer_Tiny',
        'sunrgbd_checkpoint': 'checkpoints/trained/SUNRGBD_DFormer_Tiny.pth',
        'target_nyu': 51.8,
        
        'target_sunrgbd': 48.8
    },
    'DFormer-S': {
        'nyu_config': 'local_configs.NYUDepthv2.DFormer_Small',
        'nyu_checkpoint': 'checkpoints/trained/NYUv2_DFormer_Small.pth',
        'sunrgbd_config': 'local_configs.SUNRGBD.DFormer_Small',
        'sunrgbd_checkpoint': 'checkpoints/trained/SUNRGBD_DFormer_Small.pth',
        'target_nyu': 53.6,
        'target_sunrgbd': 50.0
    },
    'DFormer-B': {
        'nyu_config': 'local_configs.NYUDepthv2.DFormer_Base',
        'nyu_checkpoint': 'checkpoints/trained/NYUv2_DFormer_Base.pth',
        'sunrgbd_config': 'local_configs.SUNRGBD.DFormer_Base',
        'sunrgbd_checkpoint': 'checkpoints/trained/SUNRGBD_DFormer_Base.pth',
        'target_nyu': 55.6,
        'target_sunrgbd': 51.2
    },
    'DFormer-L': {
        'nyu_config': 'local_configs.NYUDepthv2.DFormer_Large',
        'nyu_checkpoint': 'checkpoints/trained/NYUv2_DFormer_Large.pth',
        'sunrgbd_config': 'local_configs.SUNRGBD.DFormer_Large',
        'sunrgbd_checkpoint': 'checkpoints/trained/SUNRGBD_DFormer_Large.pth',
        'target_nyu': 57.2,
        'target_sunrgbd': 52.5
    },
    # DFormerv2 series
    'DFormerv2-S': {
        'nyu_config': 'local_configs.NYUDepthv2.DFormerv2_S',
        'nyu_checkpoint': 'checkpoints/trained/DFormerv2_Small_NYU.pth',
        'sunrgbd_config': 'local_configs.SUNRGBD.DFormerv2_S',
        'sunrgbd_checkpoint': 'checkpoints/trained/DFormerv2_Small_SUNRGBD.pth',
        'target_nyu': 56.0,
        'target_sunrgbd': 51.5
    },
    'DFormerv2-B': {
        'nyu_config': 'local_configs.NYUDepthv2.DFormerv2_B',
        'nyu_checkpoint': 'checkpoints/trained/DFormerv2_Base_NYU.pth',
        'sunrgbd_config': 'local_configs.SUNRGBD.DFormerv2_B',
        'sunrgbd_checkpoint': 'checkpoints/trained/DFormerv2_Base_SUNRGBD.pth',
        'target_nyu': 57.7,
        'target_sunrgbd': 52.8
    },
    'DFormerv2-L': {
        'nyu_config': 'local_configs.NYUDepthv2.DFormerv2_L',
        'nyu_checkpoint': 'checkpoints/trained/DFormerv2_Large_NYU.pth',
        'sunrgbd_config': 'local_configs.SUNRGBD.DFormerv2_L',
        'sunrgbd_checkpoint': 'checkpoints/trained/DFormerv2_Large_SUNRGBD.pth',
        'target_nyu': 58.4,
        'target_sunrgbd': 53.3
    }
}

def run_evaluation(config, checkpoint, dataset_name, model_name, use_multiscale=False):
    """Run evaluation for a single model configuration."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {dataset_name}")
    print(f"Config: {config}")
    print(f"Checkpoint: {checkpoint}")
    print(f"{'='*60}")
    
    if not os.path.exists(checkpoint):
        print(f"❌ Checkpoint not found: {checkpoint}")
        return None
    
    # Build command
    cmd = [
        'conda', 'run', '-n', 'jittordet',
        'python', 'utils/eval.py',
        '--config', config,
        '--continue_fpath', checkpoint,
        '--gpus', '1',
        '--verbose'
    ]
    
    if use_multiscale:
        cmd.extend(['--multi_scale', '--flip', '--scales', '0.75', '1.0', '1.25'])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        end_time = time.time()
        
        if result.returncode == 0:
            # Parse mIoU from output
            output_lines = result.stdout.split('\n')
            miou = None
            macc = None
            overall_acc = None
            
            for line in output_lines:
                if line.startswith('mIoU:'):
                    miou = float(line.split(':')[1].strip())
                elif line.startswith('mAcc:'):
                    macc = float(line.split(':')[1].strip())
                elif line.startswith('Overall Acc:'):
                    overall_acc = float(line.split(':')[1].strip())
            
            eval_time = end_time - start_time
            
            return {
                'miou': miou,
                'macc': macc,
                'overall_acc': overall_acc,
                'eval_time': eval_time,
                'success': True,
                'output': result.stdout,
                'multiscale': use_multiscale
            }
        else:
            print(f"❌ Evaluation failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return {
                'success': False,
                'error': result.stderr,
                'multiscale': use_multiscale
            }
    
    except subprocess.TimeoutExpired:
        print(f"❌ Evaluation timed out after 1 hour")
        return {
            'success': False,
            'error': 'Timeout after 1 hour',
            'multiscale': use_multiscale
        }
    except Exception as e:
        print(f"❌ Evaluation failed with exception: {e}")
        return {
            'success': False,
            'error': str(e),
            'multiscale': use_multiscale
        }

def main():
    """Main evaluation function."""
    print("Starting comprehensive DFormer evaluation...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Priority order: DFormerv2-L first (already partially done), then others
    priority_models = ['DFormerv2-L', 'DFormer-L', 'DFormerv2-B', 'DFormerv2-S', 'DFormer-B', 'DFormer-S', 'DFormer-T']
    
    for model_name in priority_models:
        if model_name not in MODELS:
            continue
            
        model_config = MODELS[model_name]
        results[model_name] = {}
        
        # Evaluate on NYUDepthv2
        print(f"\n🔄 Starting {model_name} evaluation on NYUDepthv2...")
        nyu_result = run_evaluation(
            model_config['nyu_config'],
            model_config['nyu_checkpoint'],
            'NYUDepthv2',
            model_name,
            use_multiscale=False
        )
        
        if nyu_result and nyu_result['success']:
            results[model_name]['nyu_single'] = nyu_result
            target_nyu = model_config['target_nyu']
            actual_nyu = nyu_result['miou']
            gap_nyu = target_nyu - actual_nyu
            
            print(f"✅ {model_name} NYUDepthv2 Single-scale: mIoU {actual_nyu:.4f} (target {target_nyu:.4f}, gap {gap_nyu:.4f})")
            
            # If gap > 1%, try multi-scale
            if gap_nyu > 0.01:
                print(f"🔄 Gap > 1%, trying multi-scale evaluation...")
                nyu_ms_result = run_evaluation(
                    model_config['nyu_config'],
                    model_config['nyu_checkpoint'],
                    'NYUDepthv2',
                    model_name,
                    use_multiscale=True
                )
                if nyu_ms_result and nyu_ms_result['success']:
                    results[model_name]['nyu_multiscale'] = nyu_ms_result
                    actual_nyu_ms = nyu_ms_result['miou']
                    gap_nyu_ms = target_nyu - actual_nyu_ms
                    print(f"✅ {model_name} NYUDepthv2 Multi-scale: mIoU {actual_nyu_ms:.4f} (gap {gap_nyu_ms:.4f})")
        else:
            results[model_name]['nyu_single'] = nyu_result
            print(f"❌ {model_name} NYUDepthv2 evaluation failed")
        
        # Evaluate on SUN-RGBD
        print(f"\n🔄 Starting {model_name} evaluation on SUN-RGBD...")
        sunrgbd_result = run_evaluation(
            model_config['sunrgbd_config'],
            model_config['sunrgbd_checkpoint'],
            'SUN-RGBD',
            model_name,
            use_multiscale=False
        )
        
        if sunrgbd_result and sunrgbd_result['success']:
            results[model_name]['sunrgbd_single'] = sunrgbd_result
            target_sunrgbd = model_config['target_sunrgbd']
            actual_sunrgbd = sunrgbd_result['miou']
            gap_sunrgbd = target_sunrgbd - actual_sunrgbd
            
            print(f"✅ {model_name} SUN-RGBD Single-scale: mIoU {actual_sunrgbd:.4f} (target {target_sunrgbd:.4f}, gap {gap_sunrgbd:.4f})")
            
            # If gap > 1%, try multi-scale
            if gap_sunrgbd > 0.01:
                print(f"🔄 Gap > 1%, trying multi-scale evaluation...")
                sunrgbd_ms_result = run_evaluation(
                    model_config['sunrgbd_config'],
                    model_config['sunrgbd_checkpoint'],
                    'SUN-RGBD',
                    model_name,
                    use_multiscale=True
                )
                if sunrgbd_ms_result and sunrgbd_ms_result['success']:
                    results[model_name]['sunrgbd_multiscale'] = sunrgbd_ms_result
                    actual_sunrgbd_ms = sunrgbd_ms_result['miou']
                    gap_sunrgbd_ms = target_sunrgbd - actual_sunrgbd_ms
                    print(f"✅ {model_name} SUN-RGBD Multi-scale: mIoU {actual_sunrgbd_ms:.4f} (gap {gap_sunrgbd_ms:.4f})")
        else:
            results[model_name]['sunrgbd_single'] = sunrgbd_result
            print(f"❌ {model_name} SUN-RGBD evaluation failed")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'evaluation_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to {results_file}")
    
    # Generate summary table
    generate_summary_table(results)

def generate_summary_table(results):
    """Generate a comprehensive summary table."""
    print(f"\n{'='*100}")
    print("COMPREHENSIVE EVALUATION RESULTS SUMMARY")
    print(f"{'='*100}")
    
    header = f"{'Model':<15} {'NYU Target':<10} {'NYU Single':<12} {'NYU Multi':<12} {'SUN Target':<10} {'SUN Single':<12} {'SUN Multi':<12} {'Status':<10}"
    print(header)
    print("-" * 100)
    
    for model_name in ['DFormerv2-L', 'DFormerv2-B', 'DFormerv2-S', 'DFormer-L', 'DFormer-B', 'DFormer-S', 'DFormer-T']:
        if model_name not in results or model_name not in MODELS:
            continue
            
        model_config = MODELS[model_name]
        model_results = results[model_name]
        
        # NYU results
        nyu_target = model_config['target_nyu']
        nyu_single = model_results.get('nyu_single', {}).get('miou', 'N/A') if model_results.get('nyu_single', {}).get('success') else 'FAIL'
        nyu_multi = model_results.get('nyu_multiscale', {}).get('miou', 'N/A') if model_results.get('nyu_multiscale', {}).get('success') else 'N/A'
        
        # SUN-RGBD results
        sun_target = model_config['target_sunrgbd']
        sun_single = model_results.get('sunrgbd_single', {}).get('miou', 'N/A') if model_results.get('sunrgbd_single', {}).get('success') else 'FAIL'
        sun_multi = model_results.get('sunrgbd_multiscale', {}).get('miou', 'N/A') if model_results.get('sunrgbd_multiscale', {}).get('success') else 'N/A'
        
        # Status
        status = "✅" if (isinstance(nyu_single, float) and isinstance(sun_single, float)) else "❌"
        
        # Format numbers
        nyu_single_str = f"{nyu_single:.3f}" if isinstance(nyu_single, float) else str(nyu_single)
        nyu_multi_str = f"{nyu_multi:.3f}" if isinstance(nyu_multi, float) else str(nyu_multi)
        sun_single_str = f"{sun_single:.3f}" if isinstance(sun_single, float) else str(sun_single)
        sun_multi_str = f"{sun_multi:.3f}" if isinstance(sun_multi, float) else str(sun_multi)
        
        row = f"{model_name:<15} {nyu_target:<10.1f} {nyu_single_str:<12} {nyu_multi_str:<12} {sun_target:<10.1f} {sun_single_str:<12} {sun_multi_str:<12} {status:<10}"
        print(row)
    
    print(f"{'='*100}")

if __name__ == '__main__':
    main()
