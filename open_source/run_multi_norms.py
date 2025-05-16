#!/usr/bin/env python
import os
import subprocess
import argparse
import time

def run_command(command):
    print(f"\n=== Running: {command} ===\n")
    start_time = time.time()
    subprocess.run(command, shell=True)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n=== Command completed in {elapsed:.2f} seconds ===\n")

def main():
    # Define model types with specific sizes based on batch_evaluation.py
    model_types = [
        # Gemma3 models
        "gemma3_4B", "gemma3_12B", "gemma3_27B",
        # GLM4V
        "glm4v", 
        # InternVL3 models
        "internvl3_2B", "internvl3_8B", "internvl3_14B", "internvl3_38B",
        # LLaVA models
        "llava", "llava-next",
        # Phi3 Vision model
        "phi3_v",
        # Qwen2 VL models
        "qwen2_vl_2B", "qwen2_vl_7B",
        # Qwen2.5 VL models
        "qwen2_5_vl_3B", "qwen2_5_vl_7B", "qwen2_5_vl_32B"
    ]
    
    # Define datasets (values/virtues)
    datasets = ["authority", "care", "discrimination", "fairness", "harm", 
                "integrity", "justice", "liberty", "loyalty", "reciprocity", 
                "respect", "responsibility", "sanctity"]
    
    # Define versions
    versions = ["M1", "M2"]
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run multi-norm evaluations for datasets and models")
    parser.add_argument("--base-dir", default="/work/nvme/bdyg/zeyang", 
                        help="Base directory containing M1 and M2 folders")
    parser.add_argument("--batch-size", type=int, default=16, 
                        help="Batch size for evaluation")
    parser.add_argument("--models", nargs="+", default=model_types,
                        help="Specific models to evaluate (default: all models)")
    parser.add_argument("--datasets", nargs="+", default=datasets,
                        help="Specific datasets to evaluate (default: all datasets)")
    parser.add_argument("--versions", nargs="+", default=versions,
                        help="Specific versions to evaluate (default: all versions)")
    parser.add_argument("--continue-from", default=None, nargs=3, metavar=("MODEL", "DATASET", "VERSION"),
                        help="Continue from a specific point (model dataset version)")
    parser.add_argument("--disable-mm-preprocessor-cache", action="store_true", 
                        help="Disable multimodal preprocessor cache")
    args = parser.parse_args()
    
    # Create a list of all evaluations to run
    all_evaluations = []
    for model_type in args.models:
        for dataset in args.datasets:
            for version in args.versions:
                all_evaluations.append((model_type, dataset, version))
    
    # Set starting point if continue-from is specified
    start_idx = 0
    if args.continue_from:
        cont_model, cont_dataset, cont_version = args.continue_from
        for idx, (model_type, dataset, version) in enumerate(all_evaluations):
            if model_type == cont_model and dataset == cont_dataset and version == cont_version:
                start_idx = idx
                print(f"Continuing from {cont_model} {cont_dataset} {cont_version} (index {start_idx})")
                break
    
    # Total number of evaluations to run
    total_evaluations = len(all_evaluations)
    
    # Create output directories if they don't exist
    os.makedirs(f"{args.base_dir}/results_multinorm", exist_ok=True)
    
    # Run evaluations sequentially
    for idx, (model_type, dataset, version) in enumerate(all_evaluations[start_idx:], start=start_idx):
        json_path = f"{args.base_dir}/{version}/{dataset}.json"
        image_dir = f"{args.base_dir}/{version}/{dataset}"
        
        # Create model-specific output directory
        model_output_dir = f"{args.base_dir}/results_multinorm/{model_type}"
        os.makedirs(model_output_dir, exist_ok=True)
        
        output_path = f"{model_output_dir}/{dataset}_{version}_results_multinorm_{model_type}.json"
        
        # Skip if JSON file doesn't exist
        if not os.path.exists(json_path):
            print(f"Skipping {json_path} - file not found")
            continue
        
        # Skip if image directory doesn't exist
        if not os.path.exists(image_dir):
            print(f"Skipping {image_dir} - directory not found")
            continue
        
        print(f"\n==================================================")
        print(f"Starting multi-norm evaluation {idx+1}/{total_evaluations}")
        print(f"Model: {model_type}")
        print(f"Dataset: {dataset}")
        print(f"Version: {version}")
        print(f"==================================================\n")
        
        with open("multinorm_output.log", "a") as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Starting multi-norm evaluation {idx+1}/{total_evaluations}\n")
            f.write(f"Model: {model_type}\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Version: {version}\n")
            f.write("=" * 50 + "\n\n")
        
        # Build the command with optional flags
        command = (
            f"python task3_batch_evaluation.py "
            f"--model-type {model_type} "
            f"--json-path {json_path} "
            f"--image-dir {image_dir} "
            f"--output-path {output_path} "
            f"--batch-size {args.batch_size}"
        )
            
        if args.disable_mm_preprocessor_cache:
            command += " --disable-mm-preprocessor-cache"
        
        run_command(command)
        
        print(f"\n==================================================")
        print(f"Progress: {idx+1}/{total_evaluations} evaluations completed")
        print(f"Current: {model_type} {dataset} {version}")
        print(f"Next: {all_evaluations[idx+1][0]} {all_evaluations[idx+1][1]} {all_evaluations[idx+1][2]}" if idx+1 < total_evaluations else "Done!")
        print(f"==================================================\n")

if __name__ == "__main__":
    main()