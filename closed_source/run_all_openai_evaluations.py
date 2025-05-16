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
    # Define OpenAI model types
    model_types = [
        "gpt-4o", 
        "gpt-4o-mini", 
        "gpt-o4-mini"
    ]
    
    # Define datasets (values/virtues)
    datasets = ["authority", "care", "discrimination", "fairness", "harm", 
                "integrity", "justice", "liberty", "loyalty", "reciprocity", 
                "respect", "responsibility", "sanctity"]
    
    # Define versions
    versions = ["M1", "M2"]
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run batch evaluations for all datasets and OpenAI models")
    parser.add_argument("--base-dir", default="/home/ze/MoralBenchmark", 
                        help="Base directory containing M1 and M2 folders")
    parser.add_argument("--batch-size", type=int, default=8, 
                        help="Batch size for evaluation")
    parser.add_argument("--models", nargs="+", default=model_types,
                        help="Specific models to evaluate (default: all models)")
    parser.add_argument("--datasets", nargs="+", default=datasets,
                        help="Specific datasets to evaluate (default: all datasets)")
    parser.add_argument("--versions", nargs="+", default=versions,
                        help="Specific versions to evaluate (default: all versions)")
    parser.add_argument("--continue-from", default=None, nargs=3, metavar=("MODEL", "DATASET", "VERSION"),
                        help="Continue from a specific point (model dataset version)")
    parser.add_argument("--resume", action="store_true",
                        help="Add --resume flag to batch_evaluation.py commands")
    parser.add_argument("--api-key", type=str, required=True,
                        help="OpenAI API key")
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
    os.makedirs(f"{args.base_dir}/results", exist_ok=True)
    
    # Run evaluations sequentially
    for idx, (model_type, dataset, version) in enumerate(all_evaluations[start_idx:], start=start_idx):
        json_path = f"{args.base_dir}/{version}/{dataset}.json"
        image_dir = f"{args.base_dir}/{version}/{dataset}"
        
        # Create model-specific output directory
        model_output_dir = f"{args.base_dir}/results/{model_type}"
        os.makedirs(model_output_dir, exist_ok=True)
        
        output_path = f"{model_output_dir}/{dataset}_{version}_results_{model_type}.json"
        
        # Skip if JSON file doesn't exist
        if not os.path.exists(json_path):
            print(f"Skipping {json_path} - file not found")
            continue
        
        # Skip if image directory doesn't exist
        if not os.path.exists(image_dir):
            print(f"Skipping {image_dir} - directory not found")
            continue
        
        print(f"\n==================================================")
        print(f"Starting evaluation {idx+1}/{total_evaluations}")
        print(f"Model: {model_type}")
        print(f"Dataset: {dataset}")
        print(f"Version: {version}")
        print(f"==================================================\n")
        
        with open("output.log", "a") as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Starting evaluation {idx+1}/{total_evaluations}\n")
            f.write(f"Model: {model_type}\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Version: {version}\n")
            f.write("=" * 50 + "\n\n")
        
        # Build the command with optional flags
        command = (
            f"python openai_batch_evaluation.py "
            f"--model-type {model_type} "
            f"--json-path {json_path} "
            f"--image-dir {image_dir} "
            f"--output-path {output_path} "
            f"--batch-size {args.batch_size} "
            f"--api-key {args.api_key}"
        )
        
        # Add optional flags if specified
        if args.resume:
            command += " --resume"
        
        run_command(command)
        
        print(f"\n==================================================")
        print(f"Progress: {idx+1}/{total_evaluations} evaluations completed")
        print(f"Current: {model_type} {dataset} {version}")
        print(f"Next: {all_evaluations[idx+1][0]} {all_evaluations[idx+1][1]} {all_evaluations[idx+1][2]}" if idx+1 < total_evaluations else "Done!")
        print(f"==================================================\n")

if __name__ == "__main__":
    main()