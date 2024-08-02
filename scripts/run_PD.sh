#!/bin/bash

# Default parameter values
MODEL_NAME="GPT2"
PARTIAL_ANSWERS_FILE="test"
CONDA_ENV_NAME="DebateQA"

# Help function
usage() {
    echo "Usage: $0 --input_file input_file [--model_name model_name] [--partial_answers_file partial_answers_file]"
    exit 1
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) MODEL_NAME="$2"; shift ;;
        --input_file) INPUT_FILE="$2"; shift ;;
        --partial_answers_file) PARTIAL_ANSWERS_FILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if input_file is provided
if [ -z "$INPUT_FILE" ]; then
    echo "Error: --input_file is required."
    usage
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

# Run Python script
python3 ./eval_PD.py --model_name "$MODEL_NAME" --input_file "$INPUT_FILE" --partial_answers_file "$PARTIAL_ANSWERS_FILE"

# Deactivate conda environment
conda deactivate
