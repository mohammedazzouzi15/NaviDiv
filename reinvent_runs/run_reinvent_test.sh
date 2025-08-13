#!/bin/bash

# Check if conda is available
if ! command -v conda &> /dev/null
then
    echo "conda could not be found. Please install Anaconda or Miniconda."
    exit 1
fi

ENV_NAME="NaviDiv_test"
export PYTHONPATH="${PYTHONPATH}:/media/mohammed/Work/Navi_diversity/src/navidiv/reinvent/"
echo $PYTHONPATH
PYTHON_SCRIPT_PATH="/media/mohammed/Work/Navi_diversity/src/navidiv/reinvent/"
CONFIG_PATH="/media/mohammed/Work/Navi_diversity/reinvent_runs/conf_folder"
CONFIG_NAME="test"
WD_CURRENT="/media/mohammed/Work/Navi_diversity/reinvent_runs/runs/test_new"

# Set up logging
LOG_FILE="$WD_CURRENT/run_reinvent_test.log"
mkdir -p "$WD_CURRENT"

# Function to log messages to both console and file
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

log_message "Starting REINVENT test run"
log_message "Environment: $ENV_NAME"
log_message "Python path: $PYTHONPATH"


for diversity_config in "$CONFIG_PATH"/diversity_scorer/*.yaml; do
    diversity_scorer_name=$(basename "$diversity_config" .yaml)
    run_name=$diversity_scorer_name
    smart_list_path="${WD_CURRENT}/full_smartlist.txt"  # Fixed: was using undefined $wd

    mkdir -p "$WD_CURRENT"
    log_message "Running with diversity scorer: $diversity_scorer_name"
    
    # Test the specific YAML config file first
    log_message "Testing YAML config file: $diversity_config"
    conda run -n $ENV_NAME python -c "
import yaml
import sys
try:
    with open('$diversity_config', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    print(f'Config file loaded successfully: {config}')
except Exception as e:
    print(f'Failed to load config file: {e}')
    sys.exit(1)
"
    
    # Add error handling for the Python execution
    echo "Starting Python script execution..."
    if ! conda run -n $ENV_NAME python3 "$PYTHON_SCRIPT_PATH/run_reinvent_2.py" \
        --config-name $CONFIG_NAME \
        --config-path $CONFIG_PATH \
        name=$run_name \
        wd=$WD_CURRENT \
        reinvent_common.max_steps=10 \
        diversity_scorer=$diversity_scorer_name; then
        log_message "Error: Failed to run with diversity scorer: $diversity_scorer_name"
        log_message "Continuing with next diversity scorer..."
        exit 1
    fi
    log_message "Successfully ran run_reinvent_2.py with diversity scorer: $diversity_scorer_name"
    if ! conda run -n $ENV_NAME python3 /media/mohammed/Work/Navi_diversity/src/navidiv/get_tsne.py \
            --df_path "$WD_CURRENT/${run_name}/${run_name}_1.csv" \
            --step 3
    then
        log_message "Error: Failed to run get_tsne.py with diversity scorer: $diversity_scorer_name"
        log_message "Continuing with next diversity scorer..."
        exit 1
    fi
    log_message "Successfully ran get_tsne.py with diversity scorer: $diversity_scorer_name"
    if ! conda run -n $ENV_NAME python3 /media/mohammed/Work/Navi_diversity/src/navidiv/run_all_scorers.py \
        --df_path "$WD_CURRENT/${run_name}/${run_name}_1_TSNE.csv" \
        --output_path "$WD_CURRENT/${run_name}/scorer_output"
    then
        log_message "Error: Failed to run run_all_scorers.py with diversity scorer: $diversity_scorer_name"
        log_message "Continuing with next diversity scorer..."
        exit 1
    fi
    log_message "Successfully completed run with diversity scorer: $diversity_scorer_name"
done