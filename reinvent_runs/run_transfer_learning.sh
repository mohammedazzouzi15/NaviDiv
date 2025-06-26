#!/bin/bash



# Check if conda is available
if ! command -v conda &> /dev/null
then
    echo "conda could not be found. Please install Anaconda or Miniconda."
    exit 1
fi

ENV_NAME="OPTMOLGEN2"
export PYTHONPATH="${PYTHONPATH}:/media/mohammed/Work/Navi_diversity/src/navidiv/reinvent"
python_script_path="/media/mohammed/Work/Navi_diversity/src/navidiv/reinvent/"
config_path="/media/mohammed/Work/Navi_diversity/reinvent_runs/conf_folder"
config_name=transfer_learning
wd="/media/mohammed/Work/Navi_diversity/reinvent_runs"
conda run -n $ENV_NAME python3 "$python_script_path/run_TL_reinvent.py" \
            --config-name $config_name \
            --config-path $config_path > "$wd/transfer_learning.log"
