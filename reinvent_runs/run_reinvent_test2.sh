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
config_name=test
wd="/media/mohammed/Work/Navi_diversity/reinvent_runs/runs/experiment_1506_sigma"


for i in 24 48 64 128 256 512; do
    wd_current="${wd}_${i}/"
    for diversity_config in "$config_path"/diversity_scorer/*.yaml; do
        diversity_scorer_name=$(basename "$diversity_config" .yaml)
        run_name=$diversity_scorer_name
        smart_list_path="${wd}/full_smartlist.txt"

        mkdir -p "$wd_current"
        echo "Running with diversity scorer: $diversity_scorer_name, run $i"
        conda run -n $ENV_NAME python3 "$python_script_path/run_reinvent_2.py" \
            --config-name $config_name \
            --config-path $config_path \
            name=$run_name \
            wd=$wd_current \
            reinvent_common.max_steps=1000\
            diversity_scorer=$diversity_scorer_name\
            reinvent_common.learning_strategy.sigma=$i 

        conda run -n $ENV_NAME python3 /media/mohammed/Work/Navi_diversity/src/navidiv/get_tsne.py \
            --df_path "$wd_current/${run_name}/${run_name}_1.csv" \
            --step 10
        conda run -n $ENV_NAME python3 /media/mohammed/Work/Navi_diversity/src/navidiv/run_all_scorers.py \
            --df_path "$wd_current/${run_name}/${run_name}_1_TSNE.csv" \
            --output_path "$wd_current/${run_name}/scorer_output"
    done
done