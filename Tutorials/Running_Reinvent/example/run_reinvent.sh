#!/bin/bash



# Check if conda is available
if ! command -v conda &> /dev/null
then
    echo "conda could not be found. Please install Anaconda or Miniconda."
    exit 1
fi

ENV_NAME="NaviDiv_test"
export PYTHONPATH="${PYTHONPATH}:/media/mohammed/Work/Navi_diversity/src/navidiv/reinvent"
python_script_path="/media/mohammed/Work/Navi_diversity/src/navidiv/reinvent/"
config_path="/media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/conf_folder"
config_name=test
wd="/media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/runs/test_case"
i=3

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
        reinvent_common.max_steps=100\
        diversity_scorer=$diversity_scorer_name

    # conda run -n $ENV_NAME python3 /media/mohammed/Work/Navi_diversity/src/navidiv/get_tsne.py \
    #     --df_path "$wd_current/${run_name}/${run_name}_1.csv" \
    #     --step 20
    # conda run -n $ENV_NAME python3 /media/mohammed/Work/Navi_diversity/src/navidiv/run_all_scorers.py \
    #     --df_path "$wd_current/${run_name}/${run_name}_1_TSNE.csv" \
    #     --output_path "$wd_current/${run_name}/scorer_output"
done
