# REINVENT Implementation Tutorial with NaviDiv Diversity Scorers

This tutorial will guide you through running REINVENT4 with different diversity scorers using the NaviDiv framework. We'll use the working example in the `Tutorials/Running_Reinvent/example/` directory that demonstrates QED optimization with various diversity constraints.

## Overview

The NaviDiv framework provides multiple diversity scoring mechanisms that can be integrated with REINVENT4 to control molecular generation. Instead of optimizing only for a target property (like QED), you can balance property optimization with structural diversity.

## Prerequisites

### Environment Setup
1. **Conda environment**: `NaviDiv_test` (must be activated)
2. **NaviDiv installed** with REINVENT4 dependencies
3. **REINVENT4 integration** enabled

### Verify Environment
```bash
# Activate the correct environment
conda activate NaviDiv_test

# Verify NaviDiv is installed
python -c "import navidiv; print('NaviDiv installed successfully')"

# Verify REINVENT4 is available
python -c "import reinvent; print('REINVENT4 available')"
```

### Tutorial Files Structure
All working examples are in: `/media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/`

```
Tutorials/Running_Reinvent/example/
├── conf_folder/
│   ├── diversity_scorer/          # Different diversity configurations
│   │   ├── All_constraints.yaml
│   │   ├── All_weak_constraints.yaml
│   │   ├── scaffold_only.yaml
│   │   ├── fragement_only.yaml
│   │   ├── ngram_only.yaml
│   │   └── similarity_only.yaml
│   ├── stage_comp/               # Target property configurations
│   ├── reinvent_common/          # REINVENT settings
│   └── test.yaml                 # Main configuration
├── InputGenerator_custom.py      # Custom input generator
├── run_reinvent.sh              # Execution script
└── outputs/                      # Results will be stored here
```

## Step 1: Understanding Diversity Scorer Configurations

### Available Diversity Scorers

The framework includes several pre-configured diversity scoring strategies:

#### 1. **All_constraints.yaml** - Balanced Diversity
```yaml
# Applies all diversity metrics with moderate constraints
# Good for: Balanced exploration with property optimization
```

#### 2. **All_weak_constraints.yaml** - Light Diversity Control
```yaml
# Weaker diversity constraints, prioritizes property optimization
# Good for: When you want some diversity but focus on target property
```

#### 3. **scaffold_only.yaml** - Scaffold Diversity Focus
```yaml
# Emphasizes scaffold diversity while relaxing other constraints
# Good for: Exploring different core molecular frameworks
```

#### 4. **fragement_only.yaml** - Fragment Diversity Focus
```yaml
# Focuses on fragment-level diversity
# Good for: Exploring different molecular building blocks
```

#### 5. **ngram_only.yaml** - String Pattern Diversity
```yaml
# Emphasizes N-gram (sequence pattern) diversity
# Good for: Exploring different SMILES sequence patterns
```

#### 6. **similarity_only.yaml** - Cluster-based Diversity
```yaml
# Uses similarity clustering for diversity control
# Good for: Preventing generation of too-similar molecules
```

### Key Parameters in Diversity Configurations

Each diversity scorer configuration contains:

```yaml
scorer_dicts:
  - prop_dict:
      scorer_name: "Scaffold"           # Type of diversity metric
      scaffold_type: "basic_wire_frame" # Specific variant
      min_count_fragments: 1            # Minimum fragment count
      selection_criteria:
        diff_median_score: 0.1          # Score difference threshold
    score_every: 10                     # Evaluation frequency
    groupby_every: 20                   # Grouping frequency
    selection_criteria:
      count_perc_ratio: 5               # Percentage ratio constraint
      Total Number of Molecules with Substructure: 50  # Count limit
    custom_alert_name: "customalertsscaffold"          # Alert identifier
```

## Step 2: Setting Up Your Configuration

### Main Configuration (test.yaml)

The working configuration is located at:
`Tutorials/Running_Reinvent/example/conf_folder/test.yaml`

```yaml
wd: "/media/mohammed/Work/OPT_MOL_GEN/example_running_to_find_low_band_gap_mols/reinvent_runs/first_run"
name: "stage1"
generate_input_only: false
input_generator: 
  file_path: "/media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/InputGenerator_custom.py"
  class_name: "InputGeneratorCustom"

defaults:
  - diversity_scorer: default          # Will be overridden by command line
  - reinvent_common: default
  - stage_comp: QED                   # Target property
```

**Important paths to update for your setup:**
- `wd`: Change to your desired output directory
- `input_generator.file_path`: Update to the full path of your InputGenerator file

### Target Property Configuration

You can modify the target property by creating new stage_comp configurations:
To implement a new target property, you can check the reinvent4 package on the use of other Scoring Plugins (https://github.com/MolecularAI/REINVENT4/tree/main)
Currently we can use the implemented scorers in reinvent4 with few added implementation in /src/navidiv/reinvent/reinvent_plugins

#### Example: Molecular Weight Optimization
```yaml
# stage_comp/MW.yaml
model_params_rdkit_physchem:
  params_list: ["MW"]
  lower_bound: [200.0]
  higher_bound: [500.0]
  weight: [1.0]
```

#### Example: LogP Optimization
```yaml
# stage_comp/LogP.yaml
model_params_rdkit_physchem:
  params_list: ["ALOGP"]
  lower_bound: [1.0]
  higher_bound: [3.0]
  weight: [1.0]
```

#### Example: Multi-property Optimization
```yaml
# stage_comp/QED_MW.yaml
model_params_rdkit_physchem:
  params_list: ["QED", "MW"]
  lower_bound: [0.5, 200.0]
  higher_bound: [1.0, 500.0]
  weight: [1.0, 0.5]
```

## Step 3: Running REINVENT with Different Diversity Scorers

### Basic Execution Script

The working script is provided at:
`Tutorials/Running_Reinvent/example/run_reinvent.sh`

```bash
#!/bin/bash

# Check if conda is available
if ! command -v conda &> /dev/null
then
    echo "conda could not be found. Please install Anaconda or Miniconda."
    exit 1
fi

# Environment and path configuration
ENV_NAME="NaviDiv_test"
export PYTHONPATH="${PYTHONPATH}:/media/mohammed/Work/Navi_diversity/src/navidiv/reinvent"
python_script_path="/media/mohammed/Work/Navi_diversity/src/navidiv/reinvent/"
config_path="/media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/conf_folder"
config_name=test
wd="/media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/runs/test_case"
i=2

# Create output directory
wd_current="${wd}_${i}/"

# Run all diversity scorer configurations
for diversity_config in "$config_path"/diversity_scorer/*.yaml; do
    diversity_scorer_name=$(basename "$diversity_config" .yaml)
    run_name=$diversity_scorer_name
    
    mkdir -p "$wd_current"
    echo "Running with diversity scorer: $diversity_scorer_name, run $i"
    
    # Run REINVENT with specific diversity scorer
    conda run -n $ENV_NAME python3 "$python_script_path/run_reinvent_2.py" \
        --config-name $config_name \
        --config-path $config_path \
        name=$run_name \
        wd=$wd_current \
        reinvent_common.max_steps=1000 \
        diversity_scorer=$diversity_scorer_name

    # Post-processing: t-SNE visualization
    conda run -n $ENV_NAME python3 /media/mohammed/Work/Navi_diversity/src/navidiv/get_tsne.py \
        --df_path "$wd_current/${run_name}/${run_name}_1.csv" \
        --step 20
        
    # Post-processing: Diversity analysis
    conda run -n $ENV_NAME python3 /media/mohammed/Work/Navi_diversity/src/navidiv/run_all_scorers.py \
        --df_path "$wd_current/${run_name}/${run_name}_1_TSNE.csv" \
        --output_path "$wd_current/${run_name}/scorer_output"
done
```

### Running the Script

1. **Navigate to the example directory:**
   ```bash
   cd /media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/
   ```

2. **Make the script executable:**
   ```bash
   chmod +x run_reinvent.sh
   ```

3. **Run the script:**
   ```bash
   ./run_reinvent.sh
   ```

### Running Individual Configurations

To run a specific diversity scorer, navigate to the example directory and run:

```bash
# Navigate to example directory
cd /media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/

# Activate environment
conda activate NaviDiv_test

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/media/mohammed/Work/Navi_diversity/src/navidiv/reinvent"

# Run specific diversity scorer (e.g., scaffold_only)
python3 /media/mohammed/Work/Navi_diversity/src/navidiv/reinvent/run_reinvent_2.py \
    --config-name test \
    --config-path conf_folder \
    name=scaffold_experiment \
    wd=./runs/scaffold_test \
    reinvent_common.max_steps=100 \
    diversity_scorer=scaffold_only
```

### Quick Test Run

For a quick test with minimal steps:

```bash
# Test run with only 10 steps
python3 /media/mohammed/Work/Navi_diversity/src/navidiv/reinvent/run_reinvent_2.py \
    --config-name test \
    --config-path conf_folder \
    name=test_run \
    wd=./runs/test \
    reinvent_common.max_steps=10 \
    diversity_scorer=All_weak_constraints
```

## Step 4: Understanding the Results

### Output Structure

Each run produces:
```
output_directory/
├── experiment_name/
│   ├── experiment_name_1.csv          # Generated molecules
│   ├── experiment_name_1_TSNE.csv     # With t-SNE coordinates
│   ├── scorer_output/                 # Diversity analysis results
│   └── logs/                          # REINVENT logs
```

### Interpreting Different Diversity Strategies

#### All_constraints Results
- **Characteristics**: Balanced diversity across all metrics
- **Expected outcome**: Moderate property optimization with good structural variety
- **Use case**: General-purpose diverse molecular generation

#### Scaffold_only Results
- **Characteristics**: High scaffold diversity, relaxed other constraints
- **Expected outcome**: Many different core structures, varying ring systems
- **Use case**: Exploring different molecular frameworks

#### Fragment_only Results
- **Characteristics**: Focus on fragment-level diversity
- **Expected outcome**: Varied building blocks and functional groups
- **Use case**: Exploring different molecular components

#### Ngram_only Results
- **Characteristics**: Emphasis on sequence pattern diversity
- **Expected outcome**: Varied SMILES patterns and string representations
- **Use case**: Exploring different chemical notation patterns

## Step 5: Customizing Diversity Constraints

### Creating Custom Diversity Configurations

Create your own diversity scorer configuration:

```yaml
# custom_diversity.yaml
scorer_dicts:
  # Strong scaffold constraints
  - prop_dict:
      scorer_name: Scaffold
      scaffold_type: basic_wire_frame
      min_count_fragments: 1
      selection_criteria:
        diff_median_score: 0.05       # Stricter score requirement
    score_every: 5                    # More frequent evaluation
    groupby_every: 10
    selection_criteria:
      count_perc_ratio: 2             # Stricter ratio
      Total Number of Molecules with Substructure: 25  # Lower limit
    custom_alert_name: customalertsscaffold
  
  # Relaxed fragment constraints
  - prop_dict:
      scorer_name: Fragments
      min_count_fragments: 5          # Higher minimum
      transformation_mode: none
    score_every: 15                   # Less frequent evaluation
    groupby_every: 30
    selection_criteria:
      count_perc_ratio: 10            # More relaxed
      Total Number of Molecules with Substructure: 100
    custom_alert_name: customalerts
```

### Parameter Tuning Guidelines

#### Constraint Strength
- **Lower `count_perc_ratio`**: Stricter diversity constraints
- **Higher `count_perc_ratio`**: More relaxed constraints
- **Lower `Total Number of Molecules`**: Stricter limits
- **Higher values**: More permissive

#### Evaluation Frequency
- **Lower `score_every`**: More frequent diversity checks (slower but more controlled)
- **Higher `score_every`**: Less frequent checks (faster but less controlled)

#### Selection Criteria
- **`diff_median_score`**: Minimum score improvement required
- **Lower values**: Stricter improvement requirements
- **Higher values**: More permissive acceptance

## Step 6: Advanced Workflows

### Combining Multiple Target Properties

Modify `InputGenerator_custom.py` to handle multiple properties:

```python
class model_params_multi_property:
    def __init__(self, cfg, id):
        self.property_name = cfg.params_list[id]
        self.lower_bound = cfg.lower_bound[id] 
        self.upper_bound = cfg.higher_bound[id]
        self.weight = cfg.weight[id]
        self.transformation = getattr(cfg, 'transformation', ['none'] * len(cfg.params_list))[id]

class InputGeneratorCustom(InputGenerator):
    def add_stage_component_multi_property(self, stage1_parameters, component):
        # Implementation for multiple properties
        # ... (detailed implementation)
```

### Sequential Diversity Strategies

Run different diversity strategies in sequence:

```bash
# Phase 1: Broad exploration
python3 run_reinvent_2.py \
    diversity_scorer=All_weak_constraints \
    reinvent_common.max_steps=500

# Phase 2: Scaffold refinement
python3 run_reinvent_2.py \
    diversity_scorer=scaffold_only \
    reinvent_common.max_steps=500

# Phase 3: Fragment optimization
python3 run_reinvent_2.py \
    diversity_scorer=fragement_only \
    reinvent_common.max_steps=500
```

## Step 7: Monitoring and Analysis

### Real-time Monitoring

Monitor your runs by checking the output directory:

```bash
# Check if run is progressing
cd /media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/
tail -f runs/test_case_2/All_constraints/logs/*.log

# Check generated molecules count
wc -l runs/test_case_2/All_constraints/All_constraints_1.csv
```

### Analyzing Results with Python

Create a simple analysis script:

```python
# analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_run(run_path):
    """Analyze a single REINVENT run."""
    csv_file = None
    for file in os.listdir(run_path):
        if file.endswith('_1.csv'):
            csv_file = os.path.join(run_path, file)
            break
    
    if csv_file is None:
        print(f"No CSV file found in {run_path}")
        return None
    
    df = pd.read_csv(csv_file)
    
    # Basic statistics
    stats = {
        'total_molecules': len(df),
        'unique_smiles': df['smiles'].nunique(),
        'mean_score': df['total_score'].mean(),
        'max_score': df['total_score'].max(),
        'final_step': df['step'].max()
    }
    
    return df, stats

# Example usage
if __name__ == "__main__":
    base_path = "runs/test_case_2"
    
    # Analyze all diversity scorers
    results = {}
    for scorer in ['All_constraints', 'All_weak_constraints', 'scaffold_only']:
        run_path = os.path.join(base_path, scorer)
        if os.path.exists(run_path):
            df, stats = analyze_run(run_path)
            if stats:
                results[scorer] = stats
                print(f"\n{scorer}:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
    
    # Plot comparison
    if results:
        scorers = list(results.keys())
        mean_scores = [results[s]['mean_score'] for s in scorers]
        unique_counts = [results[s]['unique_smiles'] for s in scorers]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.bar(scorers, mean_scores)
        ax1.set_title('Mean Scores by Diversity Scorer')
        ax1.set_ylabel('Mean Score')
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        ax2.bar(scorers, unique_counts)
        ax2.set_title('Unique Molecules by Diversity Scorer')
        ax2.set_ylabel('Unique SMILES Count')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('diversity_comparison.png')
        plt.show()
```

### Using NaviDiv Analysis Tools

After your runs complete, use the built-in analysis tools:

```bash
# Navigate to example directory
cd /media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/

# Activate environment
conda activate NaviDiv_test

# Analyze results with t-SNE (if not done automatically)
python3 /media/mohammed/Work/Navi_diversity/src/navidiv/get_tsne.py \
    --df_path runs/test_case_2/All_constraints/All_constraints_1.csv \
    --step 20

# Run comprehensive diversity analysis
python3 /media/mohammed/Work/Navi_diversity/src/navidiv/run_all_scorers.py \
    --df_path runs/test_case_2/All_constraints/All_constraints_1_TSNE.csv \
    --output_path runs/test_case_2/All_constraints/scorer_output
```

## Step 8: Troubleshooting

### Common Issues and Solutions

#### 1. **Import/Module Errors**
```bash
# Error: ModuleNotFoundError: No module named 'navidiv'
# Solution: Ensure correct environment and PYTHONPATH
conda activate NaviDiv_test
export PYTHONPATH="${PYTHONPATH}:/media/mohammed/Work/Navi_diversity/src"
```

#### 2. **Configuration File Not Found**
```bash
# Error: Config file not found
# Solution: Use absolute paths or run from correct directory
cd /media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/
# Then run commands with relative paths
```

#### 3. **REINVENT Model Files Missing**
```bash
# Error: Prior model file not found
# Solution: Check REINVENT4 installation and model paths
python -c "import reinvent; print(reinvent.__file__)"
# Update model paths in configuration files
```

#### 4. **Memory Issues**
```bash
# Symptoms: Out of memory errors
# Solution: Reduce batch sizes in configuration
# Edit reinvent_common/default.yaml:
# batch_size: 64  # Reduce from default
# Or run with fewer steps for testing:
reinvent_common.max_steps=50
```

#### 5. **Slow Generation**
```bash
# Symptoms: Very slow molecule generation
# Solution: Increase score_every frequency in diversity configs
# Edit diversity_scorer/*.yaml files:
score_every: 20  # Increase from 10
```

### Validation Tests

#### Test Environment Setup
```bash
# Test 1: Check environment
conda activate NaviDiv_test
python -c "import navidiv, reinvent, torch; print('All dependencies available')"

# Test 2: Check REINVENT script
cd /media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/
export PYTHONPATH="${PYTHONPATH}:/media/mohammed/Work/Navi_diversity/src"
python3 /media/mohammed/Work/Navi_diversity/src/navidiv/reinvent/run_reinvent_2.py --help

# Test 3: Minimal run (1 step)
python3 /media/mohammed/Work/Navi_diversity/src/navidiv/reinvent/run_reinvent_2.py \
    --config-name test \
    --config-path conf_folder \
    name=minimal_test \
    wd=./runs/test_minimal \
    reinvent_common.max_steps=1 \
    diversity_scorer=All_weak_constraints
```

### Log Analysis

Check logs for issues:

```bash
# Check REINVENT logs
find runs/ -name "*.log" -exec tail -20 {} \;

# Check for common error patterns
grep -r "Error\|Exception\|Failed" runs/*/logs/

# Monitor progress in real-time
tail -f runs/test_case_2/*/logs/*.log
```

## Step 9: Best Practices

### 1. **Start with Test Runs**
Always test with minimal steps first:

```bash
# Quick test (10 steps)
python3 /media/mohammed/Work/Navi_diversity/src/navidiv/reinvent/run_reinvent_2.py \
    --config-name test \
    --config-path conf_folder \
    name=quick_test \
    wd=./runs/test_quick \
    reinvent_common.max_steps=10 \
    diversity_scorer=All_weak_constraints
```

### 2. **Use Appropriate Step Counts**
- **Testing**: 10-50 steps
- **Development**: 100-500 steps  
- **Production**: 1000+ steps

### 3. **Monitor Resource Usage**
```bash
# Monitor memory and CPU usage
htop  # or top

# Monitor GPU usage (if available)
nvidia-smi

# Monitor disk space for outputs
df -h
```

### 4. **Organize Your Runs**
```bash
# Create organized output structure
mkdir -p experiments/$(date +%Y%m%d)_QED_diversity/

# Use descriptive names
python3 run_reinvent_2.py \
    name="QED_scaffold_diversity_$(date +%H%M)" \
    wd="./experiments/$(date +%Y%m%d)_QED_diversity/" \
    # ... other parameters
```

### 5. **Version Control Your Configurations**
```bash
# Copy configurations for each experiment
cp -r conf_folder experiments/$(date +%Y%m%d)_config_backup/

# Document changes in README
echo "Experiment $(date): Modified scaffold constraints" >> experiments/README.md
```

### 6. **Automate Analysis**
Create a simple analysis pipeline:

```bash
#!/bin/bash
# analyze_run.sh
RUN_DIR=$1

if [ -z "$RUN_DIR" ]; then
    echo "Usage: $0 <run_directory>"
    exit 1
fi

echo "Analyzing run in $RUN_DIR"

# Find CSV file
CSV_FILE=$(find $RUN_DIR -name "*_1.csv" | head -1)

if [ -z "$CSV_FILE" ]; then
    echo "No CSV file found in $RUN_DIR"
    exit 1
fi

echo "Found molecules file: $CSV_FILE"

# Basic statistics
echo "Total molecules: $(tail -n +2 $CSV_FILE | wc -l)"
echo "Unique SMILES: $(tail -n +2 $CSV_FILE | cut -d',' -f2 | sort -u | wc -l)"

# Run t-SNE if not exists
TSNE_FILE=${CSV_FILE/_1.csv/_1_TSNE.csv}
if [ ! -f "$TSNE_FILE" ]; then
    echo "Running t-SNE analysis..."
    conda run -n NaviDiv_test python3 /media/mohammed/Work/Navi_diversity/src/navidiv/get_tsne.py \
        --df_path "$CSV_FILE" --step 20
fi

# Run diversity analysis
OUTPUT_DIR="${RUN_DIR}/scorer_output"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Running diversity analysis..."
    conda run -n NaviDiv_test python3 /media/mohammed/Work/Navi_diversity/src/navidiv/run_all_scorers.py \
        --df_path "$TSNE_FILE" --output_path "$OUTPUT_DIR"
fi

echo "Analysis complete. Results in $OUTPUT_DIR"
```

## Conclusion

This tutorial provides a practical, tested framework for running REINVENT4 with various diversity scoring strategies using the NaviDiv framework. All examples are based on working code in the `Tutorials/Running_Reinvent/example/` directory.

### Quick Start Summary

1. **Setup**: `conda activate NaviDiv_test`
2. **Navigate**: `cd /media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/`
3. **Test**: Run minimal test with 10 steps
4. **Execute**: Use `./run_reinvent.sh` for full analysis
5. **Analyze**: Results automatically processed with t-SNE and diversity analysis

### Key Takeaways

- **Balance is crucial**: Find the right trade-off between property optimization and diversity
- **Start small**: Always test with minimal steps first
- **Monitor actively**: Watch logs and resource usage during runs
- **Validate results**: Use post-analysis tools to confirm diversity metrics
- **Document everything**: Keep records of successful configurations

The diversity scorers provide powerful tools for controlling molecular generation beyond simple property optimization. Experiment with different configurations, monitor results carefully, and adjust parameters based on your specific research requirements.

## Additional Resources

### Working Files
- **Example Directory**: `/media/mohammed/Work/Navi_diversity/Tutorials/Running_Reinvent/example/`
- **Configuration Templates**: All YAML files are tested and working
- **Analysis Scripts**: `get_tsne.py` and `run_all_scorers.py` for result analysis

### Further Development
- **Custom Scorers**: Extend the framework by implementing custom diversity metrics
- **REINVENT4 Documentation**: https://github.com/MolecularAI/REINVENT4
- **NaviDiv App**: Use `streamlit run app.py` for interactive result exploration

### Support
- **GitHub Issues**: Report problems on the NaviDiv repository
- **Configuration Help**: Check working examples in the tutorial directory
- **Environment Issues**: Verify `NaviDiv_test` conda environment setup

Happy molecular generation with controlled diversity! 🧬⚗️🎯
