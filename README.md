# Adaptive Sequential Recommendation under Uncertain Scores

This is the code repository for our paper "Adaptive Sequential Recommendation under Uncertain Scores", currently under review at [VLDB 2027](https://vldb.org/2027/).
In addition to the code, this repository includes the submitted article and the technical report.

## Running Experiments

The experimental pipeline consists of four main steps:
1. Training a first-stage model for generating uncertain score distributions.
2. Generating the experiment data, including candidate sets.
3. Running the static and adaptive sequential recommendation methods.
4. Running the stopping experiment based on the generated recommendation results.


### Probability Computation
The first step is training a model that will be used to generate the score distribution for each user-item pair. 
We use the code from [GitHub](https://github.com/vcoscrato/uncertain), which is available under the directory `probability_computation`.

To train a model, run the script `generate_fs_model_uncertain_ranking_RS.py` located in the 
 path `./probability_computation/uncertain_SnapExplicit/tests/` using the command below:
```
python generate_fs_model_uncertain_ranking_RS.py \
  --dataset_name dataset_name \
  --train \
  --model_name model_name \
  --seed seed
```

### Generating Experiment Data

The next step is to generate the experiment data used by the recommendation pipeline.
For each sampled test user, we construct a candidate set containing held-out positive items and sampled negative items.

The experiment data is generated using the notebook `generate_experiment_data.ipynb`.
Before running the notebook, set the script parameters according to the desired dataset, seed, first-stage model, and candidate-set configuration.

The generated data is saved under `SAVE_DIR`, which is printed by the notebook during execution.
This directory contains the files required by the recommendation generation script, including `sampled_test_users.pkl` and `candidate_items_per_user.pkl`.


### Generating Adaptive Sequential and Static Recommendations

After generating the experiment data, the next step is to generate the recommendation results using the script `generate_adaptive_seq_and_static_rec.py`.
The script loads the trained first-stage model and the candidate sets generated in the previous step, and produces both static and adaptive sequential recommendation results.

To run the code, use the following command:
```
python generate_adaptive_seq_and_static_rec.py \
  --seed seed \
  --dataset_name dataset_name \
  --fs_model_name fs_model_name \
  --candidate_source load \
  --experiment_dir path_to_experiment_dir \
  --score_dist_update_feedback update_type
```

Script arguments:
* `--seed`: The seed used for reproducibility.
* `--dataset_name`: The name of the dataset.
* `--fs_model_name`: The name of the first-stage model used to generate score distributions.
* `--candidate_source`: The source of the candidate sets. Use `load` to load the candidate sets generated in the previous step.
* `--experiment_dir`: The path to the experiment directory containing the generated candidate sets.
* `--score_dist_update_feedback`: The score distribution update method. 

The script supports three score distribution update options:
* `no_update`: the score distributions are not updated after a recommendation.

* `local_update`: the observed feedback is propagated to items that are similar to the recommended item, based on the item embeddings learned by the first-stage model. This update uses the additional arguments `--alpha`, `--local_update_top_l`, and optionally `--local_update_pos_only`. 

* `global_update`: the observed feedback is used to update the user representation. This update uses the additional argument `--alpha`.

For example, for local update:
```
python generate_adaptive_seq_and_static_rec.py \
  --seed 0 \
  --dataset_name ml-25m \
  --fs_model_name CPMF \
  --candidate_source load \
  --experiment_dir path_to_experiment_dir \
  --score_dist_update_feedback local_update \
  --alpha 0.1 \
  --local_update_top_l 3 \
  --local_update_pos_only
```

The results are saved under `path_to_experiment_dir/results/`.
The output files contain the recommendation results for the static and adaptive methods, including the generated top-$K$ list and the evaluation metrics for each user.


### Evaluating Termination Criteria

After generating the recommendation results, the last step is to evaluate the termination criteria using the script `evaluate_stopping_from_saved_results.py`.
The script loads a saved recommendation results file from `path_to_experiment_dir/results/`, evaluates a termination rule over the generated recommendation prefixes, and saves the termination results to the same results directory.

To run the code, use the following command:
```
python evaluate_stopping_from_saved_results.py \
  --seed seed \
  --dataset_name dataset_name \
  --fs_model_name fs_model_name \
  --experiment_dir path_to_experiment_dir \
  --results_file_name results_file_name \
  --approaches approaches \
  --top_k_max_size top_k_max_size \
  --stopping_rule stopping_rule
```

Script arguments:
* `--seed`: The seed used for reproducibility.
* `--dataset_name`: The name of the dataset.
* `--fs_model_name`: The name of the first-stage model used to generate score distributions.
* `--experiment_dir`: The path to the experiment directory containing the saved recommendation results.
* `--results_file_name`: The name of the recommendation results file, without the `.csv` suffix.
* `--approaches`: The approaches to evaluate. By default, all adaptive approaches are evaluated. To evaluate a specific approach, provide its name.
* `--top_k_max_size`: The maximum recommendation list size considered by the stopping experiment.
* `--stopping_rule`: The stopping rule to evaluate. Supported values are `full_prefix` and `marginal_item`.



