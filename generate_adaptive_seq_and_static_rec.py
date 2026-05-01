import argparse
import copy
import os

os.environ["TQDM_DISABLE"] = "1"

from ranking_task import *



def run_sequential_rec_pipeline(
        seed=0, dataset="ml-25m", first_stage_model_name="CPMF",
        candidate_source="generate", experiment_dir=None, candidate_set_cfg=None,
        do_not_change_rd_uni=False, score_update_cfg=None,
        add_prob_to_score=False, truncate_continuous_distribution=True, prob_over_scores=True,
        top_k_min_size=5, top_k_max_size=20, evaluation_jumps=5,
        calculate_expected_precision=False, write_to_csv=False,
        measure_runtime=False
):
    if candidate_source == "generate":
        if candidate_set_cfg is None:
            raise ValueError("candidate_set_cfg must be provided when candidate_source='generate'")
    elif candidate_source == "load":
        if experiment_dir is None:
            raise ValueError("experiment_dir must be provided when candidate_source='load'")
    else:
        raise ValueError(f"Unknown candidate_source: {candidate_source}")

    set_seed(seed)

    top_k_size_options = list(range(top_k_min_size, top_k_max_size + 1, evaluation_jumps))
    scores_type = "scores_with_prob" if add_prob_to_score else "equal_scores"
    fs_models_and_data_path = f'{FIRST_STAGE_MODELS_AND_DATA_MAIN_PATH}{dataset}/seed={seed}/'

    data, first_stage_model = load_first_stage_model_and_data(fs_models_and_data_path,
                                                              first_stage_model_name)

    # Load item embeddings for Local and Global Update
    full_item_embd = None
    full_item_embd_norm = None
    if score_update_cfg["method"] != "no_update":
        if not hasattr(first_stage_model, "item_embeddings"):
            raise ValueError("Score distribution update currently requires a model with item embeddings")

        with torch.no_grad():
            full_item_embd = first_stage_model.item_embeddings.weight.detach().clone()
            full_item_embd_norm = torch.nn.functional.normalize(full_item_embd, p=2, dim=1).detach().clone()


    # Load the candidate items per user if the source is set to "load".
    candidate_items_per_user = None
    if candidate_source == "load":
        candidate_items_path = Path(experiment_dir) / "data" / "candidate_items_per_user.pkl"
        if not candidate_items_path.exists():
            raise FileNotFoundError(f"Could not find candidate items file: {candidate_items_path}")

        with open(candidate_items_path, "rb") as f:
            candidate_items_per_user = pickle.load(f)

        print(f"Loaded frozen candidate items from: {candidate_items_path}")
        print(f"Number of users in loaded candidate file: {len(candidate_items_per_user)}")

    unc_thresholds_by_percentiles = None
    if "uncertainty_based_filtering" in candidate_set_cfg['selection_method'] or \
            any("uncertainty_based_filtering" in approach for approach in STATIC_APPROACHES):
        rand_preds = first_stage_model.predict(data.rand['users'].to(device),
                                               data.rand['items'].to(device))
        unc_thresholds_by_percentiles = np.nanquantile(rand_preds[1], PERCENTILES_FOR_UBF)

    # Understand the score distribution type and prepare the score values accordingly
    score_distribution_type = MODELS_SCORE_DISTRIBUTION_TYPE[first_stage_model_name]
    dataset_raw_score_values = np.array(POSSIBLE_SCORES[dataset])
    if score_distribution_type == "discrete":
        score_values = torch.tensor(POSSIBLE_SCORES[dataset])
    elif score_distribution_type == "normal":
        if truncate_continuous_distribution:
            if prob_over_scores:
                score_values = torch.tensor(POSSIBLE_SCORES[dataset])
            else:
                score_values = torch.tensor(POSSIBLE_SCORES[dataset])[:-1]
        else:
            score_values = torch.tensor([0.0] + POSSIBLE_SCORES[dataset])
    else:
        raise ValueError(f'{score_distribution_type} for score distribution type is invalid')
    score_values = score_values.to(device)

    if write_to_csv:
        if candidate_source == "generate":
            fixed_or_rel_cs_size = candidate_set_cfg['negatives_per_positive'] \
                if candidate_set_cfg['selection_method'] == "pos_with_neg_sampling" \
                else candidate_set_cfg['fixed_size']
            items_ranked_selection_method = (
                f"{fixed_or_rel_cs_size}-{candidate_set_cfg['selection_method']}-"
                f"Max{candidate_set_cfg['max_size']}"
            )
            # is_trunc_dist = (f"trunc_normal_dist/" if score_distribution_type == "normal" and
            #                                           truncate_continuous_distribution else "")
            results_path = Path('./experiments') / dataset / first_stage_model_name / \
                           items_ranked_selection_method / f'seed={seed}'
        else:  # candidate_source == "load"
            results_path = Path(experiment_dir) / "results"

        results_path.mkdir(parents=True, exist_ok=True)

        score_dist_update_str = score_update_cfg["method"]
        if score_update_cfg["method"] != "no_update":
            if score_update_cfg["method"] == "global_update":
                score_dist_update_str = (
                    f"{score_dist_update_str}_{score_update_cfg['global_update_error_type']}"
                )

            score_dist_update_str = f"{score_dist_update_str}_alpha{score_update_cfg['alpha']}"

            if score_update_cfg["method"] == "local_update":
                only_pos = "_pos" if score_update_cfg["local_update_pos_only"] else ""
                score_dist_update_str = f"{score_dist_update_str}_top{score_update_cfg['top_l']}{only_pos}"


        if do_not_change_rd_uni:
            results_file_name = f'no_uni_update_score_{score_dist_update_str}_results.csv'
        else:
            results_file_name = f'uni_update_score_{score_dist_update_str}_results.csv'

        header_results_file = (["user_id", "approach", "k", "top_k_result"] +
                               [metric for metric in METRICS])
        if calculate_expected_precision:
            header_results_file.append("expected_precision")

        if measure_runtime:
            runtime_results_file_name = results_file_name.replace(".csv", "_runtime.csv")
            header_runtime_file = ["user_id", "k_max", "static_total_time",
                                   "adaptive_total_time", "adaptive_avg_step_time",
                                   "adaptive_first_step_time", "adaptive_later_steps_avg_time"]

    results_rows = list()
    if measure_runtime:
        runtime_rows = list()

    test_users = data.test_users if candidate_source == "generate" \
        else list(candidate_items_per_user.keys()) # candidate_source == "load"

    for user_idx, user_id in enumerate(test_users):
        print(f"User IDX: {user_idx}, User ID: {user_id}")
        start_time = time.time()

        user_id_tensor = torch.tensor(user_id).to(device)

        train_val_item_ids = list(data.train_val.item[data.train_val.user == user_id])
        user_test_data = data.test[data.test[:, 0] == user_id][:, 1::]

        rec_items_by_preds = first_stage_model.recommend(user_id_tensor,
                                                         remove_items=np.array(train_val_item_ids),
                                                         n=data.n_item - len(train_val_item_ids))

        if score_distribution_type == "discrete":
            # Note: This part is currently not suitable to global update of score distributions
            item_score_distributions = (
                first_stage_model.predict_items_distribution_for_user(user_id_tensor).to(device)
            )
            base_user_ranking_task = UserRankingTask(user_id, train_val_item_ids, user_test_data,
                                                     rec_items_by_preds, item_score_distributions,
                                                     score_values, top_k_max_size,
                                                     unc_thresholds_by_percentiles)
        elif score_distribution_type == "normal":
            item_means, item_stds = first_stage_model.predict_items_mean_and_var_for_user(user_id_tensor)
            item_score_distributions = (
                create_items_discrete_distribution_from_normal_distribution(
                    item_means, item_stds, dataset_raw_score_values,
                    truncate_continuous_distribution, prob_over_scores
                )
            ).to(device)

            user_embedding = None
            if score_update_cfg["method"] == "global_update":
                with torch.no_grad():
                    user_embedding = (
                        first_stage_model.user_embeddings(user_id_tensor).detach().clone().squeeze(0)
                    )

            base_user_ranking_task = UserRankingTask(user_id, train_val_item_ids, user_test_data,
                                                     rec_items_by_preds, item_score_distributions,
                                                     score_values, top_k_max_size,
                                                     unc_thresholds_by_percentiles,
                                                     item_means=item_means, item_stds=item_stds,
                                                     score_update_cfg=score_update_cfg,
                                                     full_item_embd_norm=full_item_embd_norm,
                                                     full_item_embd = full_item_embd,
                                                     user_embedding=user_embedding,
                                                     raw_score_values_for_rebuild=dataset_raw_score_values,
                                                     truncate_continuous_distribution=truncate_continuous_distribution,
                                                     prob_over_scores=prob_over_scores)

        else:
            raise ValueError(f'{score_distribution_type} for score distribution type is invalid')

        if candidate_source == "generate":
            base_user_ranking_task.generate_candidate_items_set(add_prob_to_score, candidate_set_cfg)
        else:  # candidate_source == "load"
            user_candidate_item_ids = candidate_items_per_user[user_id]
            base_user_ranking_task.set_candidate_items(user_candidate_item_ids, add_prob_to_score)

        base_user_ranking_task.run_rank_dist(scores_type, rd_by_scores=False)

        static_total_time = None
        adaptive_total_time = None
        adaptive_avg_step_time = None
        adaptive_first_step_time = None
        adaptive_later_steps_avg_time = None

        # Generate static recommendations
        for static_approach in STATIC_APPROACHES:
            user_ranking_task = copy.deepcopy(base_user_ranking_task)
            for top_k_size in top_k_size_options:

                static_start_time = None
                if measure_runtime and static_approach == "global_top_k" and top_k_size == top_k_max_size:
                    static_start_time = time.perf_counter()
                    user_ranking_task.run_rank_dist(scores_type, rd_by_scores=False)

                top_k_item_ids, top_k_item_idxs = user_ranking_task.get_rec(static_approach,
                                                                            top_k_size)

                if static_start_time is not None:
                    static_total_time = time.perf_counter() - static_start_time

                metric_results = user_ranking_task.compute_exact_metrics(top_k_item_ids)
                results_row = {"user_id": user_id, "approach": static_approach,
                               "k": top_k_size, "top_k_result": top_k_item_ids}
                results_row.update(metric_results)
                results_rows.append(results_row)


        # Generate adaptive sequential recommendations
        rec_size = 1
        for adaptive_approach in ADAPTIVE_APPROACHES:
            user_ranking_task = copy.deepcopy(base_user_ranking_task)

            measure_this_adaptive, step_start_time = False, None
            adaptive_step_times = []
            if measure_runtime and adaptive_approach == "A-GT-t":
                measure_this_adaptive = True
                step_start_time = time.perf_counter()
                user_ranking_task.run_rank_dist(scores_type, rd_by_scores=False)

            method = "global_top_k" if adaptive_approach in ["MPP-1", "A-GT-t"] else "expected_score"
            item_ids_recommended, item_idxs_recommended = [], []
            next_rec_item_id, next_rec_item_idx = user_ranking_task.get_rec(method, rec_size,
                                                                            k_semantics=1,
                                                                            exclude_item_ids=[])
            next_rec_item_id, next_rec_item_idx = next_rec_item_id[0], next_rec_item_idx[0]
            item_ids_recommended.append(next_rec_item_id)
            item_idxs_recommended.append(next_rec_item_idx)

            if measure_this_adaptive:
                adaptive_step_times.append(time.perf_counter() - step_start_time)

            metric_results = user_ranking_task.compute_exact_metrics(item_ids_recommended)
            results_row = {"user_id": user_id, "approach": adaptive_approach,
                           "k": 1, "top_k_result": list(item_ids_recommended)}
            results_row.update(metric_results)
            results_rows.append(results_row)

            for top_k_size in range(2, top_k_max_size + 1):
                step_start_time = time.perf_counter() if measure_this_adaptive else None

                user_ranking_task.update_uni_and_score_dist_after_rec(next_rec_item_id,
                                                                      do_not_change_rd_uni)
                if adaptive_approach != 'A-ES':
                    user_ranking_task.run_rank_dist(scores_type, rd_by_scores=False)

                k_semantics = top_k_size if adaptive_approach == "A-GT-t" else \
                    (1 if adaptive_approach == "MPP-1" else None)
                next_rec_item_id, next_rec_item_idx = user_ranking_task.get_rec(method, rec_size,
                                                                                k_semantics,
                                                                                exclude_item_ids=item_ids_recommended)
                next_rec_item_id, next_rec_item_idx = next_rec_item_id[0], next_rec_item_idx[0]
                item_ids_recommended.append(next_rec_item_id)
                item_idxs_recommended.append(next_rec_item_idx)

                if measure_this_adaptive:
                    adaptive_step_times.append(time.perf_counter() - step_start_time)

                metric_results = user_ranking_task.compute_exact_metrics(item_ids_recommended)
                results_row = {"user_id": user_id, "approach": adaptive_approach,
                               "k": top_k_size, "top_k_result": list(item_ids_recommended)}
                results_row.update(metric_results)
                results_rows.append(results_row)

            if measure_this_adaptive:
                adaptive_total_time = sum(adaptive_step_times)
                adaptive_avg_step_time = adaptive_total_time / len(adaptive_step_times)
                adaptive_first_step_time = adaptive_step_times[0]
                adaptive_later_steps_avg_time = (
                    np.mean(adaptive_step_times[1:]) if len(adaptive_step_times) > 1 else np.nan
                )

        print(f"User {user_id} execution finished in {round(time.time()-start_time, 4)}")
        print("*"*25)

        if measure_runtime:
            runtime_rows.append({
                "user_id": user_id,
                "k_max": top_k_max_size,
                "static_total_time": static_total_time,
                "adaptive_total_time": adaptive_total_time,
                "adaptive_avg_step_time": adaptive_avg_step_time,
                "adaptive_first_step_time": adaptive_first_step_time,
                "adaptive_later_steps_avg_time": adaptive_later_steps_avg_time,
            })

        if write_to_csv and (user_idx + 1) % 100 == 0:
            write_results_to_csv(results_path, header_results_file, results_rows, results_file_name)
            results_rows.clear()

            if measure_runtime:
                write_results_to_csv(results_path, header_runtime_file, runtime_rows, runtime_results_file_name)
                runtime_rows.clear()

            print(f"Writing the results to CSV, finishing execution of user_idx={user_idx}")


    if write_to_csv:
        write_results_to_csv(results_path, header_results_file, results_rows, results_file_name)
        if measure_runtime:
            write_results_to_csv(results_path, header_runtime_file, runtime_rows, runtime_results_file_name)

    print(f"Finished!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default="ml-25m")
    parser.add_argument('--fs_model_name', type=str, default="CPMF")

    # Candidate-set source
    parser.add_argument('--candidate_source', type=str,
                        default="generate", choices=["generate", "load"],
                        help=("Where to get the candidate set from. " 
                              "'generate' builds it inside the pipeline from the raw data. "
                              "'load' loads a precomputed candidate_items_per_user.pkl file "
                              "from the given experiment directory."))

    parser.add_argument('--experiment_dir', type=str, default=None,
                        help=("Path to the experiment root directory. "
                              "When candidate_source='load', the code will load candidate items from "
                              "<experiment_dir>/data/candidate_items_per_user.pkl "
                              "and may write results under <experiment_dir>/results/."))

    # Candidate-set construction args (used when candidate_source='generate')
    parser.add_argument('--candidate_set_selection_method', type=str,
                        default="pos_with_neg_sampling",
                        help=("How to construct the initial candidate set when candidate_source='generate'. "
                              "'all_pos_neg_sampling' builds U1 as all test positives + sampled negatives. "
                              "'reranking-*' selects the top items according to the specified reranking rule."))

    parser.add_argument('--candidate_set_size', type=int, default=1000,
                        help=("Fixed total candidate set size (|U1|). Used only for fixed-size candidate-set modes "
                              "(e.g., reranking-*). Ignored for 'all_pos_neg_sampling'."))

    parser.add_argument('--negatives_per_positive', type=int, default=100,
                        help=("Number of negative items to sample per test-positive item (X). "
                              "Used only when candidate_set_selection_method='all_pos_neg_sampling'."))

    parser.add_argument('--candidate_set_max_size', type=int, default=1000,
                        help=("Maximum allowed total candidate set size (cap on |U1| = #pos + #neg). "
                              "Applies to all methods. For 'all_pos_neg_sampling' it caps the total size after adding positives."))

    parser.add_argument('--do_not_change_rd_uni', action='store_true', default=False)
    parser.add_argument('--score_dist_update_feedback', type=str,
                        default="no_update", choices=["no_update", "local_update", "global_update"],
                        help=("How to update the score distribution. "
                              "Local update refers to sim_mean_update, "
                              "and global update refers to user_embd_update"))
    # parser.add_argument('--save_rank_dist_results', action='store_true', default=True)

    # Args for local\global updating
    parser.add_argument('--local_update_top_l', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--local_update_pos_only', action='store_true', default=False)

    parser.add_argument('--global_update_error_type', type=str, default="mean_error",
                        choices=["mean_error", "uncertainty_scaled_error"],
                        help=("Error term used only by the global update. "
                              "'mean_error' uses f - mu. "
                              "'uncertainty_scaled_error' uses (f - mu) / sigma^2."))

    parser.add_argument('--measure_runtime', action='store_true', default=False,
                        help=("If set, measure runtime for one selected static approach and "
                              "one selected adaptive approach during the actual execution."))

    args = parser.parse_args()

    seed = args.seed
    dataset = args.dataset_name
    first_stage_model_name = args.fs_model_name

    candidate_source = args.candidate_source
    experiment_dir = args.experiment_dir

    if candidate_source == "load" and experiment_dir is None:
        parser.error("--experiment_dir must be provided when --candidate_source load")

    candidate_set_cfg = {
        "selection_method": args.candidate_set_selection_method,
        "fixed_size": args.candidate_set_size,
        "negatives_per_positive": args.negatives_per_positive,
        "max_size": args.candidate_set_max_size
    }

    # rd_uni_exclude_shown = args.rd_uni_exclude_shown
    score_dist_update_feedback = args.score_dist_update_feedback

    # save_rank_dist_results = args.save_rank_dist_results

    score_update_cfg = dict()
    score_update_cfg["method"] = score_dist_update_feedback
    if score_dist_update_feedback != "no_update":
        score_update_cfg["alpha"] = args.alpha
        if score_dist_update_feedback == "local_update":
            score_update_cfg["top_l"] = args.local_update_top_l
            score_update_cfg["local_update_pos_only"] = args.local_update_pos_only
        elif score_dist_update_feedback == "global_update":
            score_update_cfg["global_update_error_type"] = args.global_update_error_type

    run_sequential_rec_pipeline(
        seed=seed,
        dataset=dataset,
        first_stage_model_name=first_stage_model_name,
        candidate_source=candidate_source,
        experiment_dir=experiment_dir,
        candidate_set_cfg=candidate_set_cfg,
        do_not_change_rd_uni=args.do_not_change_rd_uni,
        score_update_cfg=score_update_cfg,
        add_prob_to_score=False,
        truncate_continuous_distribution=True,
        prob_over_scores=True,
        top_k_min_size=1,
        top_k_max_size=20,
        evaluation_jumps=1,
        calculate_expected_precision=False,
        write_to_csv=True,
        measure_runtime=args.measure_runtime
    )


if __name__ == '__main__':
    main()

