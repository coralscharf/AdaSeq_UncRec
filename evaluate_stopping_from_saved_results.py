import argparse
import ast
import copy
import os
import time

os.environ["TQDM_DISABLE"] = "1"

from SASRecScoreDist.train_and_export import generate_score_distributions, load_exported_rating_head
from ranking_task import *



def get_top_k_res_from_file(experiment_dir, results_file_name,
                            approaches=ADAPTIVE_APPROACHES, top_k_max_size=20):
    results_df = pd.read_csv(f"{experiment_dir}/results/{results_file_name}.csv")
    results_df = results_df[results_df["approach"].isin(approaches)].copy()
    results_df["top_k_result"] = results_df["top_k_result"].apply(ast.literal_eval)
    results_df = results_df[results_df["k"] == top_k_max_size].copy()
    return results_df


def load_candidate_items_per_user(experiment_dir):
    candidate_items_path = Path(experiment_dir) / "data" / "candidate_items_per_user.pkl"
    if not candidate_items_path.exists():
        raise FileNotFoundError(f"Could not find candidate items file: {candidate_items_path}")

    with open(candidate_items_path, "rb") as f:
        candidate_items_per_user = pickle.load(f)

    print(f"Loaded frozen candidate items from: {candidate_items_path}")
    print(f"Number of users in loaded candidate file: {len(candidate_items_per_user)}")
    return candidate_items_per_user


def build_base_user_ranking_task(user_id, data, first_stage_model, first_stage_model_name,
                                 sasrec_model_outputs, sasrec_rating_head, score_distribution_type,
                                 score_values, dataset_raw_score_values, truncate_continuous_distribution,
                                 prob_over_scores, candidate_items_per_user, add_prob_to_score, top_k_max_size):
    user_id_tensor = torch.tensor(user_id).to(device)

    train_val_item_ids = list(data.train_val.item[data.train_val.user == user_id])
    user_test_data = data.test[data.test[:, 0] == user_id][:, 1::]

    if first_stage_model_name == "SASRecDist":
        rec_items_by_preds = None
    else:
        rec_items_by_preds = first_stage_model.recommend(user_id_tensor,
                                                         remove_items=np.array(train_val_item_ids),
                                                         n=data.n_item - len(train_val_item_ids))

    if score_distribution_type == "discrete":
        if first_stage_model_name == "SASRecDist":
            all_item_ids = np.arange(data.n_item, dtype=np.int64)
            item_score_distributions = generate_score_distributions(sasrec_model_outputs,
                                                                    sasrec_rating_head,
                                                                    user_id, all_item_ids,
                                                                    device)
        else:
            item_score_distributions = (
                first_stage_model.predict_items_distribution_for_user(user_id_tensor).to(device)
            )

        user_ranking_task = UserRankingTask(user_id, train_val_item_ids, user_test_data,
                                            rec_items_by_preds, item_score_distributions,
                                            score_values, top_k_max_size)
    elif score_distribution_type == "normal":
        item_means, item_stds = first_stage_model.predict_items_mean_and_var_for_user(user_id_tensor)
        item_score_distributions = (
            create_items_discrete_distribution_from_normal_distribution(
                item_means, item_stds, dataset_raw_score_values,
                truncate_continuous_distribution, prob_over_scores
            )
        ).to(device)

        user_ranking_task = UserRankingTask(user_id, train_val_item_ids, user_test_data,
                                            rec_items_by_preds, item_score_distributions, score_values,
                                            top_k_max_size=top_k_max_size,
                                            item_means=item_means, item_stds=item_stds,
                                            raw_score_values_for_rebuild=dataset_raw_score_values,
                                            truncate_continuous_distribution=truncate_continuous_distribution,
                                            prob_over_scores=prob_over_scores)
    else:
        raise ValueError(f"Invalid score distribution type: {score_distribution_type}")

    user_candidate_item_ids = candidate_items_per_user[user_id]
    user_ranking_task.set_candidate_items(user_candidate_item_ids, add_prob_to_score)

    return user_ranking_task



def run_stopping_from_saved_results(seed=0, dataset="ml-25m", first_stage_model_name="CPMF",
                                    sasrec_model_outputs_path=None,
                                    experiment_dir=None, results_file_name=None,
                                    add_prob_to_score=False, truncate_continuous_distribution=True,
                                    prob_over_scores=True, approaches=None, top_k_max_size=None,
                                    stopping_rule="full_prefix", write_to_csv=True):
    set_seed(seed)

    scores_type = "scores_with_prob" if add_prob_to_score else "equal_scores"
    fs_models_and_data_path = f"{FIRST_STAGE_MODELS_AND_DATA_MAIN_PATH}{dataset}/seed={seed}/"

    sasrec_model_outputs = None
    sasrec_rating_head = None
    sasrec_history_mode = None
    if first_stage_model_name == "SASRecDist":
        if sasrec_model_outputs_path is None:
            raise ValueError("SASRecDist requires sasrec_model_outputs_path")

        data = load_data(fs_models_and_data_path)
        first_stage_model = None
        with open(sasrec_model_outputs_path, "rb") as f:
            sasrec_model_outputs = pickle.load(f)

        sasrec_rating_head = load_exported_rating_head(sasrec_model_outputs, device)
        sasrec_history_mode = sasrec_model_outputs["history_mode"]
        if sasrec_history_mode not in {"train", "train_val"}:
            raise ValueError(f"Invalid SASRec history mode: {sasrec_history_mode}")

        print(f"SASRec loaded with history mode: {sasrec_history_mode}")
    else:
        data, first_stage_model = load_first_stage_model_and_data(fs_models_and_data_path,
                                                                  first_stage_model_name)

    results_experiment_dir = Path(experiment_dir)
    if first_stage_model_name == "SASRecDist":
        results_experiment_dir = results_experiment_dir / f"{sasrec_history_mode}_history"

    results_df = get_top_k_res_from_file(results_experiment_dir, results_file_name,
                                         approaches=approaches, top_k_max_size=top_k_max_size)

    candidate_items_per_user = load_candidate_items_per_user(experiment_dir)

    unc_thresholds_by_percentiles = None

    # Understand the score distribution type and prepare the score values accordingly
    score_distribution_type = MODELS_SCORE_DISTRIBUTION_TYPE[first_stage_model_name]
    dataset_raw_score_values = np.array(POSSIBLE_SCORES[dataset])
    if score_distribution_type == "discrete":
        if first_stage_model_name == "SASRecDist":
            exported_score_values = np.asarray(sasrec_model_outputs["score_labels"])
            expected_score_values = np.asarray(POSSIBLE_SCORES[dataset])

            if not np.array_equal(exported_score_values, expected_score_values):
                raise ValueError(
                    "The SASRec score labels do not match "
                    "POSSIBLE_SCORES for the dataset. "
                    f"Exported: {exported_score_values.tolist()}, "
                    f"expected: {expected_score_values.tolist()}"
                )

            score_values = torch.as_tensor(exported_score_values)
        else:
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


    stopping_results_rows = []
    for row_idx, row in results_df.iterrows():
        start_time = time.time()

        user_id = row["user_id"]
        adaptive_approach = row["approach"]
        full_result = list(row["top_k_result"])

        base_user_ranking_task = build_base_user_ranking_task(user_id, data, first_stage_model,
                                                              first_stage_model_name,
                                                              sasrec_model_outputs,
                                                              sasrec_rating_head,
                                                              score_distribution_type,
                                                              score_values,
                                                              dataset_raw_score_values,
                                                              truncate_continuous_distribution,
                                                              prob_over_scores,
                                                              candidate_items_per_user,
                                                              add_prob_to_score,
                                                              top_k_max_size)

        stopping_task = copy.deepcopy(base_user_ranking_task)
        stopping_task.run_rank_dist(scores_type, rd_by_scores=False)

        full_result_item_idxs = [stopping_task.item_ids_to_item_idxs[item_id]
                                 for item_id in full_result]

        # Compute the exact Precision_C at each t
        prefix_precision_c = []
        for prefix_size in range(1, top_k_max_size + 1):
            metric_results = stopping_task.compute_exact_metrics(full_result[:prefix_size])
            prefix_precision_c.append(metric_results["precision_c"])

        stop_k = top_k_max_size
        stop_found = False
        prefix_expected_precision_c = []
        deltas = []
        for t in range(2, top_k_max_size + 1): # Examine whether we want to add the t item, and get a list of size t
            last_item_id = full_result[t - 2]
            # next_cand_item_id = full_result[t - 1]

            if last_item_id in stopping_task.test_item_ids_all_as_list:
                stopping_task.update_score_dist_for_expected_cond_eval(last_item_id)
                stopping_task.run_rank_dist(scores_type, rd_by_scores=False)

            # Compute expected precision at step t-1 and t, and the delta, based on the current rank distribution
            rd_log_result = stopping_task.rank_dist_log_result.copy()
            cumm_rd_until_prev_step = np.logaddexp.reduce(rd_log_result[:, :t - 1], axis=1)
            rd_next_step = rd_log_result[:, t - 1]

            curr_expected_precision = np.exp(cumm_rd_until_prev_step[full_result_item_idxs[:t-1]]).mean()
            prefix_expected_precision_c.append(curr_expected_precision)

            next_step_expected_precision_elems = np.logaddexp.reduce(
                [cumm_rd_until_prev_step[full_result_item_idxs[:t-1]],
                 rd_next_step[full_result_item_idxs[:t-1]]], axis=0
            )
            next_step_expected_precision_elems = np.append(
                next_step_expected_precision_elems,
                np.logaddexp.reduce(
                    [cumm_rd_until_prev_step[full_result_item_idxs[t-1]],
                     rd_next_step[full_result_item_idxs[t-1]]], axis=0
                )
            )
            next_step_expected_precision = np.exp(next_step_expected_precision_elems).mean()

            if stopping_rule == "full_prefix":
                delta = next_step_expected_precision - curr_expected_precision

            elif stopping_rule == "marginal_item":
                marginal_item_expected_precision = np.exp(next_step_expected_precision_elems[-1])
                delta = marginal_item_expected_precision - curr_expected_precision
            else:
                raise ValueError(f"Invalid stopping rule: {stopping_rule}")
            stop_here = delta <= 0
            deltas.append(delta)

            if stop_here and not stop_found:
                stop_k = t - 1
                stop_found = True

        k_oracle = int(np.argmax(prefix_precision_c)) + 1
        precision_c_stop = prefix_precision_c[stop_k - 1]
        precision_c_oracle = prefix_precision_c[k_oracle - 1]
        precision_c_kmax = prefix_precision_c[top_k_max_size - 1]

        stopping_results_rows.append({
            "user_id": user_id,
            "approach": adaptive_approach,
            "full_result": full_result,
            "precision_c": prefix_precision_c,
            "expected_precision_c": prefix_expected_precision_c,
            "deltas": deltas,
            "k_stop": stop_k,
            "precision_c_at_k_stop": precision_c_stop,
            "k_oracle": k_oracle,
            "precision_c_at_k_oracle": precision_c_oracle,
            "precision_c_gap_to_oracle": precision_c_oracle - precision_c_stop,
            "precision_c_at_k_max": precision_c_kmax,
            "improvement_vs_k_max": precision_c_stop - precision_c_kmax,
            "oracle_minus_k_max": precision_c_oracle - precision_c_kmax,
        })

        print(
            f"  stop_k={stop_k}, oracle_k={k_oracle}, "
            f"precision_c@stop={round(precision_c_stop, 4)}, "
            f"precision_c@oracle={round(precision_c_oracle, 4)}, "
            f"precision_c@kmax={round(precision_c_kmax, 4)}"
        )
        print(f"User {user_id} evaluation finished in {round(time.time() - start_time, 4)}")
        print("*" * 25)

    stopping_results_df = pd.DataFrame(stopping_results_rows)

    print("\n========== OVERALL SUMMARY ==========")
    print("avg k_stop:", round(stopping_results_df["k_stop"].mean(), 4))
    print("avg precision_c_at_k_stop:", round(stopping_results_df["precision_c_at_k_stop"].mean(), 4))
    print("avg precision_c_at_k_oracle:", round(stopping_results_df["precision_c_at_k_oracle"].mean(), 4))
    print("avg precision_c_at_k_max:", round(stopping_results_df["precision_c_at_k_max"].mean(), 4))
    print("avg gap to oracle:", round(stopping_results_df["precision_c_gap_to_oracle"].mean(), 6))
    print("avg improvement vs k_max:", round(stopping_results_df["improvement_vs_k_max"].mean(), 6))

    if write_to_csv:
        stopping_results_path = (results_experiment_dir / "results" /
                                 f"{results_file_name}_stopping_{stopping_rule}.csv")
        stopping_results_df.to_csv(stopping_results_path, index=False)
        print(f"Saved summary results to: {stopping_results_path}")



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default="ml-25m")
    parser.add_argument('--fs_model_name', type=str, default="CPMF")

    parser.add_argument('--experiment_dir', type=str, default=None)
    parser.add_argument('--results_file_name', type=str, required=True)
    parser.add_argument('--approaches', nargs='+', default=ADAPTIVE_APPROACHES,
                        choices=ADAPTIVE_APPROACHES)
    parser.add_argument('--top_k_max_size', type=int, default=20)

    # Arg for loading saved SASRec model
    parser.add_argument("--sasrec_model_outputs_path", type=str, default=None)

    parser.add_argument('--stopping_rule', type=str, default="full_prefix",
                        choices=["full_prefix", "marginal_item"])

    args = parser.parse_args()

    run_stopping_from_saved_results(
        seed=args.seed,
        dataset=args.dataset_name,
        first_stage_model_name=args.fs_model_name,
        sasrec_model_outputs_path=args.sasrec_model_outputs_path,
        experiment_dir=args.experiment_dir,
        results_file_name=args.results_file_name,
        add_prob_to_score=False,
        truncate_continuous_distribution=True,
        prob_over_scores=True,
        approaches=args.approaches,
        top_k_max_size=args.top_k_max_size,
        stopping_rule=args.stopping_rule,
        write_to_csv=True,
    )


if __name__ == '__main__':
    main()

