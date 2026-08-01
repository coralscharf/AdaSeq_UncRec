from enum import Enum
import pandas as pd
from scipy.stats import norm, truncnorm
import pickle
import csv
from pathlib import Path
import random

from probability_computation.uncertain_SnapExplicit.uncertain.explicit import *
from constants import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Distribution(Enum):
    UNIFORM = 'uniform'


def generate_universe(universe_size=10, n_scores=3, distribution=Distribution.UNIFORM,
                      random_state=0, *args, **kwargs):
    if distribution == Distribution.UNIFORM:
        return _generate_uniform(universe_size, n_scores, random_state)


def _generate_uniform(universe_size, n_scores, random_state):
    old_seed = torch.seed()
    torch.manual_seed(random_state)
    score_probabilities = torch.rand(universe_size * n_scores, dtype=torch.double).reshape(universe_size, -1)
    score_probabilities = score_probabilities / (score_probabilities.sum(axis=1).reshape(-1, 1))
    score_values = torch.rand(universe_size * n_scores, dtype=torch.double).reshape(universe_size, -1) * 100
    torch.manual_seed(old_seed)
    return score_probabilities, score_values


def load_data(fs_models_and_data_path):
    with open(f'{fs_models_and_data_path}data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f'Data prepared: {data.n_user} users, {data.n_item} items.')
    print(f'{len(data.train)} train, {len(data.val)} validation and {len(data.test)} test interactions.')
    return data


def load_first_stage_model_and_data(fs_models_and_data_path, first_stage_model_name):

    data = load_data(fs_models_and_data_path)

    score_labels = pd.factorize(data.train[:, 2], sort=True)[1]

    trained_models_path = f"{fs_models_and_data_path}/checkpoints/{first_stage_model_name.lower()}/"
    files = {file: float(file.split('val_loss=')[1][:-5]) for file in os.listdir(trained_models_path)}

    if first_stage_model_name == "BeMF":
        first_stage_model = BeMF(data.n_user, data.n_item, score_labels=score_labels, embedding_dim=20)
        first_stage_model = first_stage_model.load_from_checkpoint(os.path.join(trained_models_path,
                                                                                min(files, key=files.get)))
    elif first_stage_model_name == "OrdRec":
        first_stage_model = OrdRec(data.n_user, data.n_item, score_labels=score_labels, embedding_dim=0)
        first_stage_model = first_stage_model.load_from_checkpoint(os.path.join(trained_models_path,
                                                                                min(files, key=files.get)))
    elif first_stage_model_name == "CPMF" or first_stage_model_name == "cpmf_with_sasrec":
        first_stage_model = CPMF(data.n_user, data.n_item, embedding_dim=0, lr=0, weight_decay=0)
        first_stage_model = first_stage_model.load_from_checkpoint(os.path.join(trained_models_path,
                                                                                min(files, key=files.get)))
    else:
        raise ValueError(f'{first_stage_model_name} model for first stage is invalid')
    first_stage_model.to(device)
    return data, first_stage_model


def create_items_discrete_distribution_from_normal_distribution(item_means,
                                                                item_stds,
                                                                score_values,
                                                                truncate_distribution=True,
                                                                return_prob_over_scores=False):
    if truncate_distribution:
        min_score, max_score = min(score_values), max(score_values)

        if return_prob_over_scores:
            score_values = np.asarray(score_values, dtype=float)
            step = float(score_values[1] - score_values[0])

            low_edge_score = min_score - step / 2.0
            high_edge_score = max_score + step / 2.0
            midpoints = score_values[:-1] + step / 2.0
            score_edges = np.concatenate(([low_edge_score], midpoints, [high_edge_score]))
        else: # Return probabilities over bins created by score_values
            low_edge_score = min_score
            high_edge_score = max_score
            score_edges = score_values

        items_a_values = (low_edge_score - item_means[:, np.newaxis]) / item_stds[:, np.newaxis]
        items_b_values = (high_edge_score - item_means[:, np.newaxis]) / item_stds[:, np.newaxis]
        edges_cdf_values = truncnorm.cdf(score_edges, items_a_values, items_b_values,
                                         loc=item_means[:, np.newaxis], scale=item_stds[:, np.newaxis])

        score_probabilities = np.diff(edges_cdf_values, axis=1)
    else:
        scores_cdf_values = norm.cdf(score_values,
                                     loc=item_means[:, np.newaxis],
                                     scale=item_stds[:, np.newaxis])
        score_probabilities = np.diff(scores_cdf_values)
        score_probabilities = np.concatenate((scores_cdf_values[:, 0][:, np.newaxis],
                                              score_probabilities), axis=1)
        score_probabilities = np.concatenate((score_probabilities,
                                              (1 - scores_cdf_values[:, -1])[:, np.newaxis]), axis=1)
    return torch.tensor(score_probabilities)


#
# def create_items_discrete_distribution_from_normal_distribution(item_means,
#                                                                 item_stds,
#                                                                 score_values,
#                                                                 truncate_distribution=True):
#     if truncate_distribution:
#         min_score, max_score = min(score_values), max(score_values)
#         items_a_values = (min_score - item_means[:, np.newaxis]) / item_stds[:, np.newaxis]
#         items_b_values = (max_score - item_means[:, np.newaxis]) / item_stds[:, np.newaxis]
#         scores_cdf_values = truncnorm.cdf(score_values, items_a_values, items_b_values,
#                                           loc=item_means[:, np.newaxis],
#                                           scale=item_stds[:, np.newaxis])
#         score_probabilities = np.diff(scores_cdf_values)
#     else:
#         scores_cdf_values = norm.cdf(score_values,
#                                      loc=item_means[:, np.newaxis],
#                                      scale=item_stds[:, np.newaxis])
#         score_probabilities = np.diff(scores_cdf_values)
#         score_probabilities = np.concatenate((scores_cdf_values[:, 0][:, np.newaxis],
#                                               score_probabilities), axis=1)
#         score_probabilities = np.concatenate((score_probabilities,
#                                               (1 - scores_cdf_values[:, -1])[:, np.newaxis]), axis=1)
#     return torch.tensor(score_probabilities)


def calculate_log_probability_of_being_first(universe_score_probabilities: torch.Tensor,
                                             universe_score_values: torch.Tensor,
                                             scores, device='cpu'):
    """
    Calculates probability of each score from a given list being the first in a universe
    :param universe_score_probabilities: array of universe score probabilities (shape |U|*n_scores)
    :param universe_score_values: array of universe score values (shape |U|*n_scores)
    :param scores: list of candidate scores
    :return: dict {score:log probability}
    """
    log_probabilities = torch.zeros(len(scores)).to(device)
    for score_probabilities, score_values in zip(universe_score_probabilities, universe_score_values):
        lower_mask = score_values.reshape(-1, 1) < scores
        equal_mask = score_values.reshape(-1, 1) == scores
        lower_probabilities = score_probabilities[None, :].mm(lower_mask.float())
        lower_probabilities += score_probabilities[None, :].mm(equal_mask.float())/2 # random tie-breaking

        log_probabilities += torch.log(lower_probabilities[0])

    log_probabilities_dict = {score.item(): probability.item() for score, probability in zip(scores, log_probabilities)}
    return log_probabilities_dict



def build_user_test_objects(user_test_data):
    user_test_df = pd.DataFrame(user_test_data, columns=["item", "score"])
    user_test_df = user_test_df.drop_duplicates(subset=["item"], keep="last")
    user_test_data = user_test_df.to_numpy()

    user_test_objects = {}

    # All test items
    user_test_objects["user_test_data"] = user_test_data
    user_test_objects["scores_by_test_items"] = {
        int(item): float(score)
        for item, score in user_test_data
    }
    user_test_objects["test_item_ids_all"] = torch.tensor(user_test_data[:, 0]).int()
    user_test_objects["test_item_ids_all_as_list"] = user_test_objects["test_item_ids_all"].tolist()

    # Relevant test items with score >= 4
    rel_test_item_ids = torch.tensor(user_test_data[user_test_data[:, 1] >= 4, 0]).int()
    user_test_objects["rel_test_item_ids"] = rel_test_item_ids
    user_test_objects["rel_test_item_ids_as_list"] = rel_test_item_ids.tolist()

    # Sort test items by score
    user_test_data_sorted = user_test_data[user_test_data[:, 1].argsort()]
    user_test_objects["user_test_data_sorted"] = user_test_data_sorted
    user_test_objects["test_scores_ranked"] = user_test_data_sorted[::-1, 1].astype(float).tolist()

    user_test_objects["rel_test_scores_ranked"] = sorted([float(score)
                                                          for score in user_test_data[:, 1]
                                                          if score >= 4], reverse=True)

    # Item groups ordered from highest to lowest rating
    if len(user_test_data_sorted) == 0:
        test_item_ids_grouped = []
    else:
        test_item_ids_grouped = np.split(user_test_data_sorted[:, 0],
                                         np.unique(user_test_data_sorted[:, 1],
                                                   return_index=True)[1][1:])
        test_item_ids_grouped.reverse()
    user_test_objects["test_item_ids_grouped"] = test_item_ids_grouped

    return user_test_objects


def compute_exact_metrics_for_user(top_k_result, user_test_objects):
    test_item_ids_grouped = user_test_objects["test_item_ids_grouped"]
    rel_test_item_ids = user_test_objects["rel_test_item_ids_as_list"]

    metric_results = {}
    metric_results["precision_l"] = compute_precision(top_k_result,
                                                      rel_test_item_ids,
                                                      test_item_ids_grouped,
                                                      reference="all_test_items")

    metric_results["precision_c"] = compute_precision(top_k_result,
                                                      rel_test_item_ids,
                                                      test_item_ids_grouped,
                                                      reference="k_ranked_test_items")

    metric_results["recall"] = compute_recall(top_k_result, rel_test_item_ids)

    metric_results.update(
        compute_dcg_and_ndcg_for_user(top_k_result, user_test_objects)
    )

    return metric_results


def compute_stopping_metrics_for_user(top_k_result, user_test_objects,
                                      dcgu_configs=(("linear", 3.0), ("linear", 4.0))):
    exact_metrics = compute_exact_metrics_for_user(top_k_result, user_test_objects)

    pdcg = compute_pdcg_for_result(
        result=top_k_result,
        relevant_items=user_test_objects["rel_test_item_ids_as_list"]
    )

    metric_results = {
        "precision_c": exact_metrics["precision_c"],
        "recall": exact_metrics["recall"],
        "ndcg_l": exact_metrics["ndcg_l"],
        "ndcg_c": exact_metrics["ndcg_c"],
        "pdcg": pdcg
    }

    scores_by_items = user_test_objects["scores_by_test_items"]

    for gain_type, effort_rating in dcgu_configs:
        effort_name = f"{effort_rating:g}".replace(".", "_")
        metric_name = f"dcgu_{gain_type}_er{effort_name}"

        metric_results[metric_name] = compute_explicit_dcgu_for_result(
            result=top_k_result, scores_by_items=scores_by_items,
            effort=effort_rating, gain_type=gain_type,
            effort_is_rating=True
        )

    return metric_results


def compute_dcg_and_ndcg_for_user(top_k_result, user_test_objects):
    scores_by_test_items = user_test_objects["scores_by_test_items"]
    test_item_ids_grouped = user_test_objects["test_item_ids_grouped"]

    dcg_l = compute_dcg_for_user(scores_by_test_items,
                                 test_item_ids_grouped,
                                 top_k_result,
                                 dcg_version="liberal")
    dcg_c = compute_dcg_for_user(scores_by_test_items,
                                 test_item_ids_grouped,
                                 top_k_result,
                                 dcg_version="conservative")

    ideal_dcg_l = compute_dcg_for_user(
        scores_by_test_items, test_item_ids_grouped,
        top_k_result, dcg_version="liberal", ideal_dcg=True,
        user_test_scores_ranked=user_test_objects["rel_test_scores_ranked"]
    )
    ideal_dcg_c = compute_dcg_for_user(
        scores_by_test_items, test_item_ids_grouped,
        top_k_result, dcg_version="conservative", ideal_dcg=True,
        user_test_scores_ranked=user_test_objects["test_scores_ranked"]
    )

    ndcg_l = dcg_l / ideal_dcg_l if ideal_dcg_l > 0 else 0.0
    ndcg_c = dcg_c / ideal_dcg_c if ideal_dcg_c > 0 else 0.0

    return {"dcg_l": dcg_l, "dcg_c": dcg_c, "ndcg_l": ndcg_l, "ndcg_c": ndcg_c}



def compute_dcg_for_user(user_scores_by_test_items, user_test_item_ids_grouped,
                         top_k_result, dcg_version='liberal',
                         ideal_dcg=False, user_test_scores_ranked=None):
    test_items_scores = user_scores_by_test_items
    if ideal_dcg:
        user_scores = user_test_scores_ranked[:len(top_k_result)]
    else:
        if dcg_version == 'conservative':
            top_k_items_in_list = check_if_item_is_in_cons_list(top_k_result,
                                                                user_test_item_ids_grouped)
            user_scores = [test_items_scores[item]
                           if (is_valid == 1 and item in test_items_scores) else 0
                           for item, is_valid in zip(top_k_result, top_k_items_in_list)]
        else:  # dcg_version == 'liberal'
            user_scores = [test_items_scores[item]
                           if item in test_items_scores and test_items_scores[item] >= 4.0 else 0
                           for item in top_k_result]

    return np.sum([(2 ** score - 1) / np.log2(rank + 2)
                   for rank, score in enumerate(user_scores)])


def compute_implicit_dcgu_for_result(result, relevant_items, epsilon=0.05):
    if not 0 <= epsilon < 1:
        raise ValueError("epsilon must be in [0, 1).")

    if len(result) == 0:
        return 0.0

    relevant_items = set(relevant_items)
    relevance = np.asarray([float(item in relevant_items) for item in result],
                           dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, len(result) + 2))

    return float(np.sum((relevance - epsilon) * discounts))


def compute_pdcg_for_result(result, relevant_items):
    """
    Compute penalized DCG using:
        +1 for a relevant item (observed test rating >= 4)
        -1 for a non-relevant item
    """
    if len(result) == 0:
        return 0.0

    relevant_items = set(relevant_items)
    gains = np.asarray([1.0 if item in relevant_items else -1.0
                        for item in result], dtype=float)
    discounts = 1.0 / np.log2(
        np.arange(2, len(result) + 2)
    )

    return float(np.sum(gains * discounts))


def compute_explicit_dcgu_for_result(result, scores_by_items, effort,
                                     gain_type="linear", effort_is_rating=True):
    """
       Compute DCGU: sum_i (gain_i - effort_gain) / log2(i + 1)
        gain_type:
           "linear":      gain(r) = r
           "exponential": gain(r) = 2^r - 1
       Items without observed test feedback receive gain 0.
   """
    if effort < 0:
        raise ValueError("effort must be non-negative.")

    if gain_type == "linear":
        gain_func = lambda rating: float(rating)
    elif gain_type == "exponential":
        gain_func = lambda rating: 2 ** float(rating) - 1
    else:
        raise ValueError("gain_type must be either 'linear' or 'exponential'.")

    if len(result) == 0:
        return 0.0

    effort_gain = (gain_func(effort) if effort_is_rating else float(effort))

    gains = np.asarray([gain_func(scores_by_items[item])
                        if item in scores_by_items else 0.0
                        for item in result], dtype=float)

    discounts = 1.0 / np.log2(np.arange(2, len(result) + 2))

    return float(np.sum((gains - effort_gain) * discounts))


def check_if_item_is_in_cons_list(top_k_result, ground_truth_items_grouped):
    # Marks recommended items that belong to the conservative top-k reference.
    top_k_items_in_list = [0 for _ in top_k_result]
    remaining_slots = len(top_k_result)

    for gt_items_by_score in ground_truth_items_grouped:
        if remaining_slots <= 0:
            break

        group_items = set(np.asarray(gt_items_by_score).astype(int))
        group_limit = min(len(group_items), remaining_slots)

        marked_in_group = 0
        for item_idx, item in enumerate(top_k_result):
            if marked_in_group >= group_limit:
                break

            if item in group_items:
                top_k_items_in_list[item_idx] = 1
                marked_in_group += 1

        remaining_slots -= group_limit

    return top_k_items_in_list



def compute_precision(top_k_result, ground_truth_items, ground_truth_items_grouped,
                      reference="all_test_items"):
    ground_truth_items = set(ground_truth_items)
    top_k_result = set(top_k_result)
    k = len(top_k_result)
    if reference == "all_test_items":
        total_common_items = len(top_k_result.intersection(ground_truth_items))
    elif reference == "k_ranked_test_items":
        items_to_pass = k
        total_common_items = 0
        for gt_items_by_score in ground_truth_items_grouped:
            if items_to_pass <= 0:
                break
            set_gt_items_by_score = set(gt_items_by_score.astype(int))
            common_items = len(top_k_result.intersection(set_gt_items_by_score))
            if common_items > items_to_pass:
                common_items = items_to_pass
            total_common_items += common_items
            items_to_pass -= len(gt_items_by_score)
    else:
        raise ValueError(f'The option {reference} for reference items in computing precision is invalid')
    if k == 0:
        return 0
    return round(total_common_items / k, 6)


def compute_recall(top_k_result, ground_truth_items):
    ground_truth_items = set(ground_truth_items)
    top_k_result = set(top_k_result)
    total_common_items = len(top_k_result.intersection(ground_truth_items))
    if len(ground_truth_items) == 0:
        return 0
    return round(total_common_items / len(ground_truth_items), 4)



def write_results_to_csv(results_path, header_results_file, results_rows,
                         results_file_name="results.csv"):
    results_file_path = Path(results_path) / results_file_name

    with open(results_file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header_results_file)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(results_rows)

