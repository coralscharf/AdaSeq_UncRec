import math
from rank_dist import RankDist
from utils import *


class UserRankingTask:
    def __init__(self, user_id, train_val_item_ids, user_test_data, rec_items_by_preds,
                 item_score_distributions, score_values, top_k_max_size=20,
                 unc_thresholds_by_percentiles=None, item_means=None, item_stds=None,
                 score_update_cfg=None, full_item_embd_norm=None, full_item_embd = None,
                 user_embedding=None, raw_score_values_for_rebuild=None,
                 truncate_continuous_distribution=True, prob_over_scores=True):

        self.user_id = user_id
        self.train_val_item_ids = train_val_item_ids

        self.full_user_test_data = user_test_data
        self.set_user_test_data(user_test_data)

        self.score_values = score_values

        self.rec_items_by_preds = rec_items_by_preds
        self.item_score_distributions = item_score_distributions
        self.item_ids = torch.tensor(range(len(self.item_score_distributions))).to(device)
        self.item_means = item_means
        self.item_stds = item_stds
        self.item_score_values = None
        self.item_idxs_to_item_ids = None
        self.item_ids_to_item_idxs = None

        self.unc_thresholds_by_percentiles = unc_thresholds_by_percentiles
        self.top_k_max_size = top_k_max_size

        # self.rank_dist_log_result = None

        # Variables for score distributions update
        self.score_update_cfg = score_update_cfg
        self.full_item_embd_norm = full_item_embd_norm
        self.item_embd_norm_current = None
        self.full_item_embd = full_item_embd
        self.item_embd_current = None

        self.user_embedding = None if user_embedding is None else user_embedding.clone()
        self.user_embedding_init = None if user_embedding is None else user_embedding.clone() # For regularization if needed

        self.raw_score_values_for_rebuild = raw_score_values_for_rebuild
        self.truncate_continuous_distribution = truncate_continuous_distribution
        self.prob_over_scores = prob_over_scores


    def set_user_test_data(self, user_test_data):
        self.user_test_objects = build_user_test_objects(user_test_data)
        for key, value in self.user_test_objects.items():
            setattr(self, key, value)


    def refresh_current_universe_metadata(self, add_prob_to_score=False):
        self.item_idxs_to_item_ids = {item_idx: item_id.item()
                                      for item_idx, item_id in enumerate(self.item_ids)}
        self.item_ids_to_item_idxs = {item_id.item(): item_idx for item_idx, item_id in enumerate(self.item_ids)}

        self.item_score_values = self.score_values.repeat(len(self.item_score_distributions), 1).to(device)

        if add_prob_to_score:
            self.item_score_values = self.item_score_values + self.item_score_distributions

        if self.full_item_embd_norm is not None:
            self.item_embd_norm_current = self.full_item_embd_norm[self.item_ids.long()]
        if self.full_item_embd is not None:
            self.item_embd_current = self.full_item_embd[self.item_ids.long()]

    def compute_item_expected_scores(self):
        if self.item_means is None:
            return torch.sum(self.item_score_distributions * self.score_values, dim=1)
        else:
            return torch.tensor(self.item_means)

    def compute_item_prr_scores(self):
        if self.item_means is None:
            return torch.sum(self.item_score_distributions[:, self.score_values >= PRR_THRESHOLD], dim=1)
        else:
            return torch.tensor(norm.sf(PRR_THRESHOLD, self.item_means, self.item_stds))

    def get_item_ids_after_ubf(self, percentile, result_size):
        if self.rec_items_by_preds is None:
            raise ValueError("uncertainty_based_filtering requires rec_items_by_preds")
        if percentile == 1.0:
            return self.rec_items_by_preds[:result_size].index.tolist()
        else:
            unc_threshold = self.unc_thresholds_by_percentiles[
                np.where(PERCENTILES_FOR_UBF == percentile)[0].item()]
            item_uncertainties_rec_by_preds = self.rec_items_by_preds.uncertainties.to_numpy()
            return self.rec_items_by_preds[item_uncertainties_rec_by_preds <
                                           unc_threshold][:result_size].index.tolist()

    def generate_candidate_items_set(self, add_prob_to_score, candidate_set_cfg):
        mask = torch.ones(len(self.item_ids), dtype=torch.bool)
        mask[self.train_val_item_ids] = False
        self.item_score_distributions = self.item_score_distributions[mask]
        self.item_ids = self.item_ids[mask]
        if self.item_means is not None:
            self.item_means = self.item_means[mask]
            self.item_stds = self.item_stds[mask]

        candidate_set_selection_method = candidate_set_cfg["selection_method"]
        candidate_set_fixed_size = candidate_set_cfg["fixed_size"]
        negatives_per_positive = candidate_set_cfg["negatives_per_positive"]
        candidate_set_max_size = candidate_set_cfg["max_size"]

        if candidate_set_selection_method is not None:
            if candidate_set_selection_method.startswith("reranking"):
                reranking_method = candidate_set_selection_method.split("reranking-")[1]
                if reranking_method.startswith("uncertainty_based_filtering"):
                    percentile = float(reranking_method.split("-")[1])
                    target_item_ids = torch.tensor(
                        self.get_item_ids_after_ubf(percentile, candidate_set_fixed_size)
                    ).to(device)
                    target_item_ids_indices = torch.nonzero(self.item_ids.unsqueeze(1) ==
                                                            target_item_ids.unsqueeze(0))[:, 0]
                else:
                    if reranking_method == "probability_of_relevance_ranking":
                        items_rr_scores = self.compute_item_prr_scores()
                    elif reranking_method == "expected_score":
                        items_rr_scores = self.compute_item_expected_scores()
                    else:
                        raise ValueError(f'Invalid reranking method')
                    target_item_ids_indices = torch.argsort(items_rr_scores, descending=True)[:candidate_set_fixed_size]

            elif candidate_set_selection_method == "pos_with_neg_sampling":
                if negatives_per_positive > 0:
                    mask = torch.ones(len(self.item_ids), dtype=torch.bool)
                    test_items_indices_in_item_ids = torch.nonzero(self.item_ids.detach().cpu().unsqueeze(1) ==
                                                                   self.test_item_ids_all.unsqueeze(0))[:, 0]
                    mask[test_items_indices_in_item_ids] = False
                    not_test_item_ids = self.item_ids[mask].detach().cpu()
                    neg_items_number_to_sample = max(0,
                                                     min(negatives_per_positive * len(self.test_item_ids_all),
                                                         candidate_set_max_size - len(self.test_item_ids_all))
                                                     )
                    not_test_indices_samples = torch.randperm(len(not_test_item_ids),
                                                              device=not_test_item_ids.device)[:neg_items_number_to_sample]
                    target_item_ids = torch.cat((not_test_item_ids[not_test_indices_samples], self.test_item_ids_all)).int()
                else:
                    target_item_ids = self.test_item_ids_all

                target_item_ids_indices = torch.nonzero(self.item_ids.detach().cpu().unsqueeze(1) ==
                                                        target_item_ids.unsqueeze(0))[:, 0]
            else:
                raise ValueError(f'{candidate_set_selection_method} for candidate set selection method is invalid')

            self.item_score_distributions = self.item_score_distributions[target_item_ids_indices]
            self.item_ids = self.item_ids[target_item_ids_indices]
            if self.item_means is not None:
                self.item_means = self.item_means[target_item_ids_indices.detach().cpu().numpy()]
                self.item_stds = self.item_stds[target_item_ids_indices.detach().cpu().numpy()]

            if self.rec_items_by_preds is not None:
                self.rec_items_by_preds = self.rec_items_by_preds[
                    self.rec_items_by_preds.index.isin(set(self.item_ids.tolist()))
                ]

        self.refresh_current_universe_metadata(add_prob_to_score=add_prob_to_score)

        # TODO: Consider running update_test_data_by_candidate_items

    def set_candidate_items(self, candidate_item_ids, add_prob_to_score):
        """
        Restrict the current task to a given external candidate set.

        candidate_item_ids: iterable of item ids that should remain in the candidate universe.
            These item ids are assumed to be post-train/val filtering, i.e.,
            they should not include train/val items.
        """
        target_item_ids = torch.tensor(candidate_item_ids).int()

        target_item_ids_indices = torch.nonzero(self.item_ids.detach().cpu().unsqueeze(1) ==
                                                target_item_ids.unsqueeze(0))[:, 0]

        if len(target_item_ids_indices) != len(target_item_ids):
            found_item_ids = set(self.item_ids[target_item_ids_indices].detach().cpu().tolist())
            missing_item_ids = [item_id for item_id in candidate_item_ids
                                if item_id not in found_item_ids]
            raise ValueError(
                f"User {self.user_id}: {len(missing_item_ids)} candidate items were not found "
                f"in the ranking universe. First few: {missing_item_ids[:10]}"
            )

        self.item_score_distributions = self.item_score_distributions[target_item_ids_indices]
        self.item_ids = self.item_ids[target_item_ids_indices]

        if self.item_means is not None:
            self.item_means = self.item_means[target_item_ids_indices.detach().cpu().numpy()]
            self.item_stds = self.item_stds[target_item_ids_indices.detach().cpu().numpy()]

        if self.rec_items_by_preds is not None:
            self.rec_items_by_preds = self.rec_items_by_preds[
                self.rec_items_by_preds.index.isin(set(self.item_ids.tolist()))
            ]

        self.refresh_current_universe_metadata(add_prob_to_score=add_prob_to_score)

        self.update_test_data_by_candidate_items(self.item_ids.detach().cpu().tolist())


    def update_test_data_by_candidate_items(self, candidate_item_ids):
        filtered_user_test_data = np.array([
            row for row in self.full_user_test_data
            if int(row[0]) in candidate_item_ids
        ])
        self.set_user_test_data(filtered_user_test_data)

    def update_uni_and_score_dist_after_rec(self, next_rec_item_id, do_not_change_rd_uni=False):
        item_with_feedback = False
        if next_rec_item_id in self.test_item_ids_all_as_list:
            item_with_feedback = True

        method = self.score_update_cfg["method"]
        if item_with_feedback and method != "no_update":
            next_rec_item_det_score = self.scores_by_test_items[next_rec_item_id]

            if method == "local_update":
                if self.item_means is None:
                    self.apply_similarity_categorical_local_update(next_rec_item_id,
                                                                   next_rec_item_det_score)
                else:
                    self.apply_similarity_mean_local_update(next_rec_item_id,
                                                            next_rec_item_det_score)
            elif method == "global_update":
                if self.item_means is None:
                    raise NotImplementedError("global_update is currently implemented "
                                              "only for normal-score models")

                self.apply_user_embedding_global_update(next_rec_item_id, next_rec_item_det_score)
            else:
                raise ValueError(f"Unsupported score update method: {method}")

            # Gaussian models must rebuild their discretized distributions after updating the means.
            if self.item_means is not None:
                self.item_score_distributions = (
                    create_items_discrete_distribution_from_normal_distribution(
                        self.item_means, self.item_stds, self.raw_score_values_for_rebuild,
                        self.truncate_continuous_distribution, self.prob_over_scores,
                    )
                ).to(device)

        if not do_not_change_rd_uni: # Universe Updating
            keep_mask = (self.item_ids != next_rec_item_id)

            self.item_score_distributions = self.item_score_distributions[keep_mask]
            self.item_ids = self.item_ids[keep_mask]

            if self.item_means is not None:
                keep_indices = keep_mask.detach().cpu().numpy()
                self.item_means = self.item_means[keep_indices]
                self.item_stds = self.item_stds[keep_indices]

            if self.rec_items_by_preds is not None:
                self.rec_items_by_preds = self.rec_items_by_preds[
                    self.rec_items_by_preds.index.isin(set(self.item_ids.detach().cpu().tolist()))
                ]

            self.refresh_current_universe_metadata(add_prob_to_score=False)

    def update_score_dist_for_expected_cond_eval(self, rec_item_id):
        next_rec_item_det_score = self.scores_by_test_items[rec_item_id]
        item_idx = self.item_ids_to_item_idxs[rec_item_id]
        deterministic_score_distribution = torch.zeros_like(
            self.item_score_distributions[item_idx]
        )
        score_index = (self.score_values == next_rec_item_det_score).nonzero(as_tuple=True)[0]
        deterministic_score_distribution[score_index] = 1.0
        self.item_score_distributions[item_idx] = deterministic_score_distribution


    def apply_similarity_mean_local_update(self, anchor_item_id, anchor_true_score):
        anchor_idx = self.item_ids_to_item_idxs[anchor_item_id]
        anchor_pred_mean = self.item_means[anchor_idx]
        error = anchor_true_score - anchor_pred_mean

        top_l = self.score_update_cfg["top_l"]
        alpha = self.score_update_cfg["alpha"]
        positive_only = self.score_update_cfg["local_update_pos_only"]

        anchor_vec = self.item_embd_norm_current[anchor_idx]
        sims = self.item_embd_norm_current @ anchor_vec

        sims[anchor_idx] = -float("inf")

        if positive_only:
            valid_mask = sims > 0
            if valid_mask.sum().item() == 0:
                return
            valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
            valid_sims = sims[valid_mask]
            k = min(top_l, len(valid_sims))
            top_vals, top_pos = torch.topk(valid_sims, k=k, largest=True)
            neighbor_indices = valid_indices[top_pos]
            sims_selected = top_vals
        else:
            available = len(sims) - 1
            if available <= 0:
                return
            k = min(top_l, available)
            sims_selected, neighbor_indices = torch.topk(sims, k=k, largest=True)

        neighbor_indices_np = neighbor_indices.detach().cpu().numpy()
        sims_selected_np = sims_selected.detach().cpu().numpy()

        self.item_means[neighbor_indices_np] += alpha * sims_selected_np * error

        min_score = float(np.min(self.raw_score_values_for_rebuild))
        max_score = float(np.max(self.raw_score_values_for_rebuild))
        self.item_means = np.clip(self.item_means, min_score, max_score)


    def get_global_update_error(self, anchor_idx, anchor_true_score):
        anchor_pred_mean = self.item_means[anchor_idx]
        mean_error = anchor_true_score - anchor_pred_mean

        error_type = self.score_update_cfg.get("global_update_error_type", "mean_error")
        if error_type == "mean_error":
            return mean_error
        elif error_type == "uncertainty_scaled_error":
            anchor_std = float(self.item_stds[anchor_idx])
            variance = anchor_std ** 2
            return mean_error / variance
        else:
            raise ValueError(f"Unsupported global_update_error_type: {error_type}")


    def apply_user_embedding_global_update(self, anchor_item_id, anchor_true_score):
        if self.user_embedding is None or self.item_embd_current is None:
            raise ValueError("user_embd_update requires current user embedding and item embeddings")

        anchor_idx = self.item_ids_to_item_idxs[anchor_item_id]
        alpha = self.score_update_cfg["alpha"]

        update_error = self.get_global_update_error(anchor_idx, anchor_true_score)
        anchor_item_embedding = self.item_embd_current[anchor_idx]

        # One online user-update step
        self.user_embedding = self.user_embedding + alpha * update_error * anchor_item_embedding

        # Recompute means for the current universe
        updated_item_means = self.item_embd_current @ self.user_embedding
        updated_item_means = updated_item_means.detach().cpu().numpy()

        min_score = float(np.min(self.raw_score_values_for_rebuild))
        max_score = float(np.max(self.raw_score_values_for_rebuild))
        self.item_means = np.clip(updated_item_means, min_score, max_score)


    def run_rank_dist(self, scores_type, rd_by_scores=False, rank_dist_results_path=None):
        rank_dist = RankDist(self.item_score_distributions, self.item_score_values, device=device)
        if scores_type == "equal_scores":
            batch_size = math.floor(math.sqrt(len(self.item_score_distributions)))
            self.rank_dist_log_result = rank_dist.rank_dist_with_tiebreaking_batch(k=self.top_k_max_size,
                                                                                   batch_size=batch_size).cpu().detach().numpy()
        else:
            self.rank_dist_log_result = rank_dist.rank_dist(k=self.top_k_max_size).cpu().detach().numpy()

        if rank_dist_results_path is not None:
            np.save(f'{rank_dist_results_path}rank_dist_result_array_user_{self.user_id}.npy', self.rank_dist_log_result)

        if not rd_by_scores:
            self.rank_dist_log_result = np.logaddexp.reduce(self.rank_dist_log_result, axis=2)

    def get_rec(self, method, rec_size, k_semantics=None, exclude_item_ids=None):
        """
        Return top-(rec_size) item_ids and item_idxs according to method,
        after removing items in exclude_item_ids.
        k_semantics is the K used by semantics when it matters (e.g., Global Top-K).
            If None, defaults to rec_size.
        """
        if k_semantics is None:
            k_semantics = rec_size
        if exclude_item_ids is None:
            exclude_item_ids = []

        if method == "expected_score":
            item_scores = self.compute_item_expected_scores().detach().cpu().numpy()
        elif method == "probability_of_relevance_ranking":
            item_scores = self.compute_item_prr_scores().detach().cpu().numpy()
        elif method.startswith("uncertainty_based_filtering"):
            percentile = float(method.split("-")[1])
            ordered_item_ids = self.get_item_ids_after_ubf(percentile, len(self.item_ids))
            possible_rec_item_ids = [item_id for item_id in ordered_item_ids
                                     if item_id not in exclude_item_ids]
            ranked_rec_item_ids = possible_rec_item_ids[:rec_size]
            ranked_rec_item_idxs = [self.item_ids_to_item_idxs[item_id]
                                          for item_id in ranked_rec_item_ids]
            return ranked_rec_item_ids, ranked_rec_item_idxs
        elif method == "global_top_k":
            rank_dist_result_at_k = self.rank_dist_log_result[:, :k_semantics]
            item_scores = np.logaddexp.reduce(rank_dist_result_at_k, axis=1)
        elif method == "U-k-ranks":
            raise ValueError(f'The method {method} for generating recommendation is not implemented yet')
        else:
            raise ValueError(f'The method {method} for generating recommendation does not exist')

        items_indices_ordered = np.argsort(item_scores)[::-1]
        possible_rec_item_ids = [self.item_idxs_to_item_ids[item_idx]
                                 for item_idx in items_indices_ordered
                                 if self.item_idxs_to_item_ids[item_idx] not in exclude_item_ids]
        ranked_rec_item_ids = possible_rec_item_ids[:rec_size]
        ranked_rec_item_idxs = [self.item_ids_to_item_idxs[item_id]
                                for item_id in ranked_rec_item_ids]
        return ranked_rec_item_ids, ranked_rec_item_idxs


    def compute_exact_metrics(self, top_k_result):
        return compute_exact_metrics_for_user(top_k_result, self.user_test_objects)

