from typing import Mapping, Optional, Tuple
import warnings

import torch
import numpy as np
import torch.nn.functional as F
from sklearn import neighbors


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


def min_loss(output, target, alpha=0, mode="per gene"):
    """
    selects one closest cell and computes the loss

    the target is the set of velocity target candidates,
    find the closest in them.

    output: torch.tensor e.g. (128, 2000)
    target: torch.tensor e.g. (128, 30, 2000)
    """
    distance = torch.pow(
        target - torch.unsqueeze(output, 1), exponent=2
    )  # (128, 30, 2000)
    if mode == "per gene":
        distance = torch.min(distance, dim=1)[
            0
        ]  # find the closest target for each gene (128, 2000)
        min_distance = torch.sum(distance, dim=1)  # (128,)
    elif mode == "per cell":
        distance = torch.sum(distance, dim=2)  # (128, 30)
        min_distance = torch.min(distance, dim=1)[0]  # (128,)
    else:
        raise NotImplementedError

    # loss = torch.mean(torch.max(torch.tensor(alpha).float(), min_distance))
    loss = torch.mean(min_distance)

    return loss


def _find_candidates(
    output: torch.Tensor,
    current_state: torch.Tensor,
    idx: torch.LongTensor,
    candidate_states: torch.Tensor,
    n_genes: int,
    inner_batch_size: int,
    cand_tp1: Optional[torch.LongTensor] = None,
    cand_tm1: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the candidate states for each cell in the batch.

    Args:
    output (torch.Tensor):
        The predicted velocity vectors, (batch_size, genes).
    current_state (torch.Tensor):
        The current cell state, (batch_size, genes).
    idx (torch.LongTensor):
        The indices future nearest neighbors, (batch_size, num_neighbors).
    candidate_states (torch.Tensor):
        The states of potential future nearest neighbors, (all_cells, genes).
    n_genes (int):
        The number of spliced genes.
    inner_batch_size (int):
        The batch size for the inner loop.
    cand_tp1, cand_tm1 (torch.LongTensor):
        The indices of the candidate states. If this is not None, then will directly
        use the candidate_ids to compute the continity loss, instead of using the
        computed future neighbor index.

    Returns:
    delta_tp1 (torch.Tensor):
        The predicted average difference to t+1 state, (batch_size, genes).
    delta_tm1 (torch.Tensor):
        The predicted average difference to t-1 state, (batch_size, genes).
    """
    inner_batch_size = min(inner_batch_size, output.shape[0])
    num_neighbors = idx.shape[1]
    delta_tp1 = []
    delta_tm1 = []

    if cand_tp1 is not None and cand_tm1 is not None:
        warnings.warn(
            "Using pre-computed candidate states. Make sure this is expected "
            "according to the config setting of `stop_pearson_after`.",
            RuntimeWarning,
        )
        for i in range(0, output.shape[0], inner_batch_size):
            batch_output = output[i : i + inner_batch_size]  # (inner_batch, genes)
            batch_current_state = current_state[
                i : i + inner_batch_size
            ]  # (inner_batch, genes)
            batch_idx = idx[i : i + inner_batch_size]  # (inner_batch, num_neighbors)
            flatten_batch_idx = batch_idx.view(-1)  # (inner_batch * num_neighbors)
            flatten_candidate_state = candidate_states[
                flatten_batch_idx, :
            ]  # (inner_batch * num_neighbors, genes)
            candidate_state = flatten_candidate_state.view(
                batch_idx.shape[0], num_neighbors, -1
            )
            delta = candidate_state - batch_current_state.unsqueeze(
                1
            )  # (inner_batch, num_neighbors, genes)

            candidates_p1 = cand_tp1[i : i + inner_batch_size]
            num_p1 = torch.sum(candidates_p1, dim=1)  # (inner_batch)
            batch_delta_p1 = torch.sum(
                delta * candidates_p1.unsqueeze(2).float(), dim=1
            )
            batch_delta_p1 = batch_delta_p1 / (
                num_p1.unsqueeze(1).float() + 1e-9
            )  # (inner_batch, genes)
            delta_tp1.append(batch_delta_p1)

            candidates_m1 = cand_tm1[i : i + inner_batch_size]
            num_m1 = torch.sum(candidates_m1, dim=1)  # (inner_batch)
            batch_delta_m1 = torch.sum(
                delta * candidates_m1.unsqueeze(2).float(), dim=1
            )
            batch_delta_m1 = batch_delta_m1 / (
                num_m1.unsqueeze(1).float() + 1e-9
            )  # (inner_batch, genes)
            delta_tm1.append(batch_delta_m1)
        delta_tp1 = torch.cat(delta_tp1, dim=0)  # (batch_size, genes)
        delta_tm1 = torch.cat(delta_tm1, dim=0)  # (batch_size, genes)
        return delta_tp1, delta_tm1, cand_tp1, cand_tm1

    cand_tp1 = []
    cand_tm1 = []
    for i in range(0, output.shape[0], inner_batch_size):
        batch_output = output[i : i + inner_batch_size]  # (inner_batch, genes)
        batch_current_state = current_state[
            i : i + inner_batch_size
        ]  # (inner_batch, genes)
        batch_idx = idx[i : i + inner_batch_size]  # (inner_batch, num_neighbors)
        flatten_batch_idx = batch_idx.view(-1)  # (inner_batch * num_neighbors)
        flatten_candidate_state = candidate_states[
            flatten_batch_idx, :
        ]  # (inner_batch * num_neighbors, genes)
        candidate_state = flatten_candidate_state.view(
            batch_idx.shape[0], num_neighbors, -1
        )
        delta = candidate_state - batch_current_state.unsqueeze(
            1
        )  # (inner_batch, num_neighbors, genes)
        cos_sim = F.cosine_similarity(
            delta[:, :, :n_genes],
            batch_output[:, :n_genes].unsqueeze(1),
            dim=2,
        )  # (inner_batch, num_neighbors)

        candidates_p1 = cos_sim > 0  # (inner_batch, num_neighbors)
        cand_tp1.append(candidates_p1)
        num_p1 = torch.sum(candidates_p1, dim=1)  # (inner_batch)
        batch_delta_p1 = torch.sum(delta * candidates_p1.unsqueeze(2).float(), dim=1)
        batch_delta_p1 = batch_delta_p1 / (
            num_p1.unsqueeze(1).float() + 1e-9
        )  # (inner_batch, genes)
        delta_tp1.append(batch_delta_p1)

        candidates_m1 = cos_sim < 0  # (inner_batch, num_neighbors)
        cand_tm1.append(candidates_m1)
        num_m1 = torch.sum(candidates_m1, dim=1)  # (inner_batch)
        batch_delta_m1 = torch.sum(delta * candidates_m1.unsqueeze(2).float(), dim=1)
        batch_delta_m1 = batch_delta_m1 / (
            num_m1.unsqueeze(1).float() + 1e-9
        )  # (inner_batch, genes)
        delta_tm1.append(batch_delta_m1)

    delta_tp1 = torch.cat(delta_tp1, dim=0)  # (batch_size, genes)
    delta_tm1 = torch.cat(delta_tm1, dim=0)  # (batch_size, genes)
    cand_tp1 = torch.cat(cand_tp1, dim=0)  # (batch_size, num_neighbors)
    cand_tm1 = torch.cat(cand_tm1, dim=0)  # (batch_size, num_neighbors)
    return delta_tp1, delta_tm1, cand_tp1, cand_tm1


def integrate_mle(
    output: torch.Tensor,
    current_state: torch.Tensor,
    idx: torch.LongTensor,
    candidate_states: torch.Tensor,
    n_spliced: int = None,
    inner_batch_size: int = None,
    reduce: bool = True,
    candidate_ids: dict = None,
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    The specialized maximum likelihood estimation loss for predicting possible
    future cell states.

    Args:
    output (torch.Tensor):
        The predicted velocity vectors, (batch_size, genes).
    current_state (torch.Tensor):
        The current cell state, (batch_size, genes).
    idx (torch.LongTensor):
        The indices future nearest neighbors, (batch_size, num_neighbors).
    candidate_states (torch.Tensor):
        The states of potential future nearest neighbors, (all_cells, genes).
    n_spliced (int):
        The number of spliced genes.
    inner_batch_size (int):
        The batch size for the inner loop.
    reduce (bool):
        Whether to reduce the loss to a scalar.
    candidate_ids (dict):
        The indices of the candidate states. If this is not None, then will directly
        use the candidate_ids to compute the continity loss, instead of using the
        computed future neighbor index.

    Returns:
        (torch.Tensor) of shape (1,) if reduce is True, otherwise (batch_size, genes).
    """
    batch_size, genes = current_state.shape
    if n_spliced is not None:
        genes = n_spliced
    if inner_batch_size is None:
        inner_batch_size = int(max(4e8 / idx.shape[1] / current_state.shape[1], 1))
    elif inner_batch_size > 0:
        inner_batch_size = int(inner_batch_size)
    else:
        raise ValueError("inner_batch_size must be a positive integer.")
    try:
        with torch.no_grad():
            delta_tp1, delta_tm1, cand_tp1, cand_tm1 = _find_candidates(
                output,
                current_state,
                idx,
                candidate_states,
                n_genes=genes,
                inner_batch_size=inner_batch_size,
                cand_tp1=candidate_ids["tp1"] if candidate_ids else None,
                cand_tm1=candidate_ids["tm1"] if candidate_ids else None,
            )  # (batch_size, genes)
    except RuntimeError as e:
        raise RuntimeError(
            f"The current inner batch size {inner_batch_size} is too large. "
            "Try reducing the 'inner_batch_size' in congigs."
        ) from e
    if not reduce:
        se_tp1 = torch.pow(output - delta_tp1, 2)
        se_tm1 = torch.pow(output - delta_tm1, 2)
        return se_tp1 + se_tm1, {"tp1": cand_tp1, "tm1": cand_tm1}

    loss_tp1 = torch.mean(torch.pow(output - delta_tp1, 2))
    loss_tm1 = torch.mean(torch.pow(output + delta_tm1, 2))
    loss = (loss_tp1 + loss_tm1) / 2 * np.sqrt(genes)
    return loss, {"tp1": cand_tp1, "tm1": cand_tm1}


def mle(
    output: torch.Tensor,
    current_state: torch.Tensor,
    idx: torch.LongTensor,
    candidate_states: torch.Tensor,
    n_spliced: int = None,
    reduce: bool = True,
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    The specialized maximum likelihood estimation loss for predicting possible
    future cell states. The implementation loops over the neighbor dimension,
    which is different from the equation in paper but works the same way. This
    has higher computational efficiency.

    Args:
    output (torch.Tensor):
        The predicted velocity vectors, (batch_size, genes).
    current_state (torch.Tensor):
        The current cell state, (batch_size, genes).
    idx (torch.LongTensor):
        The indices future nearest neighbors, (batch_size, num_neighbors).
    candidate_states (torch.Tensor):
        The states of potential future nearest neighbors, (all_cells, genes).
    n_spliced (int):
        The number of spliced genes.
    reduce (bool):
        Whether to reduce the loss to a scalar or return the raw multi-dim tensor.

    Returns: (torch.Tensor)
    """
    batch_size, genes = current_state.shape
    if n_spliced is not None:
        genes = n_spliced
    num_neighbors = idx.shape[1]
    loss = []
    for i in range(num_neighbors):
        ith_neighbors = idx[:, i]
        candidate_state = candidate_states[ith_neighbors, :]  # (batch_size, genes)
        delta = (candidate_state - current_state).detach()  # (batch_size, genes)
        cos_sim = F.cosine_similarity(
            delta[:, :genes], output[:, :genes], dim=1
        )  # (batch_size,)

        # t+1 direction
        candidates = cos_sim.detach() > 0
        squared_difference = torch.mean(torch.pow(output - delta, 2), dim=1)
        squared_difference = squared_difference[candidates]
        squared_difference = (
            squared_difference / torch.mean(torch.pow(delta, 2)) * np.sqrt(genes)
        )  # scaling
        loss.append(torch.sum(squared_difference) / len(candidates))

        # t-1 direction, (-output) - delta
        candidates = cos_sim.detach() < 0
        squared_difference = torch.mean(torch.pow(output + delta, 2), dim=1)
        squared_difference = squared_difference[candidates]
        squared_difference = (
            squared_difference / torch.mean(torch.pow(delta, 2)) * np.sqrt(genes)
        )  # scaling
        loss.append(torch.sum(squared_difference) / len(candidates))
        # TODO: check why the memory usage is high for this one
    loss = torch.stack(loss).mean()
    return loss


def pearson(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    """
    The pearson correlation coefficient between two vectors. by default, the
    first dimension is the sample dimension.

    Args:
    x (torch.Tensor):
        The first vector, (N, D).
    y (torch.Tensor):
        The second vector, (N, D).
    mask (torch.BoolTensor):
        The mask showing valid data positions, (N, D).

    Returns: (torch.Tensor), (D,)
    """
    if mask is None:
        x = x - torch.mean(x, dim=0)
        y = y - torch.mean(y, dim=0)
        x = x / (torch.std(x, dim=0) + 1e-9)
        y = y / (torch.std(y, dim=0) + 1e-9)
        return torch.mean(x * y, dim=0)  # (D,)
    else:
        assert mask.dtype == torch.bool
        mask = mask.detach().float()
        num_valid_data = torch.sum(mask, dim=0)  # (D,)

        y = y * mask
        x = x * mask
        x = x - torch.sum(x, dim=0) / (num_valid_data + 1e-9)
        y = y - torch.sum(y, dim=0) / (num_valid_data + 1e-9)
        y = y * mask  # make the invalid data to zero again to ignore
        x = x * mask
        x = x / torch.sqrt(torch.sum(torch.pow(x, 2), dim=0) + 1e-9)
        y = y / torch.sqrt(torch.sum(torch.pow(y, 2), dim=0) + 1e-9)
        return torch.sum(x * y, dim=0)  # (D,)


def direction_loss(
    velocity: torch.Tensor,
    spliced_counts: torch.Tensor,
    unspliced_counts: torch.Tensor,
    velocity_u: Optional[torch.Tensor] = None,
    coeff_u: float = 1.0,
    coeff_s: float = 1.0,
    reduce: bool = True,
) -> torch.Tensor:
    """
    The constraint for the direction of the velocity vectors. Large ratio of u/s
    should have positive direction, for each gene.

    Args:
    velocity (torch.Tensor):
        The predicted velocity vectors, (batch_size, genes).
    spliced_counts (torch.Tensor):
        The number of spliced reads, (batch_size, genes).
    unspliced_counts (torch.Tensor):
        The number of unspliced reads, (batch_size, genes).
    velocity_u (torch.Tensor):
        The predicted velocity vectors for unspliced reads, (batch_size, genes).
    coeff_u (float):
        The coefficient for the unspliced reads.
    coeff_s (float):
        The coefficient for the spliced reads.
    reduce (bool):
        Whether to reduce the loss to a scalar.

    Returns: (torch.Tensor), (1,)
    """

    # 3. Intereting for the gliogenic cells of the GABAInterneuraons ====
    mask1 = unspliced_counts > 0
    mask2 = spliced_counts > 0
    corr = coeff_u * pearson(velocity, unspliced_counts, mask1) - coeff_s * pearson(
        velocity, spliced_counts, mask2
    )

    # 4.===============================================================
    # mask = (spliced_counts > 0) & (unspliced_counts > 0)
    # corr = coeff_u * pearson(velocity, unspliced_counts, mask) - coeff_s * pearson(
    #     velocity, spliced_counts, mask
    # )

    if not reduce:
        return corr / (coeff_u + coeff_s)  # range [-1, 1], shape (genes,)

    loss = coeff_u + coeff_s - torch.mean(corr)  # to maximize the correlation
    loss = loss / (coeff_u + coeff_s)
    # should not use correlation to u, since the alpha coefficient is unknown and
    # it definitely varies along the phase portrait.
    # if velocity_u is not None:
    #     corr_unpliced = pearson(velocity_u, unspliced_counts, mask)
    #     loss_u = 1 + torch.mean(corr_unpliced)  # mininize the corr_unpliced
    #     loss = 0.5 * (loss + loss_u)
    return loss  # [0, 2.0]


def mle_plus_direction(
    output: torch.Tensor,
    current_state: torch.Tensor,
    idx: torch.LongTensor,
    candidate_states: torch.Tensor,
    spliced_counts: torch.Tensor,
    unspliced_counts: torch.Tensor,
    pearson_scale: float = 10.0,
    coeff_u: float = 1.0,
    coeff_s: float = 1.0,
    inner_batch_size: Optional[int] = None,
    reduce: bool = True,
    do_pearson: bool = True,
    candidate_ids: Optional[dict] = None,
    return_candidates: bool = False,
) -> torch.Tensor:
    """
    The combination of maximum likelihood estimation loss and direction loss.

    Args:
    output (torch.Tensor): The model output.
    current_state (torch.Tensor):
        The current cell state, (batch_size, genes).
    idx (torch.LongTensor):
        The indices future nearest neighbors, (batch_size, num_neighbors).
    candidate_states (torch.Tensor):
        The states of potential future nearest neighbors, (all_cells, genes).
    spliced_counts (torch.Tensor): The tensor of spliced counts.
    unspliced_counts (torch.Tensor): The tensor of unspliced counts.
    pearson_scale (float): The scale for the pearson correlation.
    coeff_u (float): The coefficient for the unspliced reads.
    coeff_s (float): The coefficient for the spliced reads.
    inner_batch_size (int):
        The batch size for the inner loop.
    reduce (bool):
        Whether to reduce the loss to a scalar.
    do_pearson (bool):
        Whether to do pearson correlation.
    candidate_ids (dict):
        The indices of the candidate states. If this is not None, then will directly
        use the candidate_ids to compute the continity loss, instead of using the
        computed future neighbor index.
    """
    batch_size, genes = current_state.shape
    if output.shape[1] == spliced_counts.shape[1]:
        velocity = output
    elif output.shape[1] == 2 * spliced_counts.shape[1]:
        # when predicting unspliced velocity, the output demension equals 2*genes
        velocity = output[:, : spliced_counts.shape[1]]
    else:
        raise ValueError(
            "output dimension is not correct, "
            "it should either be num of genes when `pred_unspliced` is False, "
            "or 2 * num of genes when `pred_unspliced` is True."
        )

    loss_mle, cand_ids = integrate_mle(
        output,
        current_state,
        idx,
        candidate_states,
        n_spliced=velocity.shape[1],
        inner_batch_size=inner_batch_size,
        reduce=reduce,
        candidate_ids=candidate_ids,
    )
    if do_pearson:
        loss_pearson = direction_loss(
            velocity,
            spliced_counts,
            unspliced_counts,
            coeff_u=coeff_u,
            coeff_s=coeff_s,
            reduce=reduce,
        )
    else:
        loss_pearson = torch.zeros(1, device=output.device)

    if not reduce:
        # raw mse (batch_size, genes) and raw pearson corr (genes,)
        return loss_mle, loss_pearson

    # scale the pearson to make it comparable with mle
    # TODO: should scale by * np.sqrt(genes)? and negative scale to batch size?
    _lambda = (pearson_scale * np.sqrt(genes)) / min(
        0.5, max(0.01, 1 - loss_pearson.item())
    )
    # print(f"mle: {loss_mle.item()}, direction_loss: {loss_pearson.item()}")

    if return_candidates:
        return loss_mle + _lambda * loss_pearson, cand_ids

    return loss_mle + _lambda * loss_pearson


def sim_loss():
    """the loss measuring similarities among input vectors"""
    return
