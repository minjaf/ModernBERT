# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

"""Online "hardest-token" masking for MLM training.

The :class:`AdaptiveMasker` consumes per-token hardness scores (typically
``1 - p_correct`` from a no-grad forward of the current model on the
unmasked inputs) and produces a masked ``input_ids`` tensor + ``labels``
tensor following the standard BERT 80% [MASK] / 10% random / 10% kept
recipe, while restricting masking to positions that are flagged as
"maskable" (non-pad, non-special).

Selection strategy is top-K:
  * For each row, the per-row budget is
    ``K_i = round(mlm_probability * N_maskable_i)``.
  * ``K_top = round((1 - random_fraction) * K_i)`` highest-scoring
    maskable positions are picked.
  * ``K_rand = K_i - K_top`` extra positions are picked uniformly at
    random from the remaining maskable positions (exploration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


_IGNORE_INDEX = -100
_NEG_INF = float("-inf")


@dataclass
class AdaptiveMaskingConfig:
    """Knobs for the online adaptive (hardest-token) MLM masking strategy.

    Attributes:
        enabled: Master switch. When ``False`` the module is a no-op and
            the standard random masking in the data pipeline is used
            instead.
        selection_mode: Only ``"topk"`` is implemented for the first cut.
        random_fraction: Fraction of the per-row mask budget that is
            chosen uniformly at random from the remaining maskable
            positions (the rest is taken from the top-scoring ones).
            Acts as an exploration term so the model does not stay
            stuck on the same handful of always-hard tokens.
        warmup_batches: Use random masking for the first N microbatches
            seen by the wrapper. After that, switch to adaptive.
        recompute_every_n_steps: If > 1, do the no-grad scoring forward
            once every N microbatches; on the other steps fall back to
            random masking (cheap).
        use_eval_mode_for_scoring: Toggle ``model.eval()`` around the
            no-grad scoring pass to disable dropout during scoring.
        bert_replacement: If ``True``, apply the standard 80/10/10
            corruption recipe at selected positions; if ``False``, all
            selected positions get the [MASK] token.
        apply_at_eval: If ``True``, also use adaptive masking during
            eval. By default eval uses random masking (matches existing
            ``eval_loader.mlm_probability`` semantics).
        score: Which score function to use over the no-grad logits.
            ``"one_minus_p_correct"`` (default) or
            ``"neg_log_p_correct"`` (cross-entropy per token).
    """

    enabled: bool = False
    selection_mode: str = "topk"
    random_fraction: float = 0.1
    warmup_batches: int = 0
    recompute_every_n_steps: int = 1
    use_eval_mode_for_scoring: bool = True
    bert_replacement: bool = True
    apply_at_eval: bool = False
    score: str = "one_minus_p_correct"

    def validate(self) -> "AdaptiveMaskingConfig":
        if self.selection_mode != "topk":
            raise NotImplementedError(
                f"adaptive_masking.selection_mode={self.selection_mode!r} is not implemented (only 'topk')."
            )
        if not (0.0 <= self.random_fraction <= 1.0):
            raise ValueError(f"random_fraction must be in [0, 1], got {self.random_fraction}.")
        if self.recompute_every_n_steps < 1:
            raise ValueError(f"recompute_every_n_steps must be >= 1, got {self.recompute_every_n_steps}.")
        if self.warmup_batches < 0:
            raise ValueError(f"warmup_batches must be >= 0, got {self.warmup_batches}.")
        if self.score not in ("one_minus_p_correct", "neg_log_p_correct"):
            raise NotImplementedError(f"adaptive_masking.score={self.score!r} is not implemented.")
        return self

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "AdaptiveMaskingConfig":
        if cfg is None:
            return cls(enabled=False).validate()
        known = {f.name for f in fields_of(cls)}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(
                f"Unknown adaptive_masking keys: {sorted(unknown)}. Allowed: {sorted(known)}."
            )
        return cls(**cfg).validate()


def fields_of(cls):
    # Lightweight stand-in for ``dataclasses.fields`` that avoids a top-level import
    # of ``dataclasses.fields`` (kept consistent with rest of module style).
    from dataclasses import fields as _fields

    return _fields(cls)


def compute_hardness_scores(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    maskable_mask: torch.Tensor,
    score: str = "one_minus_p_correct",
) -> torch.Tensor:
    """Per-token "hardness" of the current model w.r.t. the true tokens.

    Args:
        logits: ``(B, S, V)`` float tensor of MLM logits (unmasked forward).
        target_ids: ``(B, S)`` long tensor of the true token IDs.
        maskable_mask: ``(B, S)`` bool tensor; non-maskable positions get
            score = ``-inf`` so they are never selected by top-K.
        score: ``"one_minus_p_correct"`` or ``"neg_log_p_correct"``.

    Returns:
        ``(B, S)`` float32 tensor of scores; higher = harder.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be (B, S, V), got shape {tuple(logits.shape)}.")
    if target_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"target_ids shape {tuple(target_ids.shape)} must match logits[:2] {tuple(logits.shape[:2])}."
        )
    if maskable_mask.shape != target_ids.shape:
        raise ValueError(
            f"maskable_mask shape {tuple(maskable_mask.shape)} must match target_ids {tuple(target_ids.shape)}."
        )

    # Cast to float32 for stable softmax / log-softmax (logits are typically bf16 in train).
    logits_f = logits.float()
    if score == "one_minus_p_correct":
        probs = torch.softmax(logits_f, dim=-1)
        p_true = probs.gather(dim=-1, index=target_ids.unsqueeze(-1).long()).squeeze(-1)
        scores = 1.0 - p_true
    elif score == "neg_log_p_correct":
        log_probs = torch.log_softmax(logits_f, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1).long()).squeeze(-1)
        scores = nll
    else:
        raise NotImplementedError(f"Unknown score function {score!r}.")

    scores = scores.masked_fill(~maskable_mask, _NEG_INF)
    return scores


class AdaptiveMasker:
    """Stateful masker. Holds a CPU/GPU RNG so selection is reproducible."""

    def __init__(
        self,
        cfg: AdaptiveMaskingConfig,
        mask_token_id: int,
        vocab_size: int,
        ignore_index: int = _IGNORE_INDEX,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg.validate()
        if mask_token_id is None:
            raise ValueError("mask_token_id is required.")
        if vocab_size is None or vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive int, got {vocab_size}.")
        self.mask_token_id = int(mask_token_id)
        self.vocab_size = int(vocab_size)
        self.ignore_index = int(ignore_index)
        self._seed = int(seed)
        self._step = 0  # incremented per call to select_and_mask

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------
    def reset_step(self, step: int = 0) -> None:
        self._step = int(step)

    @property
    def step(self) -> int:
        return self._step

    def _make_generator(self, device: torch.device) -> torch.Generator:
        # Per-step, per-device generator so distributed ranks pick differently
        # only via the data they receive (we keep the seed identical across
        # ranks; data parallelism already shards the batch).
        g = torch.Generator(device=device)
        g.manual_seed(self._seed + self._step)
        return g

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_and_mask(
        self,
        input_ids: torch.Tensor,
        maskable_mask: torch.Tensor,
        scores: torch.Tensor,
        mlm_probability: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pick positions to mask and apply the 80/10/10 BERT recipe.

        Args:
            input_ids: ``(B, S)`` long; original (unmasked) token IDs.
            maskable_mask: ``(B, S)`` bool; True where masking is allowed.
            scores: ``(B, S)`` float; higher = harder. Non-maskable
                positions must be set to ``-inf`` by the caller (this is
                enforced again here defensively).
            mlm_probability: target *expected* fraction of maskable
                tokens to mask per row.
            generator: optional ``torch.Generator``; if ``None`` a
                deterministic per-step generator is built from
                ``self._seed`` + ``self._step``.

        Returns:
            ``(masked_input_ids, labels)`` both shape ``(B, S)`` with the
            same dtype as ``input_ids`` / a long ``labels`` tensor. The
            ``labels`` tensor has ``ignore_index`` at unselected
            positions and the original token at selected positions.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (B, S), got {tuple(input_ids.shape)}.")
        if maskable_mask.shape != input_ids.shape:
            raise ValueError(
                f"maskable_mask shape {tuple(maskable_mask.shape)} must match input_ids {tuple(input_ids.shape)}."
            )
        if scores.shape != input_ids.shape:
            raise ValueError(
                f"scores shape {tuple(scores.shape)} must match input_ids {tuple(input_ids.shape)}."
            )
        if not (0.0 < mlm_probability < 1.0):
            raise ValueError(f"mlm_probability must be in (0, 1), got {mlm_probability}.")

        device = input_ids.device
        gen = generator if generator is not None else self._make_generator(device)

        scores_eff = scores.float().masked_fill(~maskable_mask, _NEG_INF)
        selected = self._select_topk_with_random(
            maskable_mask=maskable_mask,
            scores=scores_eff,
            mlm_probability=mlm_probability,
            generator=gen,
        )
        masked_input_ids, labels = self._apply_corruption(
            input_ids=input_ids, selected=selected, generator=gen
        )
        self._step += 1
        return masked_input_ids, labels

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _select_topk_with_random(
        self,
        maskable_mask: torch.Tensor,
        scores: torch.Tensor,
        mlm_probability: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Per-row top-K mixed with a random tail.

        Returns a ``(B, S)`` bool tensor of selected positions.
        """
        device = scores.device
        B, S = scores.shape

        # Per-row budgets (clipped to the number of maskable positions).
        n_maskable = maskable_mask.sum(dim=1)  # (B,) long
        K_total = torch.round(mlm_probability * n_maskable.float()).long()
        K_total = torch.minimum(K_total, n_maskable)

        K_random = torch.round(self.cfg.random_fraction * K_total.float()).long()
        K_top = K_total - K_random

        # ---- top-K by score ----
        # rank[i, j] = 0 if scores[i, j] is the largest in row i.
        # argsort(descending=True) gives sorted indices; argsort of that gives ranks.
        sort_idx = scores.argsort(dim=1, descending=True)
        ranks = torch.empty_like(sort_idx)
        arange = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        ranks.scatter_(dim=1, index=sort_idx, src=arange)
        topk_selected = ranks < K_top.unsqueeze(1)
        # Defensive: also require maskable (non-maskable have -inf so should
        # only be picked when budget exceeds n_maskable, which we've already
        # clipped).
        topk_selected = topk_selected & maskable_mask

        # ---- random tail from remaining maskable positions ----
        if (K_random > 0).any():
            remaining = maskable_mask & ~topk_selected
            # Build a uniform-random ranking over remaining positions.
            rand_scores = torch.empty(B, S, dtype=torch.float32, device=device)
            rand_scores.uniform_(0.0, 1.0, generator=generator)
            rand_scores = rand_scores.masked_fill(~remaining, _NEG_INF)
            rand_sort_idx = rand_scores.argsort(dim=1, descending=True)
            rand_ranks = torch.empty_like(rand_sort_idx)
            rand_ranks.scatter_(dim=1, index=rand_sort_idx, src=arange)
            random_selected = (rand_ranks < K_random.unsqueeze(1)) & remaining
            selected = topk_selected | random_selected
        else:
            selected = topk_selected

        return selected

    def _apply_corruption(
        self,
        input_ids: torch.Tensor,
        selected: torch.Tensor,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard BERT 80/10/10 corruption at ``selected`` positions."""
        device = input_ids.device

        labels = torch.full_like(input_ids, self.ignore_index)
        labels[selected] = input_ids[selected]

        masked_input_ids = input_ids.clone()
        if not self.cfg.bert_replacement:
            masked_input_ids[selected] = self.mask_token_id
            return masked_input_ids, labels

        # 80% [MASK], 10% random, 10% keep -- among the selected positions.
        rand = torch.empty(input_ids.shape, dtype=torch.float32, device=device)
        rand.uniform_(0.0, 1.0, generator=generator)

        do_mask = selected & (rand < 0.8)
        do_random = selected & (rand >= 0.8) & (rand < 0.9)
        # The remaining (rand >= 0.9) selected positions are kept unchanged.

        masked_input_ids[do_mask] = self.mask_token_id

        if do_random.any():
            random_tokens = torch.randint(
                low=0,
                high=self.vocab_size,
                size=input_ids.shape,
                dtype=input_ids.dtype,
                device=device,
                generator=generator,
            )
            masked_input_ids[do_random] = random_tokens[do_random]

        return masked_input_ids, labels
