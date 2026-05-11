# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

"""Unit tests for ``src.adaptive_masking.AdaptiveMasker``.

These tests run entirely on CPU and do not touch the model, so they
cover the masker contract in isolation:

* per-row budget is correct,
* non-maskable positions are never selected,
* highest-scoring maskable positions are picked first (when
  ``random_fraction == 0.0``),
* ``random_fraction`` mixes in uniform-random picks,
* labels are ``-100`` at unselected positions and the original token at
  selected positions,
* 80/10/10 BERT corruption proportions match (statistical),
* selection + corruption is reproducible given the same seed/step.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Add tests folder root to path so we can import ``src.*`` regardless of cwd.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive_masking import (  # noqa: E402  (path setup above)
    AdaptiveMasker,
    AdaptiveMaskingConfig,
    compute_hardness_scores,
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _make_masker(
    *,
    random_fraction: float = 0.0,
    bert_replacement: bool = True,
    mask_token_id: int = 99,
    vocab_size: int = 100,
    seed: int = 0,
) -> AdaptiveMasker:
    cfg = AdaptiveMaskingConfig(
        enabled=True,
        selection_mode="topk",
        random_fraction=random_fraction,
        bert_replacement=bert_replacement,
    )
    return AdaptiveMasker(cfg, mask_token_id=mask_token_id, vocab_size=vocab_size, seed=seed)


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_config_rejects_unknown_selection_mode():
    cfg = AdaptiveMaskingConfig(enabled=True, selection_mode="bernoulli")
    with pytest.raises(NotImplementedError):
        cfg.validate()


def test_config_rejects_bad_random_fraction():
    with pytest.raises(ValueError):
        AdaptiveMaskingConfig(enabled=True, random_fraction=1.5).validate()


def test_config_from_dict_unknown_key():
    with pytest.raises(ValueError):
        AdaptiveMaskingConfig.from_dict({"enabled": True, "not_a_real_key": 42})


# ----------------------------------------------------------------------
# selection: top-K behavior
# ----------------------------------------------------------------------
def test_select_topk_picks_highest_scores_and_respects_maskable():
    """With random_fraction=0, exactly K_i = round(p * n_maskable_i) highest
    scoring maskable positions per row are selected; non-maskable
    positions are never selected."""
    B, S = 3, 12
    masker = _make_masker(random_fraction=0.0)

    input_ids = torch.arange(B * S).reshape(B, S).long()
    maskable = torch.ones(B, S, dtype=torch.bool)
    # Mark a chunk in each row as non-maskable.
    maskable[0, :2] = False  # row 0: 10 maskable
    maskable[1, -3:] = False  # row 1: 9 maskable
    maskable[2, 4:8] = False  # row 2: 8 maskable

    # Scores: monotonically increasing along S so the largest indices are
    # the most "hard". This makes the expected selection trivial to
    # reason about.
    scores = torch.arange(B * S, dtype=torch.float32).reshape(B, S)

    mlm_prob = 0.5
    masked_input_ids, labels = masker.select_and_mask(
        input_ids=input_ids,
        maskable_mask=maskable,
        scores=scores,
        mlm_probability=mlm_prob,
    )

    selected = labels != masker.ignore_index
    # 1. Non-maskable positions are never selected.
    assert torch.all(~selected[~maskable])

    # 2. Per-row budget = round(p * n_maskable).
    n_maskable = maskable.sum(dim=1)
    expected_k = torch.round(mlm_prob * n_maskable.float()).long()
    assert torch.equal(selected.sum(dim=1), expected_k)

    # 3. Picked positions are exactly the top-K *maskable* positions by score.
    for i in range(B):
        # rank maskable positions in row i by score, descending
        row_scores = scores[i].masked_fill(~maskable[i], float("-inf"))
        top_idx = torch.topk(row_scores, k=int(expected_k[i])).indices
        expected_mask = torch.zeros(S, dtype=torch.bool)
        expected_mask[top_idx] = True
        assert torch.equal(selected[i], expected_mask)


def test_labels_match_original_tokens_at_selected():
    masker = _make_masker(random_fraction=0.0, bert_replacement=False)
    B, S, V = 2, 8, 50
    input_ids = torch.randint(0, V, (B, S), dtype=torch.long)
    maskable = torch.ones(B, S, dtype=torch.bool)
    scores = torch.rand(B, S)

    _, labels = masker.select_and_mask(
        input_ids=input_ids,
        maskable_mask=maskable,
        scores=scores,
        mlm_probability=0.5,
    )
    selected = labels != masker.ignore_index
    assert torch.equal(labels[selected], input_ids[selected])
    assert torch.all(labels[~selected] == masker.ignore_index)


def test_no_replacement_path_uses_only_mask_token():
    """When bert_replacement=False, every selected position is the [MASK] token."""
    mask_token_id = 7
    masker = _make_masker(random_fraction=0.0, bert_replacement=False, mask_token_id=mask_token_id)
    B, S, V = 4, 16, 50
    input_ids = torch.randint(0, V, (B, S), dtype=torch.long)
    maskable = torch.ones(B, S, dtype=torch.bool)
    scores = torch.rand(B, S)

    masked, labels = masker.select_and_mask(
        input_ids=input_ids,
        maskable_mask=maskable,
        scores=scores,
        mlm_probability=0.5,
    )
    selected = labels != masker.ignore_index
    assert torch.all(masked[selected] == mask_token_id)
    # Non-selected positions are untouched.
    assert torch.equal(masked[~selected], input_ids[~selected])


# ----------------------------------------------------------------------
# selection: random_fraction
# ----------------------------------------------------------------------
def test_random_fraction_mixes_in_random_picks():
    """With random_fraction=1.0 and uniform scores, selection is purely
    random and should not match the top-K-by-deterministic-score result."""
    B, S = 1, 200
    input_ids = torch.zeros(B, S, dtype=torch.long)
    maskable = torch.ones(B, S, dtype=torch.bool)
    # Deterministic monotonic scores
    scores = torch.arange(B * S, dtype=torch.float32).reshape(B, S)

    masker_topk = _make_masker(random_fraction=0.0, seed=0)
    masker_rand = _make_masker(random_fraction=1.0, seed=0)

    _, labels_top = masker_topk.select_and_mask(input_ids, maskable, scores, mlm_probability=0.3)
    _, labels_rnd = masker_rand.select_and_mask(input_ids, maskable, scores, mlm_probability=0.3)

    sel_top = labels_top != masker_topk.ignore_index
    sel_rnd = labels_rnd != masker_rand.ignore_index

    # Both should mask roughly 30% of tokens.
    assert sel_top.sum().item() == int(round(0.3 * S))
    assert sel_rnd.sum().item() == int(round(0.3 * S))

    # The selections should differ (one is deterministic top-K, the other random).
    assert not torch.equal(sel_top, sel_rnd)

    # In top-K mode, the highest 30% indices are selected (this is the contrapositive
    # check that the random_fraction=1.0 case really is doing something different).
    expected_top_indices = torch.arange(S - int(round(0.3 * S)), S)
    expected_top_mask = torch.zeros(B, S, dtype=torch.bool)
    expected_top_mask[0, expected_top_indices] = True
    assert torch.equal(sel_top, expected_top_mask)


def test_random_fraction_partial_mix_keeps_total_budget():
    """random_fraction=0.5 should still mask the same total number of tokens per row."""
    B, S = 2, 64
    masker = _make_masker(random_fraction=0.5)
    input_ids = torch.zeros(B, S, dtype=torch.long)
    maskable = torch.ones(B, S, dtype=torch.bool)
    scores = torch.rand(B, S)
    mlm_prob = 0.25

    _, labels = masker.select_and_mask(input_ids, maskable, scores, mlm_probability=mlm_prob)
    selected = labels != masker.ignore_index

    expected_k = int(round(mlm_prob * S))
    assert torch.all(selected.sum(dim=1) == expected_k)


# ----------------------------------------------------------------------
# 80/10/10 corruption
# ----------------------------------------------------------------------
def test_80_10_10_corruption_proportions():
    """Statistical: out of the selected positions, ~80% become MASK,
    ~10% become random, ~10% are kept unchanged."""
    B, S = 32, 256
    mask_token_id = 999
    vocab_size = 1000
    masker = AdaptiveMasker(
        AdaptiveMaskingConfig(enabled=True, random_fraction=0.0, bert_replacement=True),
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        seed=123,
    )
    # Use tokens that are NOT the mask token and NOT 0 so the "kept"
    # bucket is easy to detect against the original ids.
    torch.manual_seed(0)
    input_ids = torch.randint(1, vocab_size - 1, (B, S), dtype=torch.long)
    # Ensure none equal mask_token_id (so "becomes mask" is unambiguous).
    assert (input_ids != mask_token_id).all()

    maskable = torch.ones(B, S, dtype=torch.bool)
    scores = torch.rand(B, S)

    masked, labels = masker.select_and_mask(input_ids, maskable, scores, mlm_probability=0.5)
    selected = labels != masker.ignore_index
    n_sel = int(selected.sum().item())
    assert n_sel > 0

    became_mask = ((masked == mask_token_id) & selected).sum().item()
    kept = ((masked == input_ids) & selected).sum().item()
    became_random = n_sel - became_mask - kept

    p_mask = became_mask / n_sel
    p_random = became_random / n_sel
    p_kept = kept / n_sel

    # Generous bands — N is big enough but not gigantic.
    assert 0.75 < p_mask < 0.85, f"p_mask={p_mask}"
    assert 0.05 < p_random < 0.15, f"p_random={p_random}"
    assert 0.05 < p_kept < 0.15, f"p_kept={p_kept}"


# ----------------------------------------------------------------------
# reproducibility
# ----------------------------------------------------------------------
def test_selection_is_reproducible_at_same_step():
    """Two maskers with the same seed and step produce the same result."""
    B, S = 4, 32
    input_ids = torch.randint(0, 100, (B, S), dtype=torch.long)
    maskable = torch.ones(B, S, dtype=torch.bool)
    scores = torch.rand(B, S)

    m1 = _make_masker(random_fraction=0.3, seed=42)
    m2 = _make_masker(random_fraction=0.3, seed=42)

    out1 = m1.select_and_mask(input_ids, maskable, scores, mlm_probability=0.3)
    out2 = m2.select_and_mask(input_ids, maskable, scores, mlm_probability=0.3)

    assert torch.equal(out1[0], out2[0])
    assert torch.equal(out1[1], out2[1])


def test_selection_changes_across_steps():
    """The internal step counter should advance the RNG so that
    repeated calls on the same data produce *different* random tails."""
    B, S = 1, 200
    input_ids = torch.zeros(B, S, dtype=torch.long)
    maskable = torch.ones(B, S, dtype=torch.bool)
    # Uniform scores: selection is dominated by the random RNG.
    scores = torch.zeros(B, S)

    m = _make_masker(random_fraction=1.0, seed=7)
    out_a = m.select_and_mask(input_ids, maskable, scores, mlm_probability=0.3)
    out_b = m.select_and_mask(input_ids, maskable, scores, mlm_probability=0.3)

    sel_a = out_a[1] != m.ignore_index
    sel_b = out_b[1] != m.ignore_index
    assert not torch.equal(sel_a, sel_b)


# ----------------------------------------------------------------------
# hardness score helper
# ----------------------------------------------------------------------
def test_compute_hardness_scores_one_minus_p_correct():
    B, S, V = 2, 4, 5
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0, 0.0, 0.0],  # almost surely class 0
                [0.0, 0.0, 10.0, 0.0, 0.0],  # almost surely class 2
                [0.0, 10.0, 0.0, 0.0, 0.0],  # almost surely class 1
                [0.0, 0.0, 0.0, 10.0, 0.0],  # almost surely class 3
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 10.0],
                [10.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    # Targets match the argmax everywhere => score ~= 0 at all positions.
    target_ids = torch.tensor([[0, 2, 1, 3], [4, 0, 2, 1]], dtype=torch.long)
    maskable = torch.ones(B, S, dtype=torch.bool)
    scores = compute_hardness_scores(logits, target_ids, maskable, score="one_minus_p_correct")
    assert scores.shape == (B, S)
    assert torch.all(scores < 1e-3), f"easy-token scores should be near 0, got {scores}"

    # Now use targets that *don't* match argmax => score ~= 1 everywhere.
    bad_targets = torch.zeros(B, S, dtype=torch.long)
    bad_targets[0, 0] = 1  # bad target for the "class 0" position
    scores_bad = compute_hardness_scores(logits, bad_targets, maskable, score="one_minus_p_correct")
    assert scores_bad[0, 0] > 0.99


def test_compute_hardness_scores_masks_out_non_maskable():
    B, S, V = 1, 3, 4
    logits = torch.zeros(B, S, V)
    target_ids = torch.zeros(B, S, dtype=torch.long)
    maskable = torch.tensor([[True, False, True]])
    scores = compute_hardness_scores(logits, target_ids, maskable)
    assert scores[0, 1].item() == float("-inf")
    assert torch.isfinite(scores[0, 0])
    assert torch.isfinite(scores[0, 2])


# ----------------------------------------------------------------------
# edge cases
# ----------------------------------------------------------------------
def test_zero_maskable_in_a_row_produces_no_selection():
    """A row whose maskable_mask is entirely False is left untouched."""
    B, S = 2, 8
    masker = _make_masker(random_fraction=0.0)
    input_ids = torch.randint(1, 100, (B, S), dtype=torch.long)
    maskable = torch.ones(B, S, dtype=torch.bool)
    maskable[0] = False  # row 0 is fully non-maskable
    scores = torch.rand(B, S)

    masked, labels = masker.select_and_mask(input_ids, maskable, scores, mlm_probability=0.5)
    # Row 0: nothing selected; row 1: some positions selected.
    assert (labels[0] == masker.ignore_index).all()
    assert torch.equal(masked[0], input_ids[0])
    assert (labels[1] != masker.ignore_index).any()


def test_mlm_probability_out_of_range_raises():
    masker = _make_masker()
    input_ids = torch.zeros(2, 8, dtype=torch.long)
    maskable = torch.ones(2, 8, dtype=torch.bool)
    scores = torch.zeros(2, 8)
    with pytest.raises(ValueError):
        masker.select_and_mask(input_ids, maskable, scores, mlm_probability=0.0)
    with pytest.raises(ValueError):
        masker.select_and_mask(input_ids, maskable, scores, mlm_probability=1.0)


# ----------------------------------------------------------------------
# wrapper integration smoke test
# ----------------------------------------------------------------------
def test_efficient_hf_model_falls_back_when_maskable_mask_absent(monkeypatch):
    """``EfficientHuggingFaceModel.forward`` must pass batches without
    ``maskable_mask`` straight through to ``HuggingFaceModel.forward``.

    Without this guard, eval (which uses the non-adaptive loader) or
    legacy code paths would break the moment a user flipped the
    ``adaptive_masking`` switch on.

    We assemble the wrapper manually with a stub model + stub tokenizer
    so the test does not depend on actually loading a FlexBERT.
    """
    from types import SimpleNamespace

    from composer.models.huggingface import HuggingFaceModel
    from src.flex_bert import EfficientHuggingFaceModel

    parent_calls = {"count": 0, "last_batch": None}

    def stub_parent_forward(self, batch):
        parent_calls["count"] += 1
        parent_calls["last_batch"] = batch
        return SimpleNamespace(logits=torch.zeros(1, 1))

    monkeypatch.setattr(HuggingFaceModel, "forward", stub_parent_forward, raising=True)

    # Build the wrapper bypassing __init__ (we only exercise forward()).
    wrapper = EfficientHuggingFaceModel.__new__(EfficientHuggingFaceModel)
    wrapper.adaptive_masker = AdaptiveMasker(
        AdaptiveMaskingConfig(enabled=True),
        mask_token_id=4,
        vocab_size=10,
    )
    wrapper._mlm_probability = 0.3
    wrapper._adaptive_step_counter = 0
    wrapper.model = SimpleNamespace(training=False, config=SimpleNamespace(vocab_size=10))

    # Batch WITHOUT maskable_mask -> must go straight to the parent forward.
    batch_no_mask = {
        "input_ids": torch.zeros(2, 4, dtype=torch.long),
        "labels": torch.zeros(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }
    wrapper.forward(batch_no_mask)
    assert parent_calls["count"] == 1
    # Counter does not advance when the adaptive path is not taken.
    assert wrapper._adaptive_step_counter == 0
