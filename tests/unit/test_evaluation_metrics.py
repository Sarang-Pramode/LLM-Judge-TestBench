"""Golden-value tests for src.evaluation.metrics.

Every metric is exercised with a tiny hand-computed example so a
future edit that accidentally flips a sign / normalisation is caught
by diffing against the expected numbers.
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import (
    confusion_matrix,
    exact_match_rate,
    mean_absolute_error,
    off_by_2_rate,
    off_by_3_plus_rate,
    score_distribution,
    severity_aware_alignment,
    spearman_correlation,
    weighted_kappa,
    within_1_rate,
)

# ---------------------------------------------------------------------------
# Rate metrics
# ---------------------------------------------------------------------------


class TestRateMetrics:
    def test_exact_match_counts_equal_pairs(self) -> None:
        pairs = [(5, 5), (4, 5), (3, 3), (1, 5)]
        assert exact_match_rate(pairs) == pytest.approx(0.5)

    def test_within_1_includes_exact_matches(self) -> None:
        pairs = [(5, 5), (4, 5), (2, 4), (1, 5)]
        # 5==5 (0), 4 vs 5 (1), 2 vs 4 (2), 1 vs 5 (4) -> 2/4
        assert within_1_rate(pairs) == pytest.approx(0.5)

    def test_off_by_2_is_exact_bucket_not_cumulative(self) -> None:
        pairs = [(1, 3), (2, 4), (5, 5), (1, 4)]
        # distances 2, 2, 0, 3 -> exact-2 count = 2 / 4
        assert off_by_2_rate(pairs) == pytest.approx(0.5)
        # and off_by_3_plus is 1/4 for the (1,4)
        assert off_by_3_plus_rate(pairs) == pytest.approx(0.25)

    def test_off_by_3_plus_is_cumulative_tail(self) -> None:
        pairs = [(1, 5), (2, 5), (3, 5)]
        # distances 4, 3, 2 -> >=3 count = 2/3
        assert off_by_3_plus_rate(pairs) == pytest.approx(2 / 3)

    def test_empty_input_returns_zero(self) -> None:
        for fn in (
            exact_match_rate,
            within_1_rate,
            off_by_2_rate,
            off_by_3_plus_rate,
            mean_absolute_error,
        ):
            assert fn([]) == 0.0

    def test_mean_absolute_error(self) -> None:
        pairs = [(1, 5), (5, 1), (3, 3)]
        # |1-5|=4, |5-1|=4, |3-3|=0 -> 8/3
        assert mean_absolute_error(pairs) == pytest.approx(8 / 3)

    def test_rate_buckets_partition_sample(self) -> None:
        # The three distance buckets + "within-1 minus exact" must sum to
        # 1.0. This catches any accidental off-by-one bucket mistake.
        pairs = [(1, 1), (2, 1), (2, 4), (5, 5), (3, 1)]
        # distances 0, 1, 2, 0, 2
        exact = exact_match_rate(pairs)
        within1_only = within_1_rate(pairs) - exact
        d2 = off_by_2_rate(pairs)
        d3_plus = off_by_3_plus_rate(pairs)
        assert exact + within1_only + d2 + d3_plus == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Severity-aware alignment
# ---------------------------------------------------------------------------


class TestSeverityAwareAlignment:
    def test_perfect_agreement_is_1(self) -> None:
        assert severity_aware_alignment([(3, 3), (5, 5)]) == pytest.approx(1.0)

    def test_max_disagreement_is_0(self) -> None:
        assert severity_aware_alignment([(1, 5), (5, 1)]) == pytest.approx(0.0)

    def test_standard_distance_weights(self) -> None:
        # (1,1)=1.0, (2,3)=0.75, (1,3)=0.40, (1,4)=0.10, (1,5)=0.00
        pairs = [(1, 1), (2, 3), (1, 3), (1, 4), (1, 5)]
        expected = (1.0 + 0.75 + 0.40 + 0.10 + 0.00) / 5
        assert severity_aware_alignment(pairs) == pytest.approx(expected)

    def test_custom_weights_override(self) -> None:
        pairs = [(1, 1), (1, 2)]
        weights = {0: 1.0, 1: 0.5}
        assert severity_aware_alignment(pairs, weights=weights) == pytest.approx(0.75)

    def test_missing_weight_key_falls_back_to_zero(self) -> None:
        pairs = [(1, 5)]  # distance 4
        assert severity_aware_alignment(pairs, weights={0: 1.0}) == pytest.approx(0.0)

    def test_empty_input_is_zero(self) -> None:
        assert severity_aware_alignment([]) == 0.0


# ---------------------------------------------------------------------------
# Score distribution / confusion matrix
# ---------------------------------------------------------------------------


class TestDistributions:
    def test_score_distribution_fills_full_range(self) -> None:
        dist = score_distribution([1, 1, 3])
        assert dist == {1: 2, 2: 0, 3: 1, 4: 0, 5: 0}

    def test_score_distribution_clamps_out_of_range(self) -> None:
        dist = score_distribution([-1, 0, 6, 99])
        # -1 and 0 clamp to 1, 6 and 99 clamp to 5
        assert dist == {1: 2, 2: 0, 3: 0, 4: 0, 5: 2}

    def test_confusion_matrix_orientation(self) -> None:
        # judge=5, human=3 -> row index 2 (human-3), col index 4 (judge-5)
        matrix = confusion_matrix([(5, 3)])
        assert matrix[2][4] == 1
        total = sum(sum(row) for row in matrix)
        assert total == 1

    def test_confusion_matrix_fills_shape(self) -> None:
        matrix = confusion_matrix([])
        assert len(matrix) == 5
        assert all(len(row) == 5 for row in matrix)
        assert all(v == 0 for row in matrix for v in row)

    def test_confusion_matrix_matches_distribution(self) -> None:
        pairs = [(1, 1), (3, 3), (5, 5), (2, 4)]
        matrix = confusion_matrix(pairs)
        # Diagonal hits
        assert matrix[0][0] == 1  # (1,1)
        assert matrix[2][2] == 1  # (3,3)
        assert matrix[4][4] == 1  # (5,5)
        # Off-diagonal: human=4, judge=2 -> row 3, col 1
        assert matrix[3][1] == 1


# ---------------------------------------------------------------------------
# Weighted kappa
# ---------------------------------------------------------------------------


class TestWeightedKappa:
    def test_perfect_agreement_is_one(self) -> None:
        pairs = [(1, 1), (3, 3), (5, 5), (2, 2)]
        assert weighted_kappa(pairs) == pytest.approx(1.0)

    def test_constant_judge_is_zero_kappa(self) -> None:
        # With a constant judge the observed joint distribution equals
        # the independence distribution, so observed agreement and
        # chance agreement are identical -> kappa == 0 (well-defined).
        pairs = [(3, 1), (3, 5), (3, 3)]
        assert weighted_kappa(pairs) == pytest.approx(0.0)

    def test_constant_on_both_sides_returns_none(self) -> None:
        # Both marginals have a single cell -> kappa numerator and
        # denominator are both 0 and the metric is undefined.
        pairs = [(3, 3), (3, 3), (3, 3)]
        assert weighted_kappa(pairs) is None

    def test_fewer_than_two_pairs_returns_none(self) -> None:
        assert weighted_kappa([(1, 1)]) is None
        assert weighted_kappa([]) is None

    def test_linear_and_quadratic_differ_on_mixed_data(self) -> None:
        # A sample with mostly-near-misses plus one large miss: the
        # large miss dominates the quadratic numerator far more than
        # the linear numerator, so the two kappas have to diverge.
        pairs = [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (2, 3),
            (4, 3),
            (1, 5),
        ]
        linear = weighted_kappa(pairs, weights="linear")
        quad = weighted_kappa(pairs, weights="quadratic")
        assert linear is not None and quad is not None
        assert -1.0 <= linear <= 1.0
        assert -1.0 <= quad <= 1.0
        assert linear != quad

    def test_chance_level_agreement_is_zero(self) -> None:
        # Marginals are balanced on both sides; observed joint
        # matches expected joint -> kappa == 0.
        pairs = [(1, 1), (1, 5), (5, 1), (5, 5)]
        result = weighted_kappa(pairs)
        assert result is not None
        assert abs(result) < 1e-9

    def test_perfect_anti_agreement_is_minus_one(self) -> None:
        # All mass on opposite corners -> kappa bottoms out at -1.
        pairs = [(1, 5), (5, 1), (1, 5), (5, 1)]
        result = weighted_kappa(pairs)
        assert result is not None
        assert result == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------


class TestSpearman:
    def test_perfect_rank_agreement_is_one(self) -> None:
        pairs = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        assert spearman_correlation(pairs) == pytest.approx(1.0)

    def test_perfect_inverse_rank_is_minus_one(self) -> None:
        pairs = [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)]
        assert spearman_correlation(pairs) == pytest.approx(-1.0)

    def test_constant_series_returns_none(self) -> None:
        assert spearman_correlation([(3, 1), (3, 5), (3, 3)]) is None
        assert spearman_correlation([(1, 2), (2, 2), (3, 2)]) is None

    def test_single_pair_returns_none(self) -> None:
        assert spearman_correlation([(3, 3)]) is None
        assert spearman_correlation([]) is None

    def test_ties_handled_with_average_ranks(self) -> None:
        # Both series have a tie but are monotonic; rho should still be
        # strongly positive (== 1.0 for this symmetric case).
        pairs = [(1, 1), (2, 2), (2, 2), (3, 3)]
        result = spearman_correlation(pairs)
        assert result is not None
        assert result == pytest.approx(1.0)

    def test_against_known_value(self) -> None:
        # Hand-computed:
        # X = [1,2,3,4,5] -> ranks [1,2,3,4,5]
        # Y = [2,1,5,4,3] -> ranks [2,1,5,4,3]
        # Both series have mean=3, var=10; cov on the rank series = 5.
        # Pearson on those ranks = 5/sqrt(100) = 0.5.
        pairs = [(1, 2), (2, 1), (3, 5), (4, 4), (5, 3)]
        result = spearman_correlation(pairs)
        assert result is not None
        assert result == pytest.approx(0.5)

    def test_bounded_in_unit_interval(self) -> None:
        import random

        rng = random.Random(42)
        pairs = [(rng.randint(1, 5), rng.randint(1, 5)) for _ in range(50)]
        rho = spearman_correlation(pairs)
        assert rho is not None
        assert -1.0 - 1e-9 <= rho <= 1.0 + 1e-9
        assert not math.isnan(rho)
