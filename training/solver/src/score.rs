pub const T: f32 = 4.0;
pub const INVALID_SCORE: f32 = -1.0;

/// Normalize a raw game score to [-1.0, +1.0] using tanh with temperature T=4.
pub fn normalize(raw_score: i32) -> f32 {
    (raw_score as f32 / T).tanh()
}

/// Returns the normalized score for a given column given the ranked move list.
/// Returns INVALID_SCORE if the column is not in ranked_moves (i.e., column is full).
pub fn score_for_col(col: u32, ranked_moves: &[(u32, i32)]) -> f32 {
    ranked_moves
        .iter()
        .find(|(c, _)| *c == col)
        .map(|(_, s)| normalize(*s))
        .unwrap_or(INVALID_SCORE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_zero() {
        assert!((normalize(0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_positive() {
        let s = normalize(18);
        assert!(s > 0.0 && s <= 1.0);
    }

    #[test]
    fn test_normalize_negative() {
        let s = normalize(-18);
        assert!(s < 0.0 && s >= -1.0);
    }

    #[test]
    fn test_normalize_symmetry() {
        assert!((normalize(5) + normalize(-5)).abs() < 1e-6);
    }

    #[test]
    fn test_score_for_col_found() {
        let moves = vec![(3, 10), (2, 5), (4, -3)];
        let s = score_for_col(3, &moves);
        assert!((s - normalize(10)).abs() < 1e-6);
    }

    #[test]
    fn test_score_for_col_not_found() {
        let moves = vec![(3, 10)];
        assert!((score_for_col(0, &moves) - INVALID_SCORE).abs() < 1e-6);
    }
}
