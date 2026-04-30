"""Pure-logic tests for the ConnectFour engine."""

from backend.app.engine.game import COLS, ROWS, ConnectFour


def fill_column(game: ConnectFour, col: int, count: int) -> None:
    for _ in range(count):
        assert game.drop_piece(col)


class TestDropPiece:
    def test_first_move_lands_on_bottom_row(self):
        game = ConnectFour()
        assert game.drop_piece(3)
        assert game.board[ROWS - 1][3] == 1
        assert game.current_turn == 2

    def test_gravity_stacks_pieces(self):
        game = ConnectFour()
        game.drop_piece(0)  # P1 bottom
        game.drop_piece(0)  # P2 above
        assert game.board[ROWS - 1][0] == 1
        assert game.board[ROWS - 2][0] == 2

    def test_alternates_turns(self):
        game = ConnectFour()
        for col, expected_player in zip([0, 1, 2, 3], [1, 2, 1, 2]):
            assert game.current_turn == expected_player
            game.drop_piece(col)

    def test_drop_into_full_column_returns_false(self):
        game = ConnectFour()
        fill_column(game, 0, ROWS)
        assert game.is_valid_move(0) is False
        assert game.drop_piece(0) is False

    def test_drop_into_invalid_column_returns_false(self):
        game = ConnectFour()
        assert game.drop_piece(-1) is False
        assert game.drop_piece(COLS) is False

    def test_history_records_each_move(self):
        game = ConnectFour()
        game.drop_piece(0)
        game.drop_piece(1)
        assert game.history == [
            {"player": 1, "column": 0},
            {"player": 2, "column": 1},
        ]


class TestWinDetection:
    def test_horizontal_win(self):
        game = ConnectFour()
        # P1 plays cols 0..3 on bottom; P2 stacks col 6 to consume turns.
        for c in range(4):
            game.drop_piece(c)
            if c < 3:
                game.drop_piece(6)
        assert game.winner == 1

    def test_vertical_win(self):
        game = ConnectFour()
        for _ in range(3):
            game.drop_piece(0)  # P1
            game.drop_piece(1)  # P2
        game.drop_piece(0)  # P1's 4th in column 0
        assert game.winner == 1

    def test_diagonal_up_right_win(self):
        """Build a / diagonal for P1 across (5,0)(4,1)(3,2)(2,3)."""
        game = ConnectFour()
        game.drop_piece(0)  # P1 (5,0)
        game.drop_piece(1)  # P2 (5,1)
        game.drop_piece(1)  # P1 (4,1)
        game.drop_piece(2)  # P2 (5,2)
        game.drop_piece(2)  # P1 (4,2)
        game.drop_piece(3)  # P2 (5,3)
        game.drop_piece(2)  # P1 (3,2)
        game.drop_piece(3)  # P2 (4,3)
        game.drop_piece(3)  # P1 (3,3) — not a win yet
        game.drop_piece(0)  # P2 filler (4,0)
        game.drop_piece(3)  # P1 (2,3) — completes / diagonal
        assert game.winner == 1

    def test_diagonal_down_right_win(self):
        """Build a \\ diagonal for P1 across (5,3)(4,2)(3,1)(2,0)."""
        game = ConnectFour.from_history(
            [
                {"column": 3},  # P1 (5,3)
                {"column": 2},  # P2 (5,2)
                {"column": 2},  # P1 (4,2)
                {"column": 1},  # P2 (5,1)
                {"column": 1},  # P1 (4,1) filler
                {"column": 0},  # P2 (5,0)
                {"column": 1},  # P1 (3,1)
                {"column": 0},  # P2 (4,0)
                {"column": 0},  # P1 (3,0) filler
                {"column": 6},  # P2 (5,6)
                {"column": 0},  # P1 (2,0) — completes \ diagonal
            ]
        )
        assert game.winner == 1

    def test_no_winner_after_three_in_a_row(self):
        game = ConnectFour()
        for c in range(3):
            game.drop_piece(c)  # P1
            game.drop_piece(6)  # P2 filler
        assert game.winner is None


class TestDraw:
    def test_is_draw_only_when_top_row_full_and_no_winner(self):
        game = ConnectFour()
        assert game.is_draw() is False
        for c in range(COLS):
            game.board[0][c] = 1 if c % 2 == 0 else 2
        assert game.is_draw() is True

    def test_is_draw_false_if_winner_set(self):
        game = ConnectFour()
        game.winner = 1
        for c in range(COLS):
            game.board[0][c] = 1
        assert game.is_draw() is False


class TestFromHistory:
    def test_replays_history(self):
        original = ConnectFour()
        for c in [3, 3, 4, 4, 5]:
            original.drop_piece(c)

        replayed = ConnectFour.from_history(
            [{"column": c} for c in [3, 3, 4, 4, 5]]
        )
        assert replayed.board == original.board
        assert replayed.current_turn == original.current_turn

    def test_from_history_handles_empty(self):
        replayed = ConnectFour.from_history([])
        assert all(cell == 0 for row in replayed.board for cell in row)
        assert replayed.current_turn == 1


class TestFormatting:
    def test_get_visual_board_includes_header_and_pieces(self):
        game = ConnectFour()
        game.drop_piece(0)
        rendered = game.get_visual_board()
        assert rendered.splitlines()[0].strip().startswith("0")
        assert "X" in rendered

    def test_get_textual_description_lists_columns_bottom_to_top(self):
        game = ConnectFour()
        game.drop_piece(0)  # P1 at bottom of col 0
        game.drop_piece(0)  # P2 above
        desc = game.get_textual_description()
        assert "Column 0: P1, P2" in desc
        assert "Column 1: Empty" in desc
