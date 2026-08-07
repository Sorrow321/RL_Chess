// Rules and draw adjudication on top of the embedded movegen.
//
// The movegen library (third_party/chess-library) already implements castling,
// en passant, promotions and Zobrist hashing correctly — perft proves it. What
// it does not give us is a *game*: a position plus the history needed for
// threefold repetition, plus the self-play ply cap, plus one place where the
// draw rules are ordered the way we want them. That is what Game is.
#pragma once

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <chess.hpp>

namespace az {

// Hard draw for self-play: keeps degenerate early-training games from running
// forever. Counted in plies from the start of *this* Game object.
inline constexpr int kMaxPlies = 250;

enum class Termination : std::uint8_t {
    NONE = 0,
    CHECKMATE,
    STALEMATE,
    FIFTY_MOVE,
    THREEFOLD,
    INSUFFICIENT_MATERIAL,
    PLY_CAP,
};

inline const char* termination_name(Termination t) {
    switch (t) {
        case Termination::NONE: return "none";
        case Termination::CHECKMATE: return "checkmate";
        case Termination::STALEMATE: return "stalemate";
        case Termination::FIFTY_MOVE: return "fifty_move";
        case Termination::THREEFOLD: return "threefold";
        case Termination::INSUFFICIENT_MATERIAL: return "insufficient_material";
        case Termination::PLY_CAP: return "ply_cap";
    }
    return "unknown";
}

// Result of a finished game, always from the perspective of the side to move in
// the terminal position — same convention as the planes and the value head, so
// the training target is `value` with no sign bookkeeping at the call site.
// Only CHECKMATE is ever non-zero, and it is always a loss: you are mated on
// your own turn.
struct Outcome {
    Termination reason = Termination::NONE;
    int value          = 0;  // +1 win / 0 draw / -1 loss for the side to move

    [[nodiscard]] bool over() const noexcept { return reason != Termination::NONE; }
};

class Game {
   public:
    // max_plies is a parameter rather than a hard constant so the eval harness
    // can adjudicate on a different budget than self-play — and so the cap is
    // testable without playing 250 non-repeating plies.
    explicit Game(std::string_view fen = chess::constants::STARTPOS, int max_plies = kMaxPlies)
        : max_plies_(max_plies) {
        reset(fen);
    }

    // Throws std::invalid_argument on an unparseable FEN.
    void reset(std::string_view fen = chess::constants::STARTPOS) {
        if (!board_.setFen(fen)) {
            throw std::invalid_argument("bad FEN: " + std::string(fen));
        }
        start_ply_ = 0;
        repetition_.clear();
        repetition_.push_back(board_.hash());
    }

    [[nodiscard]] const chess::Board& board() const noexcept { return board_; }
    [[nodiscard]] std::string fen() const { return board_.getFen(); }
    [[nodiscard]] std::uint64_t hash() const noexcept { return board_.hash(); }
    [[nodiscard]] chess::Color side_to_move() const noexcept { return board_.sideToMove(); }

    // Plies played on this object. Deliberately not the FEN's move counter: the
    // cap bounds *our* game, not the whole history of a position we resumed.
    [[nodiscard]] int ply() const noexcept { return start_ply_; }

    [[nodiscard]] chess::Movelist legal_moves() const {
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board_);
        return moves;
    }

    // The move must be legal in the current position (assert-checked in debug
    // builds only — the runner generates its moves from legal_moves(), and the
    // Python bindings validate before calling here).
    void push(chess::Move move) {
        assert(chess::movegen::isLegal(board_, move));

        board_.makeMove(move);
        ++start_ply_;

        // A move that resets the halfmove clock (capture or pawn move) is
        // irreversible: no earlier position can ever recur, so the repetition
        // window starts fresh. This also bounds the vector at 100 entries.
        if (board_.halfMoveClock() == 0) repetition_.clear();
        repetition_.push_back(board_.hash());
    }

    // How many times the current position has occurred in the repetition window,
    // counting the current occurrence. 3 == threefold.
    //
    // Ours rather than the library's isRepetition(): the runner (doc 02) needs
    // repetition counts along a search path, and the semantics should be defined
    // by one function we own and test, not by a library internal.
    [[nodiscard]] int repetition_count() const noexcept {
        const std::uint64_t key = board_.hash();
        int n                   = 0;
        // Same side to move implies same parity, so step by 2 from the back.
        for (int i = static_cast<int>(repetition_.size()) - 1; i >= 0; i -= 2) {
            if (repetition_[i] == key) ++n;
        }
        return n;
    }

    // Draw rules ordered so that mate wins every tie: a position can be
    // simultaneously checkmate and 50-move/threefold, and it is checkmate.
    [[nodiscard]] Outcome outcome() const {
        if (!chess::movegen::anylegalmoves(board_)) {
            return board_.inCheck() ? Outcome{Termination::CHECKMATE, -1}
                                    : Outcome{Termination::STALEMATE, 0};
        }
        if (board_.halfMoveClock() >= 100) return {Termination::FIFTY_MOVE, 0};
        if (board_.isInsufficientMaterial()) return {Termination::INSUFFICIENT_MATERIAL, 0};
        if (repetition_count() >= 3) return {Termination::THREEFOLD, 0};
        if (start_ply_ >= max_plies_) return {Termination::PLY_CAP, 0};
        return {};
    }

    [[nodiscard]] int max_plies() const noexcept { return max_plies_; }

   private:
    chess::Board board_;
    int max_plies_ = kMaxPlies;
    int start_ply_ = 0;
    std::vector<std::uint64_t> repetition_;
};

}  // namespace az
