// Position -> planes and move <-> index encoders (docs/01-engine.md).
//
// Everything here is oriented to the side to move: when black moves, ranks are
// flipped (sq ^ 56, files untouched) and piece colors are swapped, so the net
// always sees "my pawns advance toward rank 8". One implementation serves both
// training and inference — this header is the single source of truth, and the
// pure-Python reference in tests/encoding_test.py exists only to check it.
//
// Plane layout, 19 x 8 x 8, uint8, plane-major (out[plane*64 + rank*8 + file],
// rank 0 = the mover's back rank):
//   0-5   own P N B R Q K          13-16 castling: own K, own Q, opp K, opp Q
//   6-11  opponent P N B R Q K     17    en-passant file (whole file column)
//   12    all ones                 18    min(halfmove clock, 100), constant
// Plane 17 follows the movegen's en-passant square, which chess-library keeps
// only when an en-passant capture is actually legal (validated in setFen and
// makeMove) — python-chess `fen(en_passant="legal")` semantics.
//
// Move index = from_square * 73 + type, both squares oriented to the mover:
//   0-55  queen-like: direction * 7 + (distance - 1), directions clockwise
//         from the mover's forward: N NE E SE S SW W NW
//   56-63 knight: clockwise from NNE:
//         (+1,+2) (+2,+1) (+2,-1) (+1,-2) (-1,-2) (-2,-1) (-2,+1) (-1,+2)
//   64-72 underpromotion: 64 + piece * 3 + direction, piece N=0 B=1 R=2,
//         direction forward=0 capture-left=1 capture-right=2 (left = toward
//         file a in the mover's frame). Queen promotions ride the queen moves.
// Castling is the king's actual two-square move (e1g1 / e1c1 as queen-like E/W
// distance 2), not the library's internal king-takes-rook representation.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <chess.hpp>

namespace az {

inline constexpr int kNumPlanes  = 19;
inline constexpr int kPlaneBytes = kNumPlanes * 64;
inline constexpr int kMoveTypes  = 73;
inline constexpr int kPolicySize = 64 * kMoveTypes;  // 4672

// The one flip everything shares: rank mirror for black, identity for white.
inline chess::Square oriented(chess::Square sq, chess::Color stm) noexcept {
    return stm == chess::Color::WHITE ? sq : chess::Square(sq.index() ^ 56);
}

// Writes kPlaneBytes of uint8 into out (caller-provided, e.g. a numpy buffer).
inline void encode_position(const chess::Board& board, std::uint8_t* out) {
    std::memset(out, 0, kPlaneBytes);
    const chess::Color stm = board.sideToMove();

    for (int rel = 0; rel < 2; ++rel) {
        const chess::Color color = rel == 0 ? stm : ~stm;
        for (int pt = 0; pt < 6; ++pt) {
            auto bb             = board.pieces(chess::PieceType(static_cast<chess::PieceType::underlying>(pt)), color);
            std::uint8_t* plane = out + (rel * 6 + pt) * 64;
            while (!bb.empty()) plane[oriented(chess::Square(bb.pop()), stm).index()] = 1;
        }
    }

    std::memset(out + 12 * 64, 1, 64);

    using Side    = chess::Board::CastlingRights::Side;
    const auto cr = board.castlingRights();
    if (cr.has(stm, Side::KING_SIDE)) std::memset(out + 13 * 64, 1, 64);
    if (cr.has(stm, Side::QUEEN_SIDE)) std::memset(out + 14 * 64, 1, 64);
    if (cr.has(~stm, Side::KING_SIDE)) std::memset(out + 15 * 64, 1, 64);
    if (cr.has(~stm, Side::QUEEN_SIDE)) std::memset(out + 16 * 64, 1, 64);

    if (board.enpassantSq() != chess::Square::NO_SQ) {
        const int file = static_cast<int>(board.enpassantSq().file());
        for (int rank = 0; rank < 8; ++rank) out[17 * 64 + rank * 8 + file] = 1;
    }

    const auto clock = board.halfMoveClock();
    std::memset(out + 18 * 64, static_cast<std::uint8_t>(clock < 100 ? clock : 100), 64);
}

namespace encoding_detail {

struct Delta {
    int df, dr;
};
// Queen directions clockwise from N; knight offsets clockwise from NNE.
inline constexpr Delta kQueenDirs[8]  = {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
inline constexpr Delta kKnightDirs[8] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};

}  // namespace encoding_detail

// The move must be one of `board`'s legal moves; anything else is a logic
// error upstream and throws rather than returning a wrong index.
inline int move_to_index(const chess::Board& board, chess::Move move) {
    const chess::Color stm = board.sideToMove();
    const chess::Square from = oriented(move.from(), stm);

    // The library encodes castling as king-takes-rook; the policy encodes the
    // king's actual destination (file g or c on the king's own rank).
    chess::Square to_abs = move.to();
    if (move.typeOf() == chess::Move::CASTLING) {
        const bool kingside = move.to() > move.from();
        to_abs              = chess::Square(chess::File(kingside ? 6 : 2), move.from().rank());
    }
    const chess::Square to = oriented(to_abs, stm);

    const int df = static_cast<int>(to.file()) - static_cast<int>(from.file());
    const int dr = static_cast<int>(to.rank()) - static_cast<int>(from.rank());

    int type = -1;
    if (move.typeOf() == chess::Move::PROMOTION && move.promotionType() != chess::PieceType::QUEEN) {
        const int piece = move.promotionType() == chess::PieceType::KNIGHT ? 0
                        : move.promotionType() == chess::PieceType::BISHOP ? 1
                                                                           : 2;
        const int dir   = df == 0 ? 0 : (df < 0 ? 1 : 2);
        type            = 64 + piece * 3 + dir;
    } else if (df != 0 && dr != 0 && std::abs(df) != std::abs(dr)) {
        for (int i = 0; i < 8; ++i) {
            if (encoding_detail::kKnightDirs[i].df == df && encoding_detail::kKnightDirs[i].dr == dr) {
                type = 56 + i;
                break;
            }
        }
    } else {
        const int dist = std::abs(df) > std::abs(dr) ? std::abs(df) : std::abs(dr);
        const int uf   = (df > 0) - (df < 0);
        const int ur   = (dr > 0) - (dr < 0);
        for (int i = 0; i < 8; ++i) {
            if (encoding_detail::kQueenDirs[i].df == uf && encoding_detail::kQueenDirs[i].dr == ur) {
                type = i * 7 + dist - 1;
                break;
            }
        }
    }
    if (type < 0) {
        throw std::logic_error("unencodable move " + chess::uci::moveToUci(move, board.chess960()) + " in " +
                               board.getFen());
    }
    return from.index() * kMoveTypes + type;
}

// Inverse via the legal movelist: the index space is only meaningful over
// legal moves, and resolving against them rejects garbage indices for free.
// O(moves) — fine for tooling; the runner only ever maps in the forward
// direction (legal move -> logit slot).
inline chess::Move index_to_move(const chess::Board& board, int index) {
    if (index < 0 || index >= kPolicySize) {
        throw std::out_of_range("move index " + std::to_string(index) + " outside [0, " +
                                std::to_string(kPolicySize) + ")");
    }
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    for (const auto& move : moves) {
        if (move_to_index(board, move) == index) return move;
    }
    throw std::invalid_argument("move index " + std::to_string(index) + " is not legal in " + board.getFen());
}

}  // namespace az

// --- pybind11 bindings -------------------------------------------------------
// Compiled only inside the pybind module (chess_engine.cpp includes pybind11
// before this header; the C++ test binaries never define PYBIND11_MODULE and
// get just the encoders above).
#ifdef PYBIND11_MODULE

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>

#include "game.hpp"

namespace az {

namespace encoding_detail {

inline chess::Board bound_board_from_fen(const std::string& fen) {
    chess::Board board;
    if (!board.setFen(fen)) throw std::invalid_argument("bad FEN: " + fen);
    return board;
}

inline chess::Move bound_move_from_uci(const chess::Board& board, const std::string& uci) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    for (const auto& move : moves) {
        if (chess::uci::moveToUci(move, board.chess960()) == uci) return move;
    }
    throw std::invalid_argument("illegal move " + uci + " in position " + board.getFen());
}

}  // namespace encoding_detail

inline void register_encoding_bindings(pybind11::module_& m) {
    namespace py = pybind11;

    m.attr("NUM_PLANES")  = kNumPlanes;
    m.attr("POLICY_SIZE") = kPolicySize;

    m.def(
        "encode_batch",
        [](const py::sequence& positions) {
            const py::ssize_t n = py::len(positions);
            py::array_t<std::uint8_t> out({n, static_cast<py::ssize_t>(kNumPlanes), static_cast<py::ssize_t>(8),
                                           static_cast<py::ssize_t>(8)});
            std::uint8_t* data = out.mutable_data();
            py::ssize_t i      = 0;
            for (const auto& item : positions) {
                if (py::isinstance<Game>(item)) {
                    encode_position(item.cast<const Game&>().board(), data + i * kPlaneBytes);
                } else {
                    encode_position(encoding_detail::bound_board_from_fen(item.cast<std::string>()),
                                    data + i * kPlaneBytes);
                }
                ++i;
            }
            return out;
        },
        py::arg("positions"),
        "(B,19,8,8) uint8 planes for a sequence of FEN strings and/or Game objects, "
        "oriented to the side to move.");

    m.def(
        "move_to_index",
        [](const std::string& fen, const std::string& uci) {
            const auto board = encoding_detail::bound_board_from_fen(fen);
            return move_to_index(board, encoding_detail::bound_move_from_uci(board, uci));
        },
        py::arg("fen"), py::arg("uci"), "Policy index (0..4671) of a legal move.");

    m.def(
        "index_to_move",
        [](const std::string& fen, int index) {
            const auto board = encoding_detail::bound_board_from_fen(fen);
            return chess::uci::moveToUci(index_to_move(board, index), board.chess960());
        },
        py::arg("fen"), py::arg("index"),
        "UCI move for a policy index. Raises ValueError if the index is not a legal move.");

    m.def(
        "legal_move_indices",
        [](const std::string& fen) {
            const auto board = encoding_detail::bound_board_from_fen(fen);
            chess::Movelist moves;
            chess::movegen::legalmoves(moves, board);
            std::vector<int> out;
            out.reserve(moves.size());
            for (const auto& move : moves) out.push_back(move_to_index(board, move));
            return out;
        },
        py::arg("fen"), "Policy indices aligned with legal_moves(fen).");

    // Index-based overload of push(); the string overload is registered first
    // in chess_engine.cpp and pybind dispatches on argument type.
    m.def(
        "push",
        [](const std::string& fen, int index) {
            auto board = encoding_detail::bound_board_from_fen(fen);
            board.makeMove(index_to_move(board, index));
            return board.getFen();
        },
        py::arg("fen"), py::arg("index"),
        "FEN after playing a legal move given as a policy index.");
}

}  // namespace az

#endif  // PYBIND11_MODULE
