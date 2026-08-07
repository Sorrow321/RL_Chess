// Python bindings for the engine layer.
//
// Enough surface for Python-side tooling and the eval harness (docs/06) to work
// before the batched Runner (docs/02) exists: generate legal moves, push one,
// ask whether the game is over and why, and run perft.
//
// Moves cross this boundary as UCI strings. That is deliberately not the hot
// path — the hot path is the Runner, which stays entirely in C++ and never
// serialises a move. Anything here that looks slow (FEN in, FEN out) is a
// tooling convenience, not something self-play will call per node.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "game.hpp"
#include "perft.hpp"

// --- encoder seam ------------------------------------------------------------
// The plane encoder and the move<->index encoder (docs/01-engine.md) are a
// separate work item and live in their own header. If cpp/encoding.hpp exists
// and defines `void az::register_encoding_bindings(pybind11::module_&)`, its
// bindings — encode_batch, move_to_index, index_to_move, and any index-based
// push overload — join this same module automatically, with no edit here.
#if __has_include("encoding.hpp")
#include "encoding.hpp"
#define AZ_HAS_ENCODING 1
#endif

namespace py = pybind11;

namespace {

chess::Board board_from_fen(const std::string& fen) {
    chess::Board board;
    if (!board.setFen(fen)) throw std::invalid_argument("bad FEN: " + fen);
    return board;
}

// UCI in, legal move out. Resolving against the generated movelist rather than
// trusting uci::uciToMove is the point: it rejects illegal and malformed input
// instead of corrupting the board with it.
chess::Move move_from_uci(const chess::Board& board, const std::string& uci) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    for (const auto& move : moves) {
        if (chess::uci::moveToUci(move, board.chess960()) == uci) return move;
    }
    throw std::invalid_argument("illegal move " + uci + " in position " + board.getFen());
}

std::vector<std::string> uci_list(const chess::Board& board) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    std::vector<std::string> out;
    out.reserve(moves.size());
    for (const auto& move : moves) out.push_back(chess::uci::moveToUci(move, board.chess960()));
    return out;
}

}  // namespace

PYBIND11_MODULE(chess_engine, m) {
    m.doc() = "Chess engine: perft-validated movegen, rules and draw adjudication";

    m.attr("MAX_PLIES") = az::kMaxPlies;
    m.attr("STARTPOS")  = std::string(chess::constants::STARTPOS);

    py::enum_<az::Termination>(m, "Termination")
        .value("NONE", az::Termination::NONE)
        .value("CHECKMATE", az::Termination::CHECKMATE)
        .value("STALEMATE", az::Termination::STALEMATE)
        .value("FIFTY_MOVE", az::Termination::FIFTY_MOVE)
        .value("THREEFOLD", az::Termination::THREEFOLD)
        .value("INSUFFICIENT_MATERIAL", az::Termination::INSUFFICIENT_MATERIAL)
        .value("PLY_CAP", az::Termination::PLY_CAP);

    py::class_<az::Outcome>(m, "Outcome")
        .def_readonly("reason", &az::Outcome::reason)
        // +1 win / 0 draw / -1 loss, always for the side to move.
        .def_readonly("value", &az::Outcome::value)
        .def_property_readonly("over", &az::Outcome::over)
        .def("__repr__", [](const az::Outcome& o) {
            return "<Outcome " + std::string(az::termination_name(o.reason)) + " value=" +
                   std::to_string(o.value) + ">";
        });

    // --- stateless helpers (tooling, tests, eval scripts) ---
    m.def(
        "legal_moves", [](const std::string& fen) { return uci_list(board_from_fen(fen)); },
        py::arg("fen"), "Legal moves in UCI notation.");

    m.def(
        "push",
        [](const std::string& fen, const std::string& uci) {
            auto board = board_from_fen(fen);
            board.makeMove(move_from_uci(board, uci));
            return board.getFen();
        },
        py::arg("fen"), py::arg("uci"),
        "FEN after playing a legal move. Raises ValueError if the move is illegal.");

    m.def(
        "is_legal",
        [](const std::string& fen, const std::string& uci) {
            const auto board = board_from_fen(fen);
            const auto list  = uci_list(board);
            return std::find(list.begin(), list.end(), uci) != list.end();
        },
        py::arg("fen"), py::arg("uci"));

    m.def(
        "perft",
        [](const std::string& fen, int depth, int threads) {
            const auto board = board_from_fen(fen);
            py::gil_scoped_release release;
            return az::perft_threaded(board, depth, threads);
        },
        py::arg("fen"), py::arg("depth"), py::arg("threads") = 1);

    m.def(
        "perft_divide",
        [](const std::string& fen, int depth) {
            auto board = board_from_fen(fen);
            std::map<std::string, std::uint64_t> out;
            for (auto& [uci, nodes] : az::perft_divide(board, depth)) out[uci] = nodes;
            return out;
        },
        py::arg("fen"), py::arg("depth"), "Per-root-move node counts; the tool for locating a perft mismatch.");

    // --- Game: a position plus the history the draw rules need ---
    py::class_<az::Game>(m, "Game")
        .def(py::init<const std::string&, int>(), py::arg("fen") = std::string(chess::constants::STARTPOS),
             py::arg("max_plies") = az::kMaxPlies)
        .def("reset", [](az::Game& g, const std::string& fen) { g.reset(fen); },
             py::arg("fen") = std::string(chess::constants::STARTPOS))
        .def_property_readonly("fen", &az::Game::fen)
        .def_property_readonly("ply", &az::Game::ply)
        .def_property_readonly("max_plies", &az::Game::max_plies)
        .def_property_readonly("hash", &az::Game::hash)
        .def_property_readonly("halfmove_clock", [](const az::Game& g) { return g.board().halfMoveClock(); })
        .def_property_readonly("white_to_move",
                               [](const az::Game& g) { return g.side_to_move() == chess::Color::WHITE; })
        .def_property_readonly("in_check", [](const az::Game& g) { return g.board().inCheck(); })
        .def("legal_moves", [](const az::Game& g) { return uci_list(g.board()); })
        .def("push", [](az::Game& g, const std::string& uci) { g.push(move_from_uci(g.board(), uci)); },
             py::arg("uci"))
        .def("outcome", &az::Game::outcome)
        .def("repetition_count", &az::Game::repetition_count)
        .def("__repr__", [](const az::Game& g) {
            return "<Game ply=" + std::to_string(g.ply()) + " fen='" + g.fen() + "'>";
        });

#ifdef AZ_HAS_ENCODING
    az::register_encoding_bindings(m);
#endif
}
