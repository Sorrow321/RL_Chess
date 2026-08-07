// Perft: the trust anchor for the movegen.
//
// Nothing else in this project is allowed to be trusted until these numbers are
// exact. See tests/perft_test.cpp for the suite and docs/01-engine.md for why.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <chess.hpp>

namespace az {

// Bulk-counted perft: at depth 1 the movelist size *is* the answer, which is
// what the published node counts assume.
inline std::uint64_t perft(chess::Board& board, int depth) {
    if (depth <= 0) return 1;

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (depth == 1) return moves.size();

    std::uint64_t nodes = 0;
    for (const auto& move : moves) {
        board.makeMove(move);
        nodes += perft(board, depth - 1);
        board.unmakeMove(move);
    }
    return nodes;
}

// Per-root-move breakdown. The only tool that matters when a count is wrong:
// diff against a reference engine's `go perft` and recurse into the move that
// disagrees.
inline std::vector<std::pair<std::string, std::uint64_t>> perft_divide(chess::Board& board, int depth) {
    std::vector<std::pair<std::string, std::uint64_t>> out;
    if (depth <= 0) return out;

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    for (const auto& move : moves) {
        board.makeMove(move);
        const std::uint64_t nodes = depth == 1 ? 1 : perft(board, depth - 1);
        board.unmakeMove(move);
        out.emplace_back(chess::uci::moveToUci(move, board.chess960()), nodes);
    }
    return out;
}

// Root-split parallel perft. Each thread gets its own Board copy, so this
// shares no state and cannot change the answer — it only makes the depth-6
// entries of the suite finish in seconds instead of a coffee break.
inline std::uint64_t perft_threaded(const chess::Board& board, int depth, int threads) {
    if (depth <= 1 || threads <= 1) {
        chess::Board local = board;
        return perft(local, depth);
    }

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    const int n_workers = std::min<int>(threads, moves.size());
    std::vector<std::uint64_t> counts(moves.size(), 0);
    std::atomic<int> next{0};

    std::vector<std::thread> pool;
    pool.reserve(n_workers);
    for (int t = 0; t < n_workers; ++t) {
        pool.emplace_back([&] {
            for (int i = next.fetch_add(1); i < moves.size(); i = next.fetch_add(1)) {
                chess::Board local = board;
                local.makeMove(moves[i]);
                counts[i] = perft(local, depth - 1);
            }
        });
    }
    for (auto& th : pool) th.join();

    std::uint64_t total = 0;
    for (const auto c : counts) total += c;
    return total;
}

}  // namespace az
