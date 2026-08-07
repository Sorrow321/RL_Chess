// The move encoding's perft (docs/01-engine.md): for every legal move in
// every position of a depth-4 walk from the five perft suite positions,
// index_to_move(move_to_index(m)) == m and no two legal moves in a position
// share an index. Plus a handful of hand-computed convention pins so a silent
// change of direction ordering reads as "e2e4 is no longer 877", not as a
// mysteriously retrained-from-scratch net.
//
// The plane encoder is deliberately not tested here: its oracle is the pure
// Python reference in tests/encoding_test.py, which shares no code with the
// C++ encoder.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "encoding.hpp"

namespace {

constexpr const char* kSuite[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
};

std::atomic<std::uint64_t> g_failures{0};
std::atomic<std::uint64_t> g_positions{0};
std::atomic<std::uint64_t> g_moves{0};
std::mutex g_print_mutex;

void fail(const std::string& what) {
    // Cap the noise: the count is exact, the messages are diagnostics.
    if (g_failures.fetch_add(1) < 20) {
        std::lock_guard<std::mutex> lock(g_print_mutex);
        std::cout << "  FAIL: " << what << "\n";
    }
}

// Checks one position and recurses. Collision detection via sorted indices;
// round-trip through the real az::index_to_move, movelist regeneration and
// all — this is the exposed function under test, not a shortcut of it.
void walk(chess::Board& board, int depth) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    g_positions.fetch_add(1, std::memory_order_relaxed);
    g_moves.fetch_add(moves.size(), std::memory_order_relaxed);

    int indices[256];
    for (int i = 0; i < moves.size(); ++i) {
        const int idx = az::move_to_index(board, moves[i]);
        indices[i]    = idx;
        if (idx < 0 || idx >= az::kPolicySize) {
            fail("index " + std::to_string(idx) + " out of range for " +
                 chess::uci::moveToUci(moves[i]) + " in " + board.getFen());
            continue;
        }
        if (az::index_to_move(board, idx) != moves[i]) {
            fail("round trip broke for " + chess::uci::moveToUci(moves[i]) + " (index " +
                 std::to_string(idx) + ") in " + board.getFen());
        }
    }

    std::sort(indices, indices + moves.size());
    if (std::adjacent_find(indices, indices + moves.size()) != indices + moves.size()) {
        fail("index collision in " + board.getFen());
    }

    if (depth <= 0) return;
    for (const auto& move : moves) {
        board.makeMove(move);
        walk(board, depth - 1);
        board.unmakeMove(move);
    }
}

// Hand-computed pins of the encoding convention (see encoding.hpp header
// comment for the tables these numbers come from).
void pin(const std::string& fen, const std::string& uci, int want) {
    chess::Board board;
    board.setFen(fen);
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    for (const auto& move : moves) {
        if (chess::uci::moveToUci(move) == uci) {
            const int got = az::move_to_index(board, move);
            if (got != want) {
                fail("pin " + uci + " in " + fen + ": got " + std::to_string(got) + ", want " +
                     std::to_string(want));
            }
            return;
        }
    }
    fail("pin move " + uci + " is not legal in " + fen);
}

void convention_pins() {
    const std::string startpos = kSuite[0];
    const std::string castlepos = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
    const std::string castlepos_b = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1";

    pin(startpos, "e2e4", 12 * 73 + 1);            // N, distance 2          = 877
    pin(startpos, "g1f3", 6 * 73 + 63);            // knight (-1,+2)         = 501
    pin(castlepos, "e1g1", 4 * 73 + 2 * 7 + 1);    // king E, distance 2     = 307
    pin(castlepos, "e1c1", 4 * 73 + 6 * 7 + 1);    // king W, distance 2     = 335
    pin(castlepos_b, "e8g8", 4 * 73 + 2 * 7 + 1);  // same square+type once oriented
    pin("8/P6k/8/8/8/8/7K/8 w - - 0 1", "a7a8n", 48 * 73 + 64);      // fwd underpromo N = 3568
    pin("1n5k/P7/8/8/8/8/7K/8 w - - 0 1", "a7b8r", 48 * 73 + 72);    // capture-right underpromo R
    pin("7k/8/8/8/8/8/p6K/8 b - - 0 1", "a2a1n", 48 * 73 + 64);      // black mirrors to 3568
    pin("7k/4P3/8/8/8/8/8/K7 w - - 0 1", "e7e8q", 52 * 73 + 0);      // queen promo = plain N move
}

}  // namespace

int main(int argc, char** argv) {
    int depth   = 4;
    int threads = static_cast<int>(std::thread::hardware_concurrency());
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--depth" && i + 1 < argc) depth = std::stoi(argv[++i]);
        if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
    }

    convention_pins();

    // Root-split: one job per (suite position, root move) subtree, same
    // pattern as perft_threaded. The five root positions themselves are
    // covered by walk(root, 0) first.
    std::vector<chess::Board> jobs;
    for (const char* fen : kSuite) {
        chess::Board root;
        root.setFen(fen);
        walk(root, 0);

        chess::Movelist moves;
        chess::movegen::legalmoves(moves, root);
        for (const auto& move : moves) {
            root.makeMove(move);
            jobs.push_back(root);
            root.unmakeMove(move);
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> pool;
    const int n_workers = std::max(1, std::min<int>(threads, static_cast<int>(jobs.size())));
    pool.reserve(n_workers);
    for (int t = 0; t < n_workers; ++t) {
        pool.emplace_back([&] {
            for (std::size_t i = next.fetch_add(1); i < jobs.size(); i = next.fetch_add(1)) {
                chess::Board board = jobs[i];
                walk(board, depth - 1);  // the root move consumed one ply
            }
        });
    }
    for (auto& th : pool) th.join();
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::cout << "encoding walk: depth " << depth << ", " << g_positions.load() << " positions, "
              << g_moves.load() << " moves round-tripped in " << elapsed << "s\n";

    if (g_failures.load() != 0) {
        std::cout << "ENCODING TEST FAILED: " << g_failures.load() << " failure(s)\n";
        return 1;
    }
    std::cout << "encoding test OK\n";
    return 0;
}
