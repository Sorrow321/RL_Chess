// The trust anchor. Exact node counts on the standard suite, or the build is
// not usable and nothing downstream of it means anything.
//
//   ./perft_test                  full suite (the depths in docs/01-engine.md)
//   ./perft_test --max-depth 4    quick pass, for `ctest` and pre-commit
//   ./perft_test --threads 1      single-threaded (honest nodes/s numbers)
//   ./perft_test --divide "<fen>" 3
//
// Any mismatch anywhere = stop, fix, re-run all.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "perft.hpp"

namespace {

struct Case {
    const char* name;
    const char* fen;
    // Expected node counts for depth 1, 2, 3, ... Intermediate depths are not
    // decoration: when depth 6 is wrong, the shallowest wrong depth is where you
    // start looking, and --divide is small enough to read there.
    std::vector<std::uint64_t> expected;
};

const std::vector<Case>& suite() {
    static const std::vector<Case> cases = {
        {"startpos",
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         {20, 400, 8902, 197281, 4865609, 119060324}},
        {"kiwipete",
         "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
         {48, 2039, 97862, 4085603, 193690690}},
        {"position3",
         "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
         {14, 191, 2812, 43238, 674624, 11030083}},
        {"position4",
         "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -",
         {6, 264, 9467, 422333, 15833292}},
        {"position5",
         "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -",
         {44, 1486, 62379, 2103487, 89941194}},
    };
    return cases;
}

int run_divide(const std::string& fen, int depth) {
    chess::Board board;
    if (!board.setFen(fen)) {
        std::cerr << "bad FEN: " << fen << "\n";
        return 2;
    }
    std::uint64_t total = 0;
    for (const auto& [move, nodes] : az::perft_divide(board, depth)) {
        std::cout << move << ": " << nodes << "\n";
        total += nodes;
    }
    std::cout << "\ntotal: " << total << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int max_depth = 99;
    int threads   = static_cast<int>(std::thread::hardware_concurrency());
    if (threads < 1) threads = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--max-depth" && i + 1 < argc) {
            max_depth = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (arg == "--divide" && i + 2 < argc) {
            const std::string fen = argv[i + 1];
            return run_divide(fen, std::atoi(argv[i + 2]));
        } else {
            std::cerr << "usage: " << argv[0]
                      << " [--max-depth N] [--threads N] [--divide \"<fen>\" DEPTH]\n";
            return 2;
        }
    }

    std::cout << "perft suite (" << threads << " thread" << (threads == 1 ? "" : "s") << ")\n"
              << std::left << std::setw(12) << "position" << std::setw(7) << "depth" << std::right
              << std::setw(14) << "nodes" << std::setw(10) << "sec" << std::setw(11) << "Mn/s"
              << "  result\n";

    int failures            = 0;
    std::uint64_t all_nodes = 0;
    double all_seconds      = 0.0;

    for (const auto& c : suite()) {
        chess::Board board;
        if (!board.setFen(c.fen)) {
            std::cout << c.name << ": FAILED to parse FEN\n";
            ++failures;
            continue;
        }

        for (std::size_t d = 0; d < c.expected.size(); ++d) {
            const int depth = static_cast<int>(d) + 1;
            if (depth > max_depth) break;

            const auto t0    = std::chrono::steady_clock::now();
            const auto nodes = az::perft_threaded(board, depth, threads);
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            all_nodes += nodes;
            all_seconds += sec;

            const bool ok = nodes == c.expected[d];
            if (!ok) ++failures;

            std::cout << std::left << std::setw(12) << c.name << std::setw(7) << depth << std::right
                      << std::setw(14) << nodes << std::setw(10) << std::fixed << std::setprecision(3)
                      << sec << std::setw(11) << std::setprecision(2)
                      << (sec > 0 ? nodes / sec / 1e6 : 0.0) << "  "
                      << (ok ? "ok" : "MISMATCH, expected " + std::to_string(c.expected[d])) << "\n";
        }
    }

    std::cout << "\n" << all_nodes << " nodes in " << std::setprecision(2) << all_seconds << "s ("
              << (all_seconds > 0 ? all_nodes / all_seconds / 1e6 : 0.0) << " Mn/s)\n";

    if (failures) {
        std::cout << failures << " MISMATCH(ES) — the movegen is not trustworthy, fix before anything else\n";
        return 1;
    }
    std::cout << "perft suite exact\n";
    return 0;
}
