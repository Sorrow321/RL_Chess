// The runner's negamax and draw-rule tests (docs/02-mcts-runner.md).
//
// The negamax convention is the classic bug farm: a single sign error makes
// the tree chase the opponent's mates instead of its own. So the search gets
// the doc-mandated mate tests with uniform priors (the tree must find forced
// mates on structure alone), a lost-position test (root value must be ~-1 for
// the side about to be mated), and the in-tree draw rules get feed-counting
// tests: with priors rigged to walk straight into a draw-by-rule, the search
// must adjudicate the leaf itself — the number of network evaluations it
// requests is exact, and any extra feed means a draw rule failed in-tree.
// Full games through RunnerCore then check the game loop end to end: a
// mate-in-2 must be played out in exactly 3 plies, and a dead-lost side must
// resign under the docs/02 rule.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "game.hpp"
#include "mcts.hpp"
#include "runner.hpp"

namespace {

int g_failures = 0;
int g_checks   = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::cout << "  FAIL: " << what << "\n";
    }
}

template <typename A, typename B>
void check_eq(const A& got, const B& want, const std::string& what) {
    ++g_checks;
    if (!(got == want)) {
        ++g_failures;
        std::cout << "  FAIL: " << what << " (got " << got << ", want " << want << ")\n";
    }
}

void section(const char* name) { std::cout << name << "\n"; }

chess::Move move_from_uci(const chess::Board& board, const std::string& uci) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    for (const auto& m : moves) {
        if (chess::uci::moveToUci(m) == uci) return m;
    }
    throw std::runtime_error("no legal move " + uci + " in " + board.getFen());
}

void push_uci(az::Game& g, const std::string& uci) { g.push(move_from_uci(g.board(), uci)); }

std::string uci_of(std::uint16_t move) { return chess::uci::moveToUci(chess::Move(move)); }

// A policy putting all mass on one move of `board` (feed renormalizes over the
// legal set, so this pins the priors: 1 for the move, 0 elsewhere).
std::vector<float> policy_on(const chess::Board& board, const std::string& uci) {
    std::vector<float> p(az::kPolicySize, 0.0f);
    p[az::move_to_index(board, move_from_uci(board, uci))] = 1.0f;
    return p;
}

// Drive a search to completion with uniform priors and a fixed value; returns
// the number of evaluations the search requested.
int drive_uniform(az::Search& s, float value = 0.0f) {
    const std::vector<float> uniform(az::kPolicySize, 1.0f / az::kPolicySize);
    int feeds = 0;
    while (s.advance()) {
        s.feed(uniform.data(), value);
        ++feeds;
    }
    return feeds;
}

int most_visited(const az::Search& s) {
    const az::StateNode& root = s.root();
    int best = 0;
    for (int i = 1; i < root.n_edges; ++i) {
        if (root.edges[i].n_visits > root.edges[best].n_visits) best = i;
    }
    return best;
}

const az::ActionEdge* find_edge(const az::Search& s, const std::string& uci) {
    const az::StateNode& root = s.root();
    for (int i = 0; i < root.n_edges; ++i) {
        if (uci_of(root.edges[i].move) == uci) return &root.edges[i];
    }
    return nullptr;
}

// --- forced mates with uniform priors (the doc-mandated tests) ---------------

void test_mate_in_one() {
    section("mate in 1, uniform priors");

    az::Game g("1k6/8/1K6/8/8/8/8/7R w - - 0 1");  // Rh8#
    az::Search s;
    s.init({/*sims=*/800, /*c_puct=*/1.5f, /*dir_alpha=*/0.3f, /*dir_eps=*/0.0f}, /*noise_seed=*/1);
    s.start_move(g);
    drive_uniform(s);

    check_eq(uci_of(s.root().edges[most_visited(s)].move), std::string("h1h8"), "the mating move gets most visits");
    const az::ActionEdge* mate = find_edge(s, "h1h8");
    check(mate && mate->n_visits > 0, "the mating edge was visited");
    if (mate && mate->n_visits > 0) {
        check(mate->acc_value / static_cast<float>(mate->n_visits) > 0.99f, "the mating edge's Q is ~+1");
    }
    check(s.root_q() > 0.8f, "root value approaches +1 with a mate in hand (got " + std::to_string(s.root_q()) + ")");
}

void test_mate_in_two() {
    section("mate in 2, uniform priors");

    // 1.Ra7 (or 1.Rg7) cuts the 7th rank, 2.R(other)8#. No mate in 1 exists:
    // either rook checking on the 8th rank lets the king out via the 7th.
    az::Game g("3k4/8/R7/8/8/8/8/6RK w - - 0 1");
    az::Search s;
    s.init({/*sims=*/5000, 1.5f, 0.3f, 0.0f}, 1);
    s.start_move(g);
    drive_uniform(s);

    const std::string best = uci_of(s.root().edges[most_visited(s)].move);
    check(best == "a6a7" || best == "g1g7", "a rank-7 cut gets most visits (got " + best + ")");

    const az::ActionEdge* a7 = find_edge(s, "a6a7");
    const az::ActionEdge* g7 = find_edge(s, "g1g7");
    std::int64_t mate_visits = (a7 ? a7->n_visits : 0) + (g7 ? g7->n_visits : 0);
    check(mate_visits > s.sims_done() / 2, "the two mating moves dominate the visit budget");
    check(s.root_q() > 0.6f, "root value is strongly positive (got " + std::to_string(s.root_q()) + ")");
}

void test_lost_position_value() {
    section("lost position (negamax sign)");

    // Black to move after 1.Rg7: both king moves are met by 2.Ra8#. A backup
    // sign error turns this into a position black likes.
    az::Game g("3k4/6R1/R7/8/8/8/8/7K b - - 0 1");
    az::Search s;
    s.init({/*sims=*/1500, 1.5f, 0.3f, 0.0f}, 1);
    s.start_move(g);
    drive_uniform(s);

    check_eq(static_cast<int>(s.root().n_edges), 2, "black has exactly two king moves");
    check(s.root_q() < -0.8f, "root value approaches -1 for the side being mated (got " +
                                  std::to_string(s.root_q()) + ")");
    for (int i = 0; i < s.root().n_edges; ++i) {
        const az::ActionEdge& e = s.root().edges[i];
        if (e.n_visits == 0) continue;
        check(e.acc_value / static_cast<float>(e.n_visits) < -0.5f,
              "every reply's Q is clearly losing (" + uci_of(e.move) + ")");
    }
}

// --- in-tree draw adjudication (feed counting) -------------------------------

void test_fifty_move_in_tree() {
    section("fifty-move rule in-tree");

    // hmc=99, no capture or pawn move available: every depth-1 node is a
    // 100-halfmove draw, so the only network evaluation is the root's.
    az::Game g("8/8/4k3/8/8/4K3/8/6R1 w - - 99 80");
    az::Search s;
    s.init({/*sims=*/64, 1.5f, 0.3f, 0.0f}, 1);
    s.start_move(g);
    const int feeds = drive_uniform(s, /*value=*/0.7f);

    check_eq(feeds, 1, "only the root needed an evaluation");
    check(s.done(), "all sims completed");
    check(std::fabs(s.root_q()) < 1e-6f, "every line is adjudicated a draw");
    for (int i = 0; i < s.root().n_edges; ++i) {
        check(s.root().edges[i].acc_value == 0.0f, "draw terminals back up exactly 0 (" +
                                                       uci_of(s.root().edges[i].move) + ")");
    }
}

void test_threefold_in_tree() {
    section("threefold repetition in-tree");

    // Knight shuffle: the current position is its second occurrence, and two
    // more shuffle plies inside the tree reach the start position's third.
    // The third occurrence spans game history AND search path — a search that
    // ignores the game's hash history keeps asking for evaluations here.
    az::Game g;
    for (const auto& uci : {"g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"}) push_uci(g, uci);

    const std::vector<float> white_p = policy_on(g.board(), "f3g1");
    chess::Board after = g.board();
    after.makeMove(move_from_uci(after, "f3g1"));
    const std::vector<float> black_p = policy_on(after, "f6g8");

    const int sims = 32;
    az::Search s;
    s.init({sims, 1.5f, 0.3f, 0.0f}, 1);
    s.start_move(g);
    int feeds = 0;
    while (s.advance()) {
        const bool white = s.pending_board().sideToMove() == chess::Color::WHITE;
        s.feed(white ? white_p.data() : black_p.data(), 0.0f);
        ++feeds;
    }

    check_eq(feeds, 2, "the repetition leaf never reaches the network");
    const az::ActionEdge* e = find_edge(s, "f3g1");
    check(e != nullptr, "the shuffle edge exists");
    if (e) {
        check_eq(e->n_visits, sims - 1, "every sim after the root walks the shuffle line");
        check(e->acc_value == 0.0f, "the repetition draw backs up exactly 0");
    }
    check(std::fabs(s.root_q()) < 1e-6f, "root value is the draw");
}

void test_insufficient_material_in_tree() {
    section("insufficient material in-tree");

    // White is in check; capturing the undefended rook leaves K vs K, which
    // must be adjudicated 0 in-tree, not sent to the network.
    az::Game g("8/8/4k3/8/8/3rK3/8/8 w - - 0 1");
    const std::vector<float> p = policy_on(g.board(), "e3d3");

    const int sims = 16;
    az::Search s;
    s.init({sims, 1.5f, 0.3f, 0.0f}, 1);
    s.start_move(g);
    int feeds = 0;
    while (s.advance()) {
        s.feed(p.data(), 0.0f);
        ++feeds;
    }

    check_eq(feeds, 1, "only the root needed an evaluation");
    const az::ActionEdge* e = find_edge(s, "e3d3");
    check(e != nullptr, "the capture edge exists");
    if (e) {
        check_eq(e->n_visits, sims - 1, "every sim after the root takes the capture");
        check(e->acc_value == 0.0f, "K vs K backs up exactly 0");
    }
}

// --- full games through the runner core --------------------------------------

// Feed every pending leaf uniform priors and value_fn(board); returns cycles.
template <typename ValueFn>
int drive_core(az::RunnerCore& core, ValueFn value_fn, int max_cycles) {
    std::vector<float> p, v;
    int cycles = 0;
    while (true) {
        core.advance_all();
        const std::size_t n = core.pending_count();
        if (n == 0) return cycles;
        if (++cycles > max_cycles) throw std::runtime_error("runner did not finish within the cycle budget");
        p.assign(n * static_cast<std::size_t>(az::kPolicySize), 1.0f / az::kPolicySize);
        v.resize(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = value_fn(core.pending_board(i));
        core.feed_all(p.data(), v.data(), n);
    }
}

struct DrainedExamples {
    std::vector<az::MoveRecord> recs;
    std::vector<std::uint16_t> legal_idx;
    std::vector<std::uint16_t> visits;
};

DrainedExamples drain(az::RunnerCore& core) {
    DrainedExamples d;
    core.drain_examples(d.recs, d.legal_idx, d.visits);
    return d;
}

void test_runner_plays_out_mate() {
    section("runner plays out a mate in 2");

    const std::string fen = "3k4/8/R7/8/8/8/8/6RK w - - 0 1";
    const int sims = 4000;
    az::RunnerCore runner(/*total_games=*/2, /*n_parallel=*/2, sims, /*c_puct=*/1.5f,
                          /*dirichlet_alpha=*/0.3f, /*dirichlet_eps=*/0.25f, /*temp_plies=*/0,
                          /*resign_threshold=*/-1.0f, /*seed=*/1, fen);
    drive_core(runner, [](const chess::Board&) { return 0.0f; }, /*max_cycles=*/200000);
    check_eq(runner.games_completed(), std::int64_t{2}, "both games completed");

    const auto results = runner.drain_results();
    check_eq(results.size(), std::size_t{2}, "two game summaries");
    for (const auto& r : results) {
        check_eq(static_cast<int>(r.termination), static_cast<int>(az::Termination::CHECKMATE),
                 "the game ends in checkmate");
        check_eq(static_cast<int>(r.result_white), 1, "white wins");
        check_eq(r.plies, 3, "the mate in 2 takes exactly 3 plies");
    }

    const auto ex = drain(runner);
    check_eq(ex.recs.size(), std::size_t{6}, "three examples per game");
    std::size_t off = 0;
    std::map<std::uint32_t, az::Game> replay;
    replay.emplace(0u, az::Game(fen));
    replay.emplace(1u, az::Game(fen));
    for (const auto& rec : ex.recs) {
        // mover-perspective results: white recorded +1, black -1
        check_eq(static_cast<int>(rec.result), rec.stm_black ? -1 : 1, "example result is mover-relative");
        std::int64_t total = 0;
        for (std::uint16_t i = 0; i < rec.n_legal; ++i) total += ex.visits[off + i];
        check_eq(total, static_cast<std::int64_t>(sims - 1), "root visits sum to sims-1");
        off += rec.n_legal;

        az::Game& g = replay.at(rec.game_id);
        check_eq(static_cast<int>(rec.ply), g.ply(), "examples arrive in ply order");
        g.push(chess::Move(rec.move));  // throws in debug if illegal; outcome checks below
    }
    check_eq(off, ex.legal_idx.size(), "n_legal sums to the flat array length");
    for (auto& [id, g] : replay) {
        check(g.outcome().reason == az::Termination::CHECKMATE,
              "replaying game " + std::to_string(id) + "'s dump reaches the checkmate");
    }
}

void test_resign_rule() {
    section("resign rule");

    // KQ+K vs K, truthful values by material: the bare-king side sees -1 at
    // every root, hits 8 consecutive triggers on its 8th move (ply 15) and
    // resigns; the queen side never triggers.
    const std::string fen = "k7/8/8/8/8/8/8/1QK5 w - - 0 1";
    const auto material_value = [](const chess::Board& b) {
        return b.pieces(chess::PieceType::QUEEN, b.sideToMove()).empty() ? -1.0f : 1.0f;
    };

    az::RunnerCore runner(/*total_games=*/1, /*n_parallel=*/1, /*sims=*/200, 1.5f, 0.3f, 0.25f,
                          /*temp_plies=*/0, /*resign_threshold=*/-0.95f, /*seed=*/5, fen);
    drive_core(runner, material_value, /*max_cycles=*/500000);

    const auto results = runner.drain_results();
    check_eq(results.size(), std::size_t{1}, "one game summary");
    std::int32_t plies = 0;
    if (results.size() == 1) {
        const auto& r = results[0];
        plies = r.plies;
        check_eq(static_cast<int>(r.resign_disabled), 0, "this seed is not an audit game");
        check_eq(static_cast<int>(r.termination), static_cast<int>(az::Termination::RESIGN), "black resigns");
        check_eq(static_cast<int>(r.result_white), 1, "resignation scores as a loss for the resigner");
        // 8 consecutive triggers need at least black's 8th move (ply 15); a
        // streak reset (a drawish search, e.g. a stalemate-flavored subtree)
        // only pushes the resignation later, always onto a black ply.
        check(r.plies >= 15 && r.plies % 2 == 1,
              "resignation lands on a black move at ply >= 15 (got " + std::to_string(r.plies) + ")");
        check_eq(static_cast<int>(r.would_resign_side), 1, "the trigger side is black");
        check_eq(r.would_resign_ply, r.plies, "the trigger ply is the resignation ply");
    }

    const auto ex = drain(runner);
    check_eq(ex.recs.size(), static_cast<std::size_t>(plies) + 1,
             "the triggering search is recorded on top of the played plies");
    for (const auto& rec : ex.recs) {
        check_eq(static_cast<int>(rec.result), rec.stm_black ? -1 : 1, "results are mover-relative");
    }
}

void test_parameter_validation() {
    section("parameter validation");

    auto throws = [](auto fn) {
        try {
            fn();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    check(throws([] { az::RunnerCore r(0, 1, 100, 1.5f, 0.3f, 0.25f, 15, -0.95f, 1); }), "total_games >= 1");
    check(throws([] { az::RunnerCore r(1, 1, 1, 1.5f, 0.3f, 0.25f, 15, -0.95f, 1); }), "sims >= 2");
    check(throws([] { az::RunnerCore r(1, 1, 100, 1.5f, 0.3f, 0.25f, 15, 0.5f, 1); }), "resign_threshold < 0");
    check(throws([] { az::RunnerCore r(1, 1, 100, 1.5f, 0.3f, 0.25f, 15, -0.95f, 1, "not a fen"); }),
          "bad start FEN throws");
}

}  // namespace

int main() {
    test_mate_in_one();
    test_mate_in_two();
    test_lost_position_value();
    test_fifty_move_in_tree();
    test_threefold_in_tree();
    test_insufficient_material_in_tree();
    test_runner_plays_out_mate();
    test_resign_rule();
    test_parameter_validation();

    std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " checks passed\n";
    if (g_failures) {
        std::cout << g_failures << " FAILURE(S)\n";
        return 1;
    }
    std::cout << "runner ok\n";
    return 0;
}
