// Single-position search for the evaluation harness (docs/06-eval.md).
//
// The Runner (runner.hpp) plays self-play games in which the net moves both
// sides. Evaluation needs the other shape: search *one* position handed in
// from outside — a UCI `position ... go` from cutechess, or one game of a
// match whose other side is Stockfish — and hand back a move. That is the
// "direct search entry point" docs/06 asks for.
//
// It is a fleet rather than a single search because batching is what makes
// evaluation affordable: a 100-game match at 400 sims/move is a few million
// network evaluations, and one GPU forward per position would run for hours.
// N slots search their own positions and every pending leaf in the fleet rides
// in one batch, exactly as self-play does:
//
//   s = chess_engine.Searcher(n_slots=64, sims=400)
//   for i, game in enumerate(games): s.set_position(i, game)   # ce.Game objects
//   while (batch := s.pending()).size:
//       s.feed(*evaluator(batch))
//   moves = [s.best_move(i) for i in range(len(games))]
//
// Match play differs from self-play in exactly two settings, both defaulted
// here: no Dirichlet noise at the root (a match measures strength, not
// exploration) and move choice by argmax visits with no temperature. With
// noise off the whole thing is deterministic given the network's outputs,
// which is what makes a reported Elo reproducible.
//
// A slot copies the Game it is given, history included, so threefold and
// fifty-move adjudication inside the tree see the real game — the caller's
// Game object is never aliased and may be advanced or destroyed freely.
#pragma once

#include <cstdint>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <chess.hpp>

#include "encoding.hpp"
#include "game.hpp"
#include "mcts.hpp"

namespace az {

// pybind-free, like RunnerCore, so C++ callers and tests can drive it directly.
class SearcherCore {
   public:
    SearcherCore(int n_slots, const SearchParams& params, std::uint64_t seed) : params_(params) {
        if (n_slots < 1) throw std::invalid_argument("n_slots must be >= 1");
        if (params.sims < 2) throw std::invalid_argument("sims must be >= 2");
        if (params.dir_eps < 0.0f || params.dir_eps > 1.0f)
            throw std::invalid_argument("dirichlet_eps must be in [0, 1]");
        if (params.dir_eps > 0.0f && params.dir_alpha <= 0.0f)
            throw std::invalid_argument("dirichlet_alpha must be > 0 when dirichlet_eps > 0");

        // Sized once and never grown: each slot's Search holds a pointer into
        // its own Game's repetition window, which a reallocation would dangle.
        std::mt19937_64 seeder(seed);
        slots_.resize(static_cast<std::size_t>(n_slots));
        for (auto& s : slots_) s.search.init(params, seeder());
    }

    [[nodiscard]] int n_slots() const noexcept { return static_cast<int>(slots_.size()); }
    [[nodiscard]] const SearchParams& params() const noexcept { return params_; }

    // Point a slot at a position and begin its search. The game must not be
    // over — the caller adjudicates, this only searches.
    void set_position(int slot, const Game& game) {
        Slot& s = at(slot);
        if (game.outcome().over()) throw std::invalid_argument("set_position: the game is already over");
        s.game = game;  // copy; the search reads this slot's history, not the caller's
        s.search.start_move(s.game);
        s.active = s.started = true;
    }

    // Abandon a slot's search without finishing it (a game ended, a UCI `stop`).
    void clear(int slot) { at(slot).active = false; }

    // Advance every searching slot until it needs an evaluation or completes
    // its simulations, then rebuild the pending list — the batch order feed_all
    // must follow. An empty pending list means every slot has finished.
    void advance_all() {
        const int n = static_cast<int>(slots_.size());
        std::string error;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int i = 0; i < n; ++i) {
            try {
                advance_slot(slots_[static_cast<std::size_t>(i)]);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(mu_);
                if (error.empty()) error = e.what();
            }
        }
        if (!error.empty()) throw std::runtime_error(error);

        pending_slots_.clear();
        for (int i = 0; i < n; ++i) {
            if (slots_[static_cast<std::size_t>(i)].search.awaiting_eval()) pending_slots_.push_back(i);
        }
    }

    [[nodiscard]] std::size_t pending_count() const noexcept { return pending_slots_.size(); }
    [[nodiscard]] const chess::Board& pending_board(std::size_t batch_i) const {
        return slots_[static_cast<std::size_t>(pending_slots_[batch_i])].search.pending_board();
    }

    // policy: pending_count() x kPolicySize softmax probs, value: pending_count().
    void feed_all(const float* policy, const float* value, std::size_t n) {
        if (n != pending_slots_.size()) throw std::invalid_argument("feed size does not match the pending batch");
        for (std::size_t i = 0; i < n; ++i) {
            slots_[static_cast<std::size_t>(pending_slots_[i])].search.feed(
                policy + i * static_cast<std::size_t>(kPolicySize), value[i]);
        }
        pending_slots_.clear();
    }

    [[nodiscard]] bool active(int slot) const { return at(slot).active; }
    [[nodiscard]] int sims_done(int slot) const { return at(slot).search.sims_done(); }
    [[nodiscard]] float root_q(int slot) const { return searched(slot).search.root_q(); }
    [[nodiscard]] const StateNode& root(int slot) const { return searched(slot).search.root(); }
    [[nodiscard]] const Game& game(int slot) const { return searched(slot).game; }

    // Most-visited root move — the move a match plays. Ties break toward the
    // first edge, so a fed-back set of priors and values always yields the same
    // move. Reading an abandoned (cleared) search is fine and gives the best
    // move found so far, which is what a UCI `stop` wants.
    [[nodiscard]] int best_edge(int slot) const {
        const StateNode& r = searched(slot).search.root();
        if (r.n_edges <= 0) throw std::runtime_error("best_edge: slot has no searched root");
        int best = 0;
        for (int i = 1; i < r.n_edges; ++i) {
            if (r.edges[i].n_visits > r.edges[best].n_visits) best = i;
        }
        return best;
    }

   private:
    struct Slot {
        Game game;
        Search search;
        bool active  = false;
        bool started = false;  // has ever been given a position
    };

    [[nodiscard]] Slot& at(int slot) {
        if (slot < 0 || slot >= static_cast<int>(slots_.size())) throw std::out_of_range("slot out of range");
        return slots_[static_cast<std::size_t>(slot)];
    }
    [[nodiscard]] const Slot& at(int slot) const {
        if (slot < 0 || slot >= static_cast<int>(slots_.size())) throw std::out_of_range("slot out of range");
        return slots_[static_cast<std::size_t>(slot)];
    }

    // Search::root() dereferences a pointer that start_move() sets, so every
    // accessor that reaches into the tree has to come through here.
    [[nodiscard]] const Slot& searched(int slot) const {
        const Slot& s = at(slot);
        if (!s.started) throw std::runtime_error("slot has not been given a position yet; call set_position first");
        return s;
    }

    static void advance_slot(Slot& s) {
        if (!s.active || s.search.awaiting_eval()) return;
        if (!s.search.advance()) s.active = false;  // simulations complete
    }

    SearchParams params_;
    std::vector<Slot> slots_;
    std::vector<int> pending_slots_;
    std::mutex mu_;
};

}  // namespace az

// --- pybind11 bindings -------------------------------------------------------
// Compiled only inside the pybind module, same seam as encoding.hpp.
#ifdef PYBIND11_MODULE

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace az {

class Searcher {
   public:
    Searcher(int n_slots, int sims, float c_puct, float dirichlet_eps, float dirichlet_alpha, std::uint64_t seed)
        : core_(n_slots, SearchParams{sims, c_puct, dirichlet_alpha, dirichlet_eps}, seed) {}

    void set_position(int slot, const Game& game) { core_.set_position(slot, game); }
    void clear(int slot) { core_.clear(slot); }

    pybind11::array_t<std::uint8_t> pending() {
        {
            pybind11::gil_scoped_release release;
            core_.advance_all();
        }
        const auto n = static_cast<pybind11::ssize_t>(core_.pending_count());
        pybind11::array_t<std::uint8_t> out({n, static_cast<pybind11::ssize_t>(kNumPlanes),
                                             static_cast<pybind11::ssize_t>(8), static_cast<pybind11::ssize_t>(8)});
        std::uint8_t* data = out.mutable_data();
        for (pybind11::ssize_t i = 0; i < n; ++i) {
            encode_position(core_.pending_board(static_cast<std::size_t>(i)), data + i * kPlaneBytes);
        }
        return out;
    }

    void feed(pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> policy,
              pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> value) {
        const auto n = core_.pending_count();
        if (policy.ndim() != 2 || policy.shape(1) != kPolicySize ||
            static_cast<std::size_t>(policy.shape(0)) != n) {
            throw std::invalid_argument("policy must be (n_pending, " + std::to_string(kPolicySize) + ")");
        }
        if (value.ndim() != 1 || static_cast<std::size_t>(value.shape(0)) != n) {
            throw std::invalid_argument("value must be (n_pending,)");
        }
        const float* p = policy.data();
        const float* v = value.data();
        pybind11::gil_scoped_release release;
        core_.feed_all(p, v, n);
    }

    [[nodiscard]] std::string best_move(int slot) const {
        const chess::Board& board = core_.game(slot).board();
        return chess::uci::moveToUci(chess::Move(core_.root(slot).edges[core_.best_edge(slot)].move),
                                     board.chess960());
    }

    // The root's move statistics: what an `info` line reports, what a test
    // asserts on, and the raw material for any policy other than argmax.
    [[nodiscard]] pybind11::dict root(int slot) const {
        const StateNode& r        = core_.root(slot);
        const chess::Board& board = core_.game(slot).board();
        const auto n              = static_cast<pybind11::ssize_t>(r.n_edges > 0 ? r.n_edges : 0);

        pybind11::list moves;
        pybind11::array_t<std::int32_t> index(n), visits(n);
        pybind11::array_t<float> q(n), prior(n);
        for (pybind11::ssize_t i = 0; i < n; ++i) {
            const ActionEdge& e = r.edges[i];
            moves.append(chess::uci::moveToUci(chess::Move(e.move), board.chess960()));
            index.mutable_data()[i]  = e.move_index;
            visits.mutable_data()[i] = e.n_visits;
            q.mutable_data()[i]      = e.n_visits ? e.acc_value / static_cast<float>(e.n_visits) : 0.0f;
            prior.mutable_data()[i]  = e.prior;
        }
        pybind11::dict d;
        d["moves"]  = moves;
        d["index"]  = index;
        d["visits"] = visits;
        d["q"]      = q;
        d["prior"]  = prior;
        return d;
    }

    [[nodiscard]] float root_q(int slot) const { return core_.root_q(slot); }
    [[nodiscard]] bool active(int slot) const { return core_.active(slot); }
    [[nodiscard]] int sims_done(int slot) const { return core_.sims_done(slot); }
    [[nodiscard]] std::string fen(int slot) const { return core_.game(slot).fen(); }
    [[nodiscard]] int n_slots() const noexcept { return core_.n_slots(); }
    [[nodiscard]] int sims() const noexcept { return core_.params().sims; }

   private:
    SearcherCore core_;
};

inline void register_eval_bindings(pybind11::module_& m) {
    namespace py = pybind11;

    py::class_<Searcher>(
        m, "Searcher",
        "Batched single-position PUCT search for evaluation (docs/06). Point slots at Game objects with "
        "set_position, drive pending()/feed() until the batch comes back empty, then read best_move(slot). "
        "Defaults are match settings: no root noise, argmax-visits move choice, hence reproducible.")
        .def(py::init<int, int, float, float, float, std::uint64_t>(), py::arg("n_slots") = 1, py::arg("sims") = 400,
             py::arg("c_puct") = 1.5f, py::arg("dirichlet_eps") = 0.0f, py::arg("dirichlet_alpha") = 0.3f,
             py::arg("seed") = 0)
        .def("set_position", &Searcher::set_position, py::arg("slot"), py::arg("game"),
             "Copy a Game (history included, for in-tree repetition rules) into a slot and start its search. "
             "Raises ValueError if the game is already over.")
        .def("clear", &Searcher::clear, py::arg("slot"), "Abandon a slot's unfinished search.")
        .def("pending", &Searcher::pending,
             "Advance all searching slots (OpenMP, GIL released); return the (B,19,8,8) uint8 planes of the leaves "
             "awaiting evaluation. Empty <=> every slot has finished its simulations.")
        .def("feed", &Searcher::feed, py::arg("policy"), py::arg("value"),
             "policy: (n_pending, 4672) float32 softmax probs over the full head; value: (n_pending,) float32 in "
             "[-1, 1] from the side to move at the leaf. Same contract as Runner.feed.")
        .def("best_move", &Searcher::best_move, py::arg("slot"), "Most-visited root move, in UCI.")
        .def("root", &Searcher::root, py::arg("slot"),
             "Root move statistics: moves (UCI, aligned with the rest), index i32 into the policy head, visits i32, "
             "q f32 (mover's perspective, 0 if unvisited) and prior f32.")
        .def("root_q", &Searcher::root_q, py::arg("slot"),
             "Visit-weighted mean root value from the side to move — the score an `info` line reports.")
        .def("active", &Searcher::active, py::arg("slot"), "True while the slot still has simulations to run.")
        .def("sims_done", &Searcher::sims_done, py::arg("slot"))
        .def("fen", &Searcher::fen, py::arg("slot"), "FEN of the position the slot is searching.")
        .def_property_readonly("n_slots", &Searcher::n_slots)
        .def_property_readonly("sims", &Searcher::sims);
}

}  // namespace az

#endif  // PYBIND11_MODULE
