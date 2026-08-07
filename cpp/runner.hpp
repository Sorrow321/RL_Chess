// Batched self-play runner (docs/02-mcts-runner.md) — the bridge.
//
// N concurrent games, each an arena-allocated PUCT tree (mcts.hpp); every game
// pauses when its search reaches a leaf that needs a network evaluation.
// Python collects all pending leaves as one (B,19,8,8) uint8 batch, runs one
// GPU forward for the whole fleet, and feeds priors+values back:
//
//   runner = chess_engine.Runner(total_games, n_parallel, sims, ...)
//   while (batch := runner.pending()).size:
//       p, v = net(gpu(batch))            # p: (B,4672) softmax, v: (B,) tanh
//       runner.feed(p, v)
//   examples, results = runner.get_examples(), runner.get_results()
//
// RunnerCore below is pybind-free so the C++ tests can drive full games
// (including from a custom start FEN, which the Python surface deliberately
// does not expose — self-play always starts from the initial position). The
// numpy/GIL layer at the bottom of this header joins the chess_engine module
// through the same __has_include seam as encoding.hpp.
#pragma once

#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <chess.hpp>

#include "game.hpp"
#include "mcts.hpp"

namespace az {

// Resign rule (docs/02): root value below the threshold for this many
// consecutive own moves resigns, and 1 game in kResignAuditDenom plays through
// with resignation disabled so the false-positive rate stays measurable.
inline constexpr int kResignConsecutive = 8;
inline constexpr int kResignAuditDenom  = 10;

// One training example, the raw-facts dump of docs/02. legal_idx and visits
// live in flat side arrays, aligned by the running sum of n_legal. result is
// from THIS mover's perspective, filled at game end.
struct MoveRecord {
    std::uint32_t game_id;
    std::uint16_t ply;
    std::uint16_t move;       // library-native; replay via Game.push_u16
    std::int8_t result;
    std::uint8_t stm_black;   // internal, for the result sign at game end
    std::uint16_t n_legal;
};

struct GameSummary {
    std::uint32_t game_id;
    std::int8_t result_white;         // +1 white wins / 0 draw / -1 black wins
    std::int32_t plies;
    std::uint8_t termination;         // az::Termination
    std::uint8_t resign_disabled;     // 1 = audit game, played through
    std::int8_t would_resign_side;    // first resign trigger: -1 none, 0 white, 1 black
    std::int32_t would_resign_ply;    // ply of that trigger, -1 none
};

namespace runner_detail {

// splitmix64, same generator the 2048 runner used for all slot-local choices.
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    std::uint32_t below(std::uint32_t n) {
        return static_cast<std::uint32_t>(((next() & 0xFFFFFFFFULL) * static_cast<std::uint64_t>(n)) >> 32);
    }
};

}  // namespace runner_detail

class RunnerCore {
   public:
    RunnerCore(std::int64_t total_games, int n_parallel, int sims, float c_puct, float dirichlet_alpha,
               float dirichlet_eps, int temp_plies, float resign_threshold, std::uint64_t seed,
               std::string_view start_fen = chess::constants::STARTPOS)
        : total_games_(total_games),
          temp_plies_(temp_plies),
          resign_threshold_(resign_threshold),
          start_fen_(start_fen) {
        if (total_games < 1) throw std::invalid_argument("total_games must be >= 1");
        if (n_parallel < 1) throw std::invalid_argument("n_parallel must be >= 1");
        if (sims < 2) throw std::invalid_argument("sims must be >= 2");
        if (temp_plies < 0) throw std::invalid_argument("temp_plies must be >= 0");
        if (dirichlet_eps < 0.0f || dirichlet_eps > 1.0f) throw std::invalid_argument("dirichlet_eps must be in [0, 1]");
        if (dirichlet_eps > 0.0f && dirichlet_alpha <= 0.0f)
            throw std::invalid_argument("dirichlet_alpha must be > 0 when dirichlet_eps > 0");
        if (resign_threshold < -1.0f || resign_threshold >= 0.0f)
            throw std::invalid_argument("resign_threshold must be in [-1, 0); -1 disables resignation");

        if (static_cast<std::int64_t>(n_parallel) > total_games) n_parallel = static_cast<int>(total_games);

        // A bad start FEN throws std::invalid_argument out of the first
        // start_new_game_locked below, before any search begins.
        const SearchParams params{sims, c_puct, dirichlet_alpha, dirichlet_eps};
        runner_detail::Rng seeder(seed ? seed : 0x8badf00d12345678ULL);
        slots_.resize(n_parallel);
        for (auto& s : slots_) {
            s.search.init(params, seeder.next());
            s.rng.s = seeder.next();
            start_new_game_locked(s);
        }
    }

    // Advance every slot until it needs an eval or runs out of games, then
    // rebuild the pending list — the batch order feed_all must follow. An
    // empty pending list means every requested game is finished.
    void advance_all() {
        const int n = static_cast<int>(slots_.size());
        std::string error;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int i = 0; i < n; ++i) {
            try {
                advance_slot(slots_[i]);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(mu_);
                if (error.empty()) error = e.what();
            }
        }
        if (!error.empty()) throw std::runtime_error(error);

        pending_slots_.clear();
        for (int i = 0; i < n; ++i) {
            if (slots_[i].search.awaiting_eval()) pending_slots_.push_back(i);
        }
    }

    [[nodiscard]] std::size_t pending_count() const noexcept { return pending_slots_.size(); }
    [[nodiscard]] const chess::Board& pending_board(std::size_t batch_i) const {
        return slots_[pending_slots_[batch_i]].search.pending_board();
    }

    // policy: pending_count() x kPolicySize softmax probs, value: pending_count().
    void feed_all(const float* policy, const float* value, std::size_t n) {
        if (n != pending_slots_.size()) throw std::invalid_argument("feed size does not match the pending batch");
        for (std::size_t i = 0; i < n; ++i) {
            slots_[pending_slots_[i]].search.feed(policy + i * static_cast<std::size_t>(kPolicySize), value[i]);
        }
        pending_slots_.clear();
    }

    // Everything accumulated since the last drain, cleared on return.
    void drain_examples(std::vector<MoveRecord>& recs, std::vector<std::uint16_t>& legal_idx,
                        std::vector<std::uint16_t>& visits) {
        std::lock_guard<std::mutex> lock(mu_);
        recs.swap(examples_);
        legal_idx.swap(ex_legal_idx_);
        visits.swap(ex_visits_);
        examples_.clear();
        ex_legal_idx_.clear();
        ex_visits_.clear();
    }

    std::vector<GameSummary> drain_results() {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<GameSummary> out;
        out.swap(results_);
        return out;
    }

    [[nodiscard]] std::int64_t games_completed() {
        std::lock_guard<std::mutex> lock(mu_);
        return games_completed_;
    }

   private:
    struct Slot {
        Game game;
        Search search;
        runner_detail::Rng rng{0};
        std::uint32_t game_id = 0;
        bool in_game = false;
        bool resign_disabled = false;
        int resign_streak[2] = {0, 0};  // [white, black]
        std::int8_t would_resign_side = -1;
        std::int32_t would_resign_ply = -1;
        // this game's examples, staged until the result is known
        std::vector<MoveRecord> recs;
        std::vector<std::uint16_t> rec_legal_idx;
        std::vector<std::uint16_t> rec_visits;
    };

    [[nodiscard]] bool resign_enabled() const noexcept { return resign_threshold_ > -1.0f; }

    void start_new_game_locked(Slot& s) {  // caller holds mu_ (or is the ctor)
        s.game_id = static_cast<std::uint32_t>(games_started_++);
        s.game.reset(start_fen_);
        s.in_game = true;
        s.resign_disabled = resign_enabled() && s.rng.below(kResignAuditDenom) == 0;
        s.resign_streak[0] = s.resign_streak[1] = 0;
        s.would_resign_side = -1;
        s.would_resign_ply = -1;
        s.recs.clear();
        s.rec_legal_idx.clear();
        s.rec_visits.clear();
        s.search.start_move(s.game);
    }

    void advance_slot(Slot& s) {
        while (s.in_game) {
            if (s.search.awaiting_eval()) return;  // fed next cycle
            if (s.search.advance()) return;        // paused at a leaf
            finish_move(s);                        // sims complete: play the move
        }
    }

    void finish_move(Slot& s) {
        const StateNode& root = s.search.root();
        const int n = root.n_edges;

        MoveRecord rec;
        rec.game_id = s.game_id;
        rec.ply = static_cast<std::uint16_t>(s.game.ply());
        rec.result = 0;
        rec.stm_black = s.game.side_to_move() == chess::Color::BLACK ? 1 : 0;
        rec.n_legal = static_cast<std::uint16_t>(n);

        std::int64_t total_visits = 0;
        for (int i = 0; i < n; ++i) {
            s.rec_legal_idx.push_back(root.edges[i].move_index);
            s.rec_visits.push_back(static_cast<std::uint16_t>(root.edges[i].n_visits));
            total_visits += root.edges[i].n_visits;
        }

        // Resign bookkeeping happens on the mover's turn, before the move is
        // chosen; the triggering search is still recorded as an example (it is
        // full-quality) even when the game ends here.
        bool resign_now = false;
        if (resign_enabled()) {
            const int side = rec.stm_black;
            if (s.search.root_q() < resign_threshold_) {
                if (++s.resign_streak[side] >= kResignConsecutive) {
                    if (s.would_resign_side < 0) {
                        s.would_resign_side = static_cast<std::int8_t>(side);
                        s.would_resign_ply = s.game.ply();
                    }
                    resign_now = !s.resign_disabled;
                }
            } else {
                s.resign_streak[side] = 0;
            }
        }

        // τ=1 visit-sampling for the first temp_plies plies, then argmax.
        int pick = 0;
        if (s.game.ply() < temp_plies_) {
            auto r = static_cast<std::int64_t>(s.rng.below(static_cast<std::uint32_t>(total_visits)));
            for (int i = 0; i < n; ++i) {
                r -= root.edges[i].n_visits;
                if (r < 0) {
                    pick = i;
                    break;
                }
            }
        } else {
            for (int i = 1; i < n; ++i) {
                if (root.edges[i].n_visits > root.edges[pick].n_visits) pick = i;
            }
        }
        rec.move = root.edges[pick].move;
        s.recs.push_back(rec);

        if (resign_now) {
            finish_game(s, Termination::RESIGN,
                        /*result_white=*/rec.stm_black ? std::int8_t{1} : std::int8_t{-1});
            return;
        }

        s.game.push(chess::Move(rec.move));
        const Outcome o = s.game.outcome();
        if (o.over()) {
            const bool white_to_move = s.game.side_to_move() == chess::Color::WHITE;
            finish_game(s, o.reason, static_cast<std::int8_t>(white_to_move ? o.value : -o.value));
            return;
        }
        s.search.start_move(s.game);
    }

    void finish_game(Slot& s, Termination reason, std::int8_t result_white) {
        for (auto& rec : s.recs) rec.result = rec.stm_black ? static_cast<std::int8_t>(-result_white) : result_white;

        std::lock_guard<std::mutex> lock(mu_);
        examples_.insert(examples_.end(), s.recs.begin(), s.recs.end());
        ex_legal_idx_.insert(ex_legal_idx_.end(), s.rec_legal_idx.begin(), s.rec_legal_idx.end());
        ex_visits_.insert(ex_visits_.end(), s.rec_visits.begin(), s.rec_visits.end());
        results_.push_back({s.game_id, result_white, s.game.ply(), static_cast<std::uint8_t>(reason),
                            static_cast<std::uint8_t>(s.resign_disabled), s.would_resign_side, s.would_resign_ply});
        ++games_completed_;
        if (games_started_ < total_games_) {
            start_new_game_locked(s);
        } else {
            s.in_game = false;
        }
    }

    std::int64_t total_games_;
    int temp_plies_;
    float resign_threshold_;
    std::string start_fen_;

    std::vector<Slot> slots_;
    std::vector<int> pending_slots_;

    std::mutex mu_;
    std::int64_t games_started_ = 0;
    std::int64_t games_completed_ = 0;
    std::vector<MoveRecord> examples_;
    std::vector<std::uint16_t> ex_legal_idx_;
    std::vector<std::uint16_t> ex_visits_;
    std::vector<GameSummary> results_;
};

}  // namespace az

// --- pybind11 bindings -------------------------------------------------------
// Compiled only inside the pybind module, same seam as encoding.hpp.
#ifdef PYBIND11_MODULE

#include <pybind11/numpy.h>

namespace az {

class Runner {
   public:
    Runner(std::int64_t total_games, int n_parallel, int sims, float c_puct, float dirichlet_alpha,
           float dirichlet_eps, int temp_plies, float resign_threshold, std::uint64_t seed)
        : core_(total_games, n_parallel, sims, c_puct, dirichlet_alpha, dirichlet_eps, temp_plies, resign_threshold,
                seed) {}

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

    pybind11::dict get_examples() {
        std::vector<MoveRecord> recs;
        std::vector<std::uint16_t> legal_idx, visits;
        core_.drain_examples(recs, legal_idx, visits);

        const auto n = static_cast<pybind11::ssize_t>(recs.size());
        pybind11::array_t<std::uint32_t> game_id(n);
        pybind11::array_t<std::uint16_t> ply(n), move(n), n_legal(n);
        pybind11::array_t<std::int8_t> result(n);
        for (pybind11::ssize_t i = 0; i < n; ++i) {
            game_id.mutable_data()[i] = recs[i].game_id;
            ply.mutable_data()[i] = recs[i].ply;
            move.mutable_data()[i] = recs[i].move;
            result.mutable_data()[i] = recs[i].result;
            n_legal.mutable_data()[i] = recs[i].n_legal;
        }
        pybind11::array_t<std::uint16_t> idx_arr(static_cast<pybind11::ssize_t>(legal_idx.size()));
        pybind11::array_t<std::uint16_t> vis_arr(static_cast<pybind11::ssize_t>(visits.size()));
        if (!legal_idx.empty())
            std::memcpy(idx_arr.mutable_data(), legal_idx.data(), legal_idx.size() * sizeof(std::uint16_t));
        if (!visits.empty())
            std::memcpy(vis_arr.mutable_data(), visits.data(), visits.size() * sizeof(std::uint16_t));

        pybind11::dict d;
        d["game_id"] = game_id;
        d["ply"] = ply;
        d["move"] = move;
        d["result"] = result;
        d["n_legal"] = n_legal;
        d["legal_idx"] = idx_arr;
        d["visits"] = vis_arr;
        return d;
    }

    pybind11::dict get_results() {
        const auto results = core_.drain_results();
        const auto n = static_cast<pybind11::ssize_t>(results.size());
        pybind11::array_t<std::uint32_t> game_id(n);
        pybind11::array_t<std::int8_t> result(n), would_resign_side(n);
        pybind11::array_t<std::int32_t> plies(n), would_resign_ply(n);
        pybind11::array_t<std::uint8_t> termination(n), resign_disabled(n);
        for (pybind11::ssize_t i = 0; i < n; ++i) {
            game_id.mutable_data()[i] = results[i].game_id;
            result.mutable_data()[i] = results[i].result_white;
            plies.mutable_data()[i] = results[i].plies;
            termination.mutable_data()[i] = results[i].termination;
            resign_disabled.mutable_data()[i] = results[i].resign_disabled;
            would_resign_side.mutable_data()[i] = results[i].would_resign_side;
            would_resign_ply.mutable_data()[i] = results[i].would_resign_ply;
        }
        pybind11::dict d;
        d["game_id"] = game_id;
        d["result"] = result;
        d["plies"] = plies;
        d["termination"] = termination;
        d["resign_disabled"] = resign_disabled;
        d["would_resign_side"] = would_resign_side;
        d["would_resign_ply"] = would_resign_ply;
        return d;
    }

    std::int64_t games_completed() { return core_.games_completed(); }

   private:
    RunnerCore core_;
};

inline void register_runner_bindings(pybind11::module_& m) {
    namespace py = pybind11;

    py::class_<Runner>(m, "Runner",
                       "Batched self-play MCTS runner (docs/02). pending() returns the (B,19,8,8) uint8 planes of "
                       "every leaf awaiting evaluation; feed(policy, value) resumes the trees. Empty batch <=> all "
                       "requested games finished.")
        .def(py::init<std::int64_t, int, int, float, float, float, int, float, std::uint64_t>(),
             py::arg("total_games"), py::arg("n_parallel"), py::arg("sims"), py::arg("c_puct") = 1.5f,
             py::arg("dirichlet_alpha") = 0.3f, py::arg("dirichlet_eps") = 0.25f, py::arg("temp_plies") = 15,
             py::arg("resign_threshold") = -0.95f, py::arg("seed") = 0)
        .def("pending", &Runner::pending,
             "Advance all games (OpenMP over slots, GIL released); return planes of leaves needing evaluation.")
        .def("feed", &Runner::feed, py::arg("policy"), py::arg("value"),
             "policy: (n_pending, 4672) float32 softmax probs over the full head (the runner masks to legal moves "
             "and renormalizes; Dirichlet noise is mixed at roots). value: (n_pending,) float32, side-to-move "
             "perspective in [-1, 1].")
        .def("get_examples", &Runner::get_examples,
             "Move examples of finished games, cleared on return. Flat dict of arrays: game_id u32, ply u16, "
             "move u16 (library-native; replay with Game.push_u16), result i8 (mover's perspective), n_legal u16, "
             "and legal_idx/visits u16 aligned by the running sum of n_legal.")
        .def("get_results", &Runner::get_results,
             "Per finished game, cleared on return: game_id u32, result i8 (+1 white/-1 black/0 draw), plies i32, "
             "termination u8 (Termination), resign_disabled u8 (audit game), would_resign_side i8 (-1/0=white/"
             "1=black) and would_resign_ply i32 for the resignation false-positive audit.")
        .def_property_readonly("games_completed", &Runner::games_completed);
}

}  // namespace az

#endif  // PYBIND11_MODULE
