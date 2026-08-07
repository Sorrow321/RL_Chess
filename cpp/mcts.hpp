// One game slot's PUCT search tree (docs/02-mcts-runner.md).
//
// Port of the 2048 az_engine tree to a two-player deterministic game: the
// chance nodes are gone, and every backed-up value flips sign at each ply
// (negamax). The convention that everything below hangs on: a value is always
// from the perspective of the side to move at the node where it was produced,
// and an edge's acc_value accumulates from the perspective of the player
// *making* that move. So select() reads Q = acc/n with no sign fixup, and
// backup() negates once per step upward.
//
// Positions are not stored in nodes. The slot keeps one Board that descends by
// makeMove and rewinds by unmakeMove during backup; while a leaf waits for its
// network evaluation the board is parked at that leaf (that is what pending()
// encodes). Trees are arena-allocated and reset per move, as in 2048.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <chess.hpp>

#include "encoding.hpp"
#include "game.hpp"

namespace az {

struct StateNode;

struct ActionEdge {           // one per legal move of the parent
    std::uint16_t move_index;  // 0..4671, the policy logit this edge reads
    std::uint16_t move;        // library-native move for makeMove/unmakeMove
    float prior;
    float acc_value;           // sum of backed-up values, mover's perspective
    std::int32_t n_visits;
    StateNode* child;          // nullptr until first descent
};

struct StateNode {
    ActionEdge* edges;         // arena array
    ActionEdge* parent_edge;
    std::int32_t n_visits;
    std::int16_t n_edges;      // -1 unexpanded, 0 terminal
    std::int8_t terminal_value;  // mover's perspective, set when n_edges == 0
};

class Arena {
   public:
    void init(std::size_t capacity) {
        buf_.reset(new std::uint8_t[capacity]);
        cap_ = capacity;
        used_ = 0;
    }
    void* alloc(std::size_t sz) {
        sz = (sz + 15) & ~std::size_t{15};
        if (used_ + sz > cap_) throw std::runtime_error("mcts: arena overflow (raise the branching allowance)");
        void* p = buf_.get() + used_;
        used_ += sz;
        return p;
    }
    void reset() noexcept { used_ = 0; }

   private:
    std::unique_ptr<std::uint8_t[]> buf_;
    std::size_t cap_ = 0, used_ = 0;
};

struct SearchParams {
    int sims           = 400;
    float c_puct       = 1.5f;
    float dir_alpha    = 0.3f;
    float dir_eps      = 0.25f;  // 0 disables root noise (evaluation play)
};

class Search {
   public:
    // Arena for one move's tree: per sim at most one StateNode plus one edge
    // array. Mean chess branching is ~35; 128 is the generous slack docs/02
    // asks for, and overflow aborts loudly rather than corrupting the tree.
    static std::size_t arena_bytes(int sims) {
        return static_cast<std::size_t>(sims + 2) * (32 + 128 * sizeof(ActionEdge)) + (std::size_t{1} << 14);
    }

    void init(const SearchParams& params, std::uint64_t noise_seed) {
        params_ = params;
        noise_rng_.seed(static_cast<std::mt19937::result_type>(noise_seed));
        arena_.init(arena_bytes(params.sims));
    }

    // Start a new move search from the game's current position. The game must
    // not be over (the game loop checks outcome() after every push).
    void start_move(const Game& game) {
        assert(!game.outcome().over());
        board_  = game.board();
        window_ = &game.repetition_window();
        arena_.reset();
        root_ = new_state(nullptr);
        pending_ = nullptr;
        path_.clear();
        path_hashes_.clear();
        sims_done_ = 0;
    }

    // Run simulations until a leaf needs a network evaluation (true; the board
    // is parked at that leaf for encoding) or all sims are done (false).
    bool advance() {
        assert(!pending_);
        while (sims_done_ < params_.sims) {
            StateNode* node = root_;
            while (node->n_edges > 0) {
                ActionEdge& e = node->edges[select_puct(*node)];
                board_.makeMove(chess::Move(e.move));
                path_.push_back(&e);
                path_hashes_.push_back(board_.hash());
                if (!e.child) e.child = new_state(&e);
                node = e.child;
            }
            if (node->n_edges == 0) {  // known terminal: back up, no net call
                backup(node, node->terminal_value);
                ++sims_done_;
                continue;
            }
            chess::Movelist moves;
            chess::movegen::legalmoves(moves, board_);
            std::int8_t tv;
            if (terminal_value(moves, &tv)) {
                node->n_edges = 0;
                node->terminal_value = tv;
                backup(node, tv);
                ++sims_done_;
                continue;
            }
            expand(node, moves);
            pending_ = node;
            return true;
        }
        return false;
    }

    [[nodiscard]] bool awaiting_eval() const noexcept { return pending_ != nullptr; }
    [[nodiscard]] const chess::Board& pending_board() const noexcept { return board_; }

    // Attach priors (masked to the legal move indices, renormalized; Dirichlet
    // mix at the root) and back the value up with sign flips, unmaking to root.
    // policy: kPolicySize softmax probs; value: mover's perspective at the leaf.
    void feed(const float* policy, float value) {
        if (!pending_) throw std::logic_error("mcts: feed() with no pending leaf");
        StateNode* node = pending_;
        pending_ = nullptr;

        const int n = node->n_edges;
        float mass = 0.0f;
        for (int i = 0; i < n; ++i) mass += policy[node->edges[i].move_index];
        if (mass > 1e-8f) {
            for (int i = 0; i < n; ++i) node->edges[i].prior = policy[node->edges[i].move_index] / mass;
        } else {
            for (int i = 0; i < n; ++i) node->edges[i].prior = 1.0f / static_cast<float>(n);
        }
        if (node == root_ && params_.dir_eps > 0.0f) {
            std::gamma_distribution<float> gamma(params_.dir_alpha, 1.0f);
            noise_.resize(n);
            float dsum = 0.0f;
            for (int i = 0; i < n; ++i) {
                noise_[i] = gamma(noise_rng_);
                dsum += noise_[i];
            }
            if (dsum > 1e-8f) {
                for (int i = 0; i < n; ++i) {
                    node->edges[i].prior =
                        (1.0f - params_.dir_eps) * node->edges[i].prior + params_.dir_eps * noise_[i] / dsum;
                }
            }
        }
        backup(node, value);
        ++sims_done_;
    }

    [[nodiscard]] bool done() const noexcept { return sims_done_ >= params_.sims; }
    [[nodiscard]] int sims_done() const noexcept { return sims_done_; }
    [[nodiscard]] const StateNode& root() const noexcept { return *root_; }

    // Visit-weighted mean Q over root edges, from the root mover's perspective —
    // the "root value" the resign rule reads. Well-defined once sims >= 2.
    [[nodiscard]] float root_q() const noexcept {
        float acc = 0.0f;
        std::int64_t n = 0;
        for (int i = 0; i < root_->n_edges; ++i) {
            acc += root_->edges[i].acc_value;
            n += root_->edges[i].n_visits;
        }
        return n ? acc / static_cast<float>(n) : 0.0f;
    }

   private:
    StateNode* new_state(ActionEdge* parent) {
        auto* sn = static_cast<StateNode*>(arena_.alloc(sizeof(StateNode)));
        sn->edges = nullptr;
        sn->parent_edge = parent;
        sn->n_visits = 0;
        sn->n_edges = -1;
        sn->terminal_value = 0;
        return sn;
    }

    [[nodiscard]] int select_puct(const StateNode& node) const noexcept {
        const float sqrt_n = std::sqrt(static_cast<float>(node.n_visits));
        float best = -1e30f;
        int best_i = 0;
        for (int i = 0; i < node.n_edges; ++i) {
            const ActionEdge& e = node.edges[i];
            const float q = e.n_visits ? e.acc_value / static_cast<float>(e.n_visits) : 0.0f;
            const float score = q + params_.c_puct * e.prior * sqrt_n / (1.0f + static_cast<float>(e.n_visits));
            if (score > best) {
                best = score;
                best_i = i;
            }
        }
        return best_i;
    }

    void expand(StateNode* node, const chess::Movelist& moves) {
        const int n = moves.size();
        auto* edges = static_cast<ActionEdge*>(arena_.alloc(static_cast<std::size_t>(n) * sizeof(ActionEdge)));
        for (int i = 0; i < n; ++i) {
            edges[i].move_index = static_cast<std::uint16_t>(move_to_index(board_, moves[i]));
            edges[i].move = moves[i].move();
            edges[i].prior = 0.0f;
            edges[i].acc_value = 0.0f;
            edges[i].n_visits = 0;
            edges[i].child = nullptr;
        }
        node->edges = edges;
        node->n_edges = static_cast<std::int16_t>(n);
    }

    // Terminal rules on the search board, mirroring Game::outcome()'s ordering
    // (mate wins every tie). Repetitions count across the game's played history
    // plus the current search path. The self-play ply cap is deliberately not
    // applied in-tree: it adjudicates games, not searches.
    bool terminal_value(const chess::Movelist& moves, std::int8_t* out) const {
        if (moves.empty()) {
            *out = board_.inCheck() ? std::int8_t{-1} : std::int8_t{0};
            return true;
        }
        if (board_.halfMoveClock() >= 100 || board_.isInsufficientMaterial() || repetition_count() >= 3) {
            *out = 0;
            return true;
        }
        return false;
    }

    // Occurrences of the current position, counting itself, across the game's
    // repetition window followed by the search path. Only positions an even
    // number of plies back and within the halfmove clock can repeat — the
    // clock bound is what makes an irreversible move anywhere on the path
    // fence off everything before it.
    [[nodiscard]] int repetition_count() const noexcept {
        const std::uint64_t key = board_.hash();
        const int hmc = static_cast<int>(board_.halfMoveClock());
        const int n_window = static_cast<int>(window_->size());
        const int len = n_window + static_cast<int>(path_hashes_.size());
        int count = 1;
        for (int d = 2; d <= hmc; d += 2) {
            const int i = len - 1 - d;
            if (i < 0) break;
            const std::uint64_t h = i < n_window ? (*window_)[i] : path_hashes_[i - n_window];
            if (h == key) ++count;
        }
        return count;
    }

    // v is from the perspective of the mover at `leaf`; flip at every ply
    // upward, and unmake the descent's moves so the board lands back on root.
    void backup(StateNode* leaf, float v) {
        ++leaf->n_visits;
        for (int i = static_cast<int>(path_.size()) - 1; i >= 0; --i) {
            ActionEdge* e = path_[i];
            v = -v;
            ++e->n_visits;
            e->acc_value += v;
            StateNode* parent = i ? path_[i - 1]->child : root_;
            ++parent->n_visits;
            board_.unmakeMove(chess::Move(e->move));
        }
        path_.clear();
        path_hashes_.clear();
    }

    SearchParams params_;
    Arena arena_;
    chess::Board board_;
    const std::vector<std::uint64_t>* window_ = nullptr;
    StateNode* root_ = nullptr;
    StateNode* pending_ = nullptr;
    std::vector<ActionEdge*> path_;
    std::vector<std::uint64_t> path_hashes_;
    std::vector<float> noise_;
    std::mt19937 noise_rng_;
    int sims_done_ = 0;
};

}  // namespace az
