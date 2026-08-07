// Rules coverage, one case per line of the checklist in docs/01-engine.md.
//
// Perft already proves castling/en-passant/promotion *generation* correct at a
// scale no hand-written test can reach — 3.2 billion nodes deep. What perft
// says nothing about is the layer we wrote ourselves: draw adjudication, the
// repetition window, the ply cap, and result signs. That is what this covers,
// plus a few movegen cases spelled out by name so a regression reads as
// "en passant pin broke" instead of "perft position 3 depth 5 is off by 12".
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "game.hpp"
#include "perft.hpp"

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

std::vector<std::string> moves_of(const az::Game& g) {
    std::vector<std::string> out;
    for (const auto& m : g.legal_moves()) out.push_back(chess::uci::moveToUci(m));
    std::sort(out.begin(), out.end());
    return out;
}

bool has_move(const az::Game& g, const std::string& uci) {
    const auto all = moves_of(g);
    return std::find(all.begin(), all.end(), uci) != all.end();
}

void push_uci(az::Game& g, const std::string& uci) {
    for (const auto& m : g.legal_moves()) {
        if (chess::uci::moveToUci(m) == uci) {
            g.push(m);
            return;
        }
    }
    ++g_failures;
    std::cout << "  FAIL: no legal move " << uci << " in " << g.fen() << "\n";
}

void section(const char* name) { std::cout << name << "\n"; }

// --- castling ----------------------------------------------------------------
void test_castling() {
    section("castling");

    az::Game g("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    check(has_move(g, "e1g1"), "white can castle king-side");
    check(has_move(g, "e1c1"), "white can castle queen-side");

    // King move clears both rights; rook move clears only its own side.
    az::Game after_king("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    push_uci(after_king, "e1f1");
    check_eq(after_king.board().getCastleString(), std::string("kq"), "king move clears both white rights");

    az::Game after_rook("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    push_uci(after_rook, "a1b1");
    check_eq(after_rook.board().getCastleString(), std::string("Kkq"), "a1 rook move clears only Q");
    push_uci(after_rook, "a8b8");
    check(has_move(after_rook, "e1g1"), "a1 rook move keeps white king-side right");
    check(!has_move(after_rook, "e1c1"), "a1 rook move clears white queen-side right");

    // Capturing a rook on its home square kills the opponent's right on that
    // side. Captured by a knight, so the capture itself is not a check and the
    // castling test that follows is not vacuous.
    az::Game capture("r3k2r/8/6N1/8/8/8/8/R3K2R w KQkq - 0 1");
    push_uci(capture, "g6h8");
    check_eq(capture.board().getCastleString(), std::string("KQq"), "capturing h8 clears black's k right");
    check(!has_move(capture, "e8g8"), "h8 rook captured -> black loses king-side castling");
    check(has_move(capture, "e8c8"), "h8 rook captured -> black keeps queen-side castling");

    // Out of, through, and into check are all illegal.
    az::Game in_check("r3k2r/8/8/8/8/8/4r3/R3K2R w KQ - 0 1");
    check(!has_move(in_check, "e1g1") && !has_move(in_check, "e1c1"), "cannot castle out of check");

    az::Game through("r3k2r/8/8/8/8/8/5r2/R3K2R w KQ - 0 1");
    check(!has_move(through, "e1g1"), "cannot castle through an attacked square");
    check(has_move(through, "e1c1"), "queen-side castling unaffected by an f-file attack");

    // The b1 square may be attacked for queen-side castling; only e1/d1/c1 matter.
    az::Game b_file("r3k2r/8/8/8/8/8/1r6/R3K2R w KQ - 0 1");
    check(has_move(b_file, "e1c1"), "queen-side castling legal with b1 attacked");
}

// --- en passant --------------------------------------------------------------
void test_en_passant() {
    section("en passant");

    az::Game g("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
    check(has_move(g, "e5f6"), "en passant capture available");
    push_uci(g, "e5f6");
    check(g.fen().rfind("rnbqkbnr/ppp1p1pp/5P2/3p4/", 0) == 0, "en passant removes the captured pawn");

    // The perft-famous pin: white king a5, black rook h5, and the b5/c5 pawns
    // between them. Capturing en passant vacates *two* squares on the fifth rank
    // at once, which no ordinary pin test catches.
    az::Game pinned("8/8/3p4/KPp4r/1R3p1k/8/4P1P1/8 w - c6 0 2");
    check(has_move(pinned, "b5b6"), "the b-pawn can still push");
    check(!has_move(pinned, "b5c6"), "en passant illegal when it discovers a rank check");

    // The right expires if not taken immediately.
    az::Game expired("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
    push_uci(expired, "d2d3");
    push_uci(expired, "a7a6");
    check(!has_move(expired, "e5f6"), "en passant right expires after one ply");
}

// --- promotions --------------------------------------------------------------
void test_promotions() {
    section("promotions");

    az::Game g("8/4P3/8/8/8/8/8/K6k w - - 0 1");
    for (const std::string suffix : {"q", "r", "b", "n"}) {
        check(has_move(g, "e7e8" + suffix), "push promotion to " + suffix + " generated");
    }
    check_eq(moves_of(g).size(), std::size_t{7}, "4 promotions + 3 king moves");

    // Underpromotion on a capture: the move encoding has dedicated planes for
    // capture-left/capture-right underpromotions, so both must exist.
    az::Game cap("3r1r2/4P3/8/8/8/8/8/K6k w - - 0 1");
    for (const std::string suffix : {"q", "r", "b", "n"}) {
        check(has_move(cap, "e7d8" + suffix), "capture-left promotion to " + suffix);
        check(has_move(cap, "e7f8" + suffix), "capture-right promotion to " + suffix);
    }

    az::Game promoted("8/4P3/8/8/8/8/8/K6k w - - 0 1");
    push_uci(promoted, "e7e8n");
    check(promoted.fen().rfind("4N3/8/", 0) == 0, "underpromotion places a knight, not a queen");
}

// --- terminal detection ------------------------------------------------------
void test_checkmate_and_stalemate() {
    section("checkmate / stalemate");

    az::Game mate("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");  // fool's mate
    const auto m = mate.outcome();
    check(m.reason == az::Termination::CHECKMATE, "fool's mate detected");
    check_eq(m.value, -1, "the mated side to move scores -1");

    az::Game stale("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    const auto s = stale.outcome();
    check(s.reason == az::Termination::STALEMATE, "stalemate detected");
    check_eq(s.value, 0, "stalemate scores 0");

    az::Game live;
    check(!live.outcome().over(), "startpos is not terminal");
}

// --- fifty-move rule ---------------------------------------------------------
void test_fifty_move() {
    section("fifty-move rule");

    az::Game g("8/8/4k3/8/8/4K3/8/6R1 w - - 99 80");
    check(!g.outcome().over(), "99 halfmoves is not yet a draw");
    push_uci(g, "g1g2");
    check_eq(static_cast<int>(g.board().halfMoveClock()), 100, "halfmove clock reaches 100");
    check(g.outcome().reason == az::Termination::FIFTY_MOVE, "100 halfmoves is a draw");

    // A capture resets the clock rather than letting it tick to a draw.
    az::Game reset("8/8/4k3/8/8/4K1p1/8/6R1 w - - 99 80");
    push_uci(reset, "g1g3");
    check_eq(static_cast<int>(reset.board().halfMoveClock()), 0, "capture resets the halfmove clock");
    check(!reset.outcome().over(), "and so the game continues");

    // Mate outranks the clock: this is the ordering bug the rule ordering in
    // Game::outcome() exists to avoid.
    az::Game mate_on_100("7k/R7/6K1/8/8/8/8/8 w - - 99 80");
    push_uci(mate_on_100, "a7a8");
    check_eq(static_cast<int>(mate_on_100.board().halfMoveClock()), 100, "mating move is the 100th halfmove");
    check(mate_on_100.outcome().reason == az::Termination::CHECKMATE, "checkmate beats the fifty-move draw");
}

// --- threefold repetition ----------------------------------------------------
void test_threefold() {
    section("threefold repetition");

    az::Game g;
    check_eq(g.repetition_count(), 1, "startpos seen once");

    const std::vector<std::string> shuffle = {"g1f3", "g8f6", "f3g1", "f6g8"};
    for (const auto& uci : shuffle) push_uci(g, uci);
    check_eq(g.repetition_count(), 2, "startpos seen twice after one shuffle");
    check(!g.outcome().over(), "twofold is not a draw");

    for (const auto& uci : shuffle) push_uci(g, uci);
    check_eq(g.repetition_count(), 3, "startpos seen three times");
    check(g.outcome().reason == az::Termination::THREEFOLD, "threefold is a draw");

    // A pawn move is irreversible: it must clear the repetition window, so the
    // positions before it can never combine with positions after it.
    az::Game cleared;
    for (const auto& uci : shuffle) push_uci(cleared, uci);
    push_uci(cleared, "e2e4");
    check_eq(cleared.repetition_count(), 1, "pawn move clears the repetition window");

    // Same board, different castling rights: not a repetition. This is the case
    // a naive piece-placement hash gets wrong, and the reason we key on Zobrist.
    az::Game rights("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    for (const auto& uci : {"e1f1", "e8f8", "f1e1", "f8e8"}) push_uci(rights, uci);
    check_eq(rights.repetition_count(), 1, "castling rights lost -> not the same position");
}

// --- insufficient material ---------------------------------------------------
void test_insufficient_material() {
    section("insufficient material");

    const std::vector<std::pair<std::string, bool>> cases = {
        {"8/8/4k3/8/8/K7/8/8 w - - 0 1", true},           // K vs K
        {"8/8/4k3/8/8/K7/8/6N1 w - - 0 1", true},         // K+N vs K
        {"8/8/4k3/8/8/K7/8/6B1 w - - 0 1", true},         // K+B vs K
        {"8/8/4k3/8/5b2/K7/8/2B5 w - - 0 1", true},       // bishops both on dark squares (f4, c1)
        {"8/8/4k3/5b2/8/K7/8/2B5 w - - 0 1", false},      // opposite colours: mate is possible
        {"8/8/4k3/8/8/K7/6P1/8 w - - 0 1", false},        // a pawn can promote
        {"8/8/4k3/8/8/K7/8/5RR1 w - - 0 1", false},       // rooks mate
    };

    for (const auto& [fen, want_draw] : cases) {
        az::Game g(fen);
        const bool is_draw = g.outcome().reason == az::Termination::INSUFFICIENT_MATERIAL;
        check_eq(is_draw, want_draw, "insufficient material: " + fen);
    }
}

// --- self-play ply cap -------------------------------------------------------
void test_ply_cap() {
    section("ply cap");

    check_eq(az::kMaxPlies, 250, "the self-play cap is 250 plies");

    // Four plies of knight shuffling repeat the start position only twice, so
    // the cap — not threefold — is what ends this game.
    az::Game g(chess::constants::STARTPOS, /*max_plies=*/4);
    for (const auto& uci : {"g1f3", "g8f6", "f3g1", "f6g8"}) {
        check(!g.outcome().over(), "game live before the cap");
        push_uci(g, uci);
    }
    check_eq(g.ply(), 4, "four plies played");
    const auto o = g.outcome();
    check(o.reason == az::Termination::PLY_CAP, "the cap adjudicates a draw");
    check_eq(o.value, 0, "the cap scores 0");

    // The default cap must not fire on a short game.
    az::Game normal;
    for (const auto& uci : {"g1f3", "g8f6", "f3g1", "f6g8"}) push_uci(normal, uci);
    check(!normal.outcome().over(), "default cap does not fire at ply 4");

    // A mate on the capping ply is still a mate.
    az::Game mate("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2", /*max_plies=*/1);
    push_uci(mate, "d8h4");
    check(mate.outcome().reason == az::Termination::CHECKMATE, "checkmate beats the ply cap");
}

// --- game bookkeeping --------------------------------------------------------
void test_game_bookkeeping() {
    section("game bookkeeping");

    az::Game g;
    check_eq(g.ply(), 0, "fresh game is at ply 0");
    check_eq(g.fen(), std::string(chess::constants::STARTPOS), "fresh game is startpos");
    push_uci(g, "e2e4");
    check_eq(g.ply(), 1, "ply advances");

    // reset() must drop the repetition history, not just the board.
    az::Game reused;
    for (const auto& uci : {"g1f3", "g8f6", "f3g1", "f6g8"}) push_uci(reused, uci);
    reused.reset();
    check_eq(reused.ply(), 0, "reset rewinds the ply counter");
    check_eq(reused.repetition_count(), 1, "reset drops the repetition history");

    bool threw = false;
    try {
        az::Game bad("not a fen");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "an unparseable FEN throws");

    // Resuming mid-game: the FEN's own move counters must not feed the cap.
    az::Game resumed("8/8/4k3/8/8/4K3/8/6R1 w - - 42 90");
    check_eq(resumed.ply(), 0, "a resumed game starts its own ply count");
    check_eq(static_cast<int>(resumed.board().halfMoveClock()), 42, "but keeps the FEN's halfmove clock");
}

}  // namespace

int main() {
    test_castling();
    test_en_passant();
    test_promotions();
    test_checkmate_and_stalemate();
    test_fifty_move();
    test_threefold();
    test_insufficient_material();
    test_ply_cap();
    test_game_bookkeeping();

    std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " checks passed\n";
    if (g_failures) {
        std::cout << g_failures << " FAILURE(S)\n";
        return 1;
    }
    std::cout << "rules ok\n";
    return 0;
}
