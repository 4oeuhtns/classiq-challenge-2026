#include <cstdio>
#include <ctime>
#include <fstream>
#include <vector>
#include <lorina/pla.hpp>
#include <mockturtle/networks/xag.hpp>
#include <mockturtle/io/pla_reader.hpp>
#include <mockturtle/algorithms/simulation.hpp>
#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/cut_rewriting.hpp>
#include <mockturtle/algorithms/node_resynthesis/xag_npn.hpp>
#include <caterpillar/caterpillar.hpp>
#include <tweedledum/networks/netlist.hpp>
#include "pieces_data.hpp"

using namespace mockturtle;
using namespace caterpillar;
using namespace tweedledum;

static const char* LOG_PATH =
    "/Users/aoeuhtns/Documents/quantum/classiq-challenge-2026/cpp/pebble_handbuilt_progress.log";

static void log_line(const std::string& line)
{
    std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%H:%M:%S", std::localtime(&now));
    std::ofstream out(LOG_PATH, std::ios::app);
    out << "[" << stamp << "] " << line << "\n";
    out.close();
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
}

// AND together (bit == val) for every requirement in one dyadic block.
xag_network::signal build_block(xag_network& xag, const std::vector<xag_network::signal>& bits,
                                 const std::vector<BitReq>& reqs)
{
    xag_network::signal acc = xag.get_constant(true);
    for (auto& r : reqs) {
        xag_network::signal lit = r.val ? bits[r.bit] : xag.create_not(bits[r.bit]);
        acc = xag.create_and(acc, lit);
    }
    return acc;
}

// XOR the blocks together (they're disjoint dyadic ranges, so XOR == OR here) --
// this is exactly the "OR of blocks" range predicate from best_blocks/blk.
xag_network::signal build_range_flag(xag_network& xag, const std::vector<xag_network::signal>& bits,
                                      const std::vector<std::vector<BitReq>>& blocks)
{
    xag_network::signal acc = xag.get_constant(false);
    for (auto& block : blocks) {
        acc = xag.create_xor(acc, build_block(xag, bits, block));
    }
    return acc;
}

int main()
{
    // --- build the network from the SAME 10-piece structure as handbuilt.py ---
    xag_network xag;
    std::vector<xag_network::signal> xbits, ybits;
    for (int i = 0; i < 6; i++) xbits.push_back(xag.create_pi());
    for (int i = 0; i < 6; i++) ybits.push_back(xag.create_pi());

    xag_network::signal total = xag.get_constant(false);
    for (auto& piece : PIECES) {
        auto yflag = build_range_flag(xag, ybits, piece.y_blocks);
        auto xflag = build_range_flag(xag, xbits, piece.x_blocks);
        auto term = xag.create_and(yflag, xflag);
        total = xag.create_xor(total, term);
    }
    xag.create_po(total);

    log_line("hand-built network: gates=" + std::to_string(xag.num_gates()));

    // --- verify it actually computes the same function as the real logo ---
    xag_network ref;
    auto res = lorina::read_pla(
        "/Users/aoeuhtns/Documents/quantum/classiq-challenge-2026/cpp/logo.pla",
        mockturtle::pla_reader(ref));
    if (res != lorina::return_code::success || ref.num_pis() == 0) {
        log_line("reference PLA read failed -- aborting");
        return 1;
    }

    bool mismatch = false;
    for (uint32_t v = 0; v < 4096 && !mismatch; v++) {
        std::vector<bool> assign(12);
        for (int i = 0; i < 12; i++) assign[i] = (v >> i) & 1;
        default_simulator<bool> sim(assign);
        bool a = simulate<bool>(xag, sim)[0];
        bool b = simulate<bool>(ref, sim)[0];
        if (a != b) {
            log_line("MISMATCH at input " + std::to_string(v));
            mismatch = true;
        }
    }
    if (mismatch) {
        log_line("hand-built network does NOT match target -- aborting before wasting time pebbling garbage");
        return 1;
    }
    log_line("verified: matches target on all 4096 inputs");

    // --- light cleanup, then try to pebble it ---
    xag_npn_resynthesis<xag_network> resyn;
    cut_rewriting_params crp;
    crp.cut_enumeration_ps.cut_size = 4;
    xag = cut_rewriting(xag, resyn, crp);
    xag = cleanup_dangling(xag);
    log_line("after cut_rewriting: gates=" + std::to_string(xag.num_gates()));

    for (uint32_t limit = 6; limit <= 60; ++limit) {
        log_line("trying pebble_limit = " + std::to_string(limit) + " ...");

        netlist<stg_gate> circ;
        pebbling_mapping_strategy_params ps;
        ps.pebble_limit = limit;
        ps.progress = false;
        ps.search_timeout = 5;

        pebbling_mapping_strategy<xag_network, bsat_pebble_solver<xag_network>> strategy(ps);
        logic_network_synthesis_stats st;
        logic_network_synthesis(circ, xag, strategy, {}, {}, &st);

        if (circ.num_gates() > 0) {
            log_line("SUCCESS at pebble_limit=" + std::to_string(limit) +
                      "  qubits=" + std::to_string(circ.num_qubits()) +
                      "  gates=" + std::to_string(circ.num_gates()));
            return 0;
        }
        log_line("  failed at pebble_limit=" + std::to_string(limit));
    }

    log_line("gave up: nothing worked up to pebble_limit=60");
    return 1;
}
