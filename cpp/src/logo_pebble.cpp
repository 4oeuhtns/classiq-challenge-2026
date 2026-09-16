#include <cstdio>
#include <ctime>
#include <fstream>
#include <lorina/pla.hpp>
#include <mockturtle/networks/xag.hpp>
#include <mockturtle/io/pla_reader.hpp>
#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/cut_rewriting.hpp>
#include <mockturtle/algorithms/node_resynthesis/xag_npn.hpp>
#include <mockturtle/algorithms/xag_resub_withDC.hpp>
#include <caterpillar/caterpillar.hpp>
#include <tweedledum/networks/netlist.hpp>

using namespace caterpillar;
using namespace mockturtle;
using namespace tweedledum;

// Every attempt gets appended here immediately, so killing the process at any
// point (Ctrl-C) never loses anything -- the file always ends with the most
// recent thing we know, whether that's "still climbing" or "found it."
static const char* LOG_PATH =
    "/Users/aoeuhtns/Documents/quantum/classiq-challenge-2026/cpp/pebble_progress.log";

static void log_line(const std::string& line)
{
    std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%H:%M:%S", std::localtime(&now));

    std::ofstream out(LOG_PATH, std::ios::app);
    out << "[" << stamp << "] " << line << "\n";
    out.close();  // closed (not just flushed) so the write actually lands on disk

    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
}

int main()
{
    mockturtle::xag_network xag;
    auto const result = lorina::read_pla(
        "/Users/aoeuhtns/Documents/quantum/classiq-challenge-2026/cpp/logo.pla",
        mockturtle::pla_reader(xag));
    if (result != lorina::return_code::success || xag.num_pis() == 0) {
        std::printf("PLA read failed\n");
        return 1;
    }

    xag_npn_resynthesis<xag_network> resyn;
    cut_rewriting_params crp;
    crp.cut_enumeration_ps.cut_size = 4;
    xag = cut_rewriting(xag, resyn, crp);
    xag = cleanup_dangling(xag);

    resubstitution_params rp;
    resubstitution_minmc_withDC(xag, rp);
    xag = cleanup_dangling(xag);

    log_line("network ready: gates=" + std::to_string(xag.num_gates()));

    // Climb pebble_limit ourselves (no library progress bar -- it floods the
    // terminal once this loops many times) so every attempt is logged on its
    // own clean line, and the log file always reflects the best thing found
    // so far, no matter when the process gets killed.
    for (uint32_t limit = 6; limit <= 60; ++limit) {
        log_line("trying pebble_limit = " + std::to_string(limit) + " ...");

        netlist<stg_gate> circ;
        pebbling_mapping_strategy_params ps;
        ps.pebble_limit = limit;
        ps.progress = false;
        ps.search_timeout = 600;  // generous -- this is meant to run unattended

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
