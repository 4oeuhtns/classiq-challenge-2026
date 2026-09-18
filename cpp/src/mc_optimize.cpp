// Iterate cut_rewriting + MC-targeted resubstitution to convergence, and report
// the AND/XOR split -- the quantity that actually matters, since in a reversible
// phase oracle every AND costs a Toffoli (3 CX, ~6 layers) while every XOR is a
// bare CX (1 CX, 1 layer).
#include <cstdio>
#include <lorina/pla.hpp>
#include <mockturtle/networks/xag.hpp>
#include <mockturtle/io/pla_reader.hpp>
#include <mockturtle/views/depth_view.hpp>
#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/cut_rewriting.hpp>
#include <mockturtle/algorithms/refactoring.hpp>
#include <mockturtle/algorithms/node_resynthesis/xag_npn.hpp>
#include <mockturtle/algorithms/xag_resub_withDC.hpp>

static unsigned ands(mockturtle::xag_network& x) {
    unsigned a = 0; x.foreach_gate([&](auto n){ if (x.is_and(n)) a++; }); return a;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s file.pla [rounds]\n", argv[0]); return 1; }
    int rounds = argc > 2 ? std::atoi(argv[2]) : 40;

    mockturtle::xag_network xag;
    if (lorina::read_pla(argv[1], mockturtle::pla_reader(xag)) != lorina::return_code::success) {
        std::printf("PLA read failed\n"); return 1;
    }
    std::printf("raw        gates %6u  AND %6u  XOR %6u\n",
                xag.num_gates(), ands(xag), xag.num_gates()-ands(xag));

    mockturtle::xag_npn_resynthesis<mockturtle::xag_network> resyn;
    unsigned best = ands(xag); int stale = 0;
    for (int i = 0; i < rounds && stale < 4; i++) {
        mockturtle::cut_rewriting_params crp;
        crp.cut_enumeration_ps.cut_size = 4;
        xag = mockturtle::cleanup_dangling(mockturtle::cut_rewriting(xag, resyn, crp));
        mockturtle::resubstitution_params rp;
        rp.max_pis = 8; rp.max_inserts = 3;
        mockturtle::resubstitution_minmc_withDC(xag, rp);
        xag = mockturtle::cleanup_dangling(xag);
        unsigned a = ands(xag);
        if (a < best) { best = a; stale = 0; } else stale++;
        if (i % 5 == 0 || stale >= 4)
            std::printf("  round %2d  gates %6u  AND %6u  XOR %6u\n",
                        i, xag.num_gates(), a, xag.num_gates()-a);
    }
    mockturtle::depth_view dv{xag};
    unsigned a = ands(xag);
    std::printf("FINAL      gates %6u  AND %6u  XOR %6u  logic-depth %u\n",
                xag.num_gates(), a, xag.num_gates()-a, dv.depth());
    std::printf("  -> reversible phase oracle: %u Toffolis (2x AND) + %u bare CX (2x XOR)\n",
                2*a, 2*(xag.num_gates()-a));
    std::printf("  -> estimated CX = %u,  Toffoli budget at depth 129 is ~91-129\n",
                2*a*3 + 2*(xag.num_gates()-a));
    return 0;
}
