#include <cstdio>
#include <lorina/pla.hpp>
#include <mockturtle/networks/xag.hpp>
#include <mockturtle/io/pla_reader.hpp>
#include <mockturtle/views/depth_view.hpp>
#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/cut_rewriting.hpp>
#include <mockturtle/algorithms/node_resynthesis/xag_npn.hpp>
#include <mockturtle/algorithms/xag_resub_withDC.hpp>

void report(const char* label, mockturtle::xag_network& xag) {
    mockturtle::depth_view dv{xag};
    unsigned and_count = 0;
    xag.foreach_gate([&](auto n) { if (xag.is_and(n)) and_count++; });
    std::printf("%-12s gates: %5u   AND: %5u   depth: %3u\n", label,
                xag.num_gates(), and_count, dv.depth());
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s path/to/file.pla\n", argv[0]);
        return 1;
    }
    mockturtle::xag_network xag;
    auto const result = lorina::read_pla(argv[1], mockturtle::pla_reader(xag));
    if (result != lorina::return_code::success || xag.num_pis() == 0) {
        std::printf("PLA read failed\n");
        return 1;
    }
    report("raw", xag);

    mockturtle::xag_npn_resynthesis<mockturtle::xag_network> resyn;
    mockturtle::cut_rewriting_params crp;
    crp.cut_enumeration_ps.cut_size = 4;
    xag = mockturtle::cut_rewriting(xag, resyn, crp);
    xag = mockturtle::cleanup_dangling(xag);
    report("rewritten", xag);

    mockturtle::resubstitution_params rp;
    mockturtle::resubstitution_minmc_withDC(xag, rp);
    xag = mockturtle::cleanup_dangling(xag);
    report("resub", xag);

    return 0;
}
