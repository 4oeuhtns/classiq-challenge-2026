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
    std::printf("%-12s gates: %5u   depth: %3u\n", label,
    xag.num_gates(), dv.depth());
}

int main(int argc, char** argv) {
    mockturtle::xag_network xag;
    auto const result =
    lorina::read_pla("../logo.pla",
    mockturtle::pla_reader(xag));
    mockturtle::xag_npn_resynthesis<mockturtle::xag_network>
    resyn;
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