#include <cstdio>
#include <lorina/pla.hpp>
#include <mockturtle/networks/xag.hpp>
#include <mockturtle/io/pla_reader.hpp>
#include <mockturtle/views/depth_view.hpp>

int main() {
    mockturtle::xag_network xag;
    auto const result =
    lorina::read_pla("../logo.pla",
    mockturtle::pla_reader(xag));
    if (result != lorina::return_code::success)
    {
        std::printf("PLA read failed\n");
        return 1;
    }
    std::printf("PIs:   %u\n", xag.num_pis());
    std::printf("gates: %u\n", xag.num_gates());
    mockturtle::depth_view depth_xag{xag};
    std::printf("depth: %u\n",
    depth_xag.depth());
    return 0;
}