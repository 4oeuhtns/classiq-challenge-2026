#include <cstdio>
#include <caterpillar/caterpillar.hpp>
#include <mockturtle/networks/aig.hpp>
#include <tweedledum/networks/netlist.hpp>

using namespace caterpillar;
using namespace mockturtle;
using namespace tweedledum;

int main() {
    aig_network sorter;
    const auto a = sorter.create_pi();
    const auto b = sorter.create_pi();
    const auto c = sorter.create_pi();

    const auto w1 = sorter.create_and(a, b);
    const auto w2 = sorter.create_and(c, w1);
    const auto w3 = sorter.create_and(!a, !b);
    const auto w4 = sorter.create_and(!c, !w1);
    const auto w5 = sorter.create_and(!w3, !w4);
    const auto w6 = sorter.create_or(c, !w3);

    sorter.create_po(w2);
    sorter.create_po(w5);
    sorter.create_po(w6);

    netlist<stg_gate> circ;
    pebbling_mapping_strategy<aig_network,
    bsat_pebble_solver<aig_network>> strategy;
    logic_network_synthesis_stats st;
    logic_network_synthesis(circ, sorter, strategy, {}, {},
    &st);

    std::printf("qubits: %u\n", circ.num_qubits());
    std::printf("gates:  %u\n", circ.num_gates());
    return 0;
}