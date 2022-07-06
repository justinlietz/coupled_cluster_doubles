#ifndef balance_hpp_
#define balance_hpp_
#include <cstdlib>

/**
 * Distributes number of tasks evenly across processrors
 * @param nProcs Number of processors
 * @param nLoads Number of loads to balance
 * @param permutation Final permutation of original loads
 * @param starts Start of each processor's region in permuted list
 * @param ends End of each processor's region in permuted list
 */
void clownBalance(std::size_t nProcs,
                  std::size_t nLoads,
                  std::size_t *permutation,
                  std::size_t *starts,
                  std::size_t *ends);

/**
 * Balances tasks randomly across processors
 * @param nProcs Number of processors
 * @param nLoads Number of loads to balance
 * @param permutation Final permutation of original loads
 * @param starts Start of each processor's region in permuted list
 * @param ends End of each processor's region in permuted list
 */
void randBalance(std::size_t nProcs,
                 std::size_t nLoads,
                 std::size_t *permutation,
                 std::size_t *starts,
                 std::size_t *ends);

/**
 * Balances tasks across processors using the first fit decreasing algorithm
 * @param nProcs Number of processors
 * @param nLoads Number of loads to balance
 * @param costs Cost of each load
 * @param permutation Final permutation of original loads
 * @param starts Start of each processor's region in permuted list
 * @param ends End of each processor's region in permuted list
 */
void ffdBalance(std::size_t nProcs,
                std::size_t nLoads,
                double      *costs,
                std::size_t *permutation,
                std::size_t *starts,
                std::size_t *ends);

#endif
