#include <cstdlib>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <queue>
#include <random>
#include "balance.hpp"

// This struct represeats a particular load.
// The index is an index into the original array of items to balance.
struct load{
  std::size_t index;
  double cost;

  bool operator <(const load other) const {
    if ( cost > other.cost ){
      return 1;
    } else if ( cost < other.cost ){
      return 0;
    }
    return index < other.index;
  }

};

// This struct represents a processor.
// It has a vector of loads (items to balance) that it is responsible for
struct proc{
  std::size_t index;
  double cost;
  std::vector<std::size_t> loads;

};

class procCompare
{
public:
    bool operator() (struct proc *proc0, struct proc * proc1)
    {
      if ( proc0->cost > proc1->cost ){
        return 1;
      } else if ( proc0->cost < proc1->cost ){
        return 0;
      }
      return proc0->index > proc1->index;
    }
};

// Evenly distributes loads in order
void clownBalance( std::size_t nProcs,           // Number of processors
                   std::size_t nLoads,           // Number of loads to balance
                   std::size_t *permutation,     // Final permutation of original loads
                   std::size_t *starts,           // Start of each processor's region in permuted list
                   std::size_t *ends          ){  // End of each processor's region in permuted list

  for ( int i = 0; i < nLoads; i++ ){
    permutation[i] = i;
  }

  // Initial start index
  starts[0] = 0;
  // Intermediate start and end indices
  for ( int i = 0; i < nProcs - 1; i++ ){
    ends[i] = starts[i] +  nLoads / nProcs + ( i < nLoads%nProcs );
    starts[i+1] = ends[i];
  }
  // Final end index
  ends[nProcs-1] = starts[nProcs-1] + nLoads / nProcs;

}

// Takes permutation matrix from clownBalance and randomly shuffles
// Starts and ends remain the same
void randBalance(  std::size_t nProcs,           // Number of processors
                  std::size_t nLoads,           // Number of loads to balance
                  std::size_t *permutation,     // Final permutation of original loads
                  std::size_t *starts,           // Start of each processor's region in permuted list
                  std::size_t *ends          ){  // End of each processor's region in permuted list

  clownBalance( nProcs, nLoads, permutation, starts, ends);
  // Mersenne Twist Random Number Generator
  std::mt19937_64 rng;
  rng.seed(13375);

  // Shuffle permutation array
  std::shuffle( permutation, permutation + nLoads, rng );

}

// Balance nLoads based on work of each load
void ffdBalance(  std::size_t nProcs,           // Number of processors
                  std::size_t nLoads,           // Number of loads to balance
                  double      *costs,           // Cost of each load
                  std::size_t *permutation,     // Final permutation of original loads
                  std::size_t *starts,           // Start of each processor's region in permuted list
                  std::size_t *ends          ){  // End of each processor's region in permuted list

  struct load *loads =  new struct load[nLoads];
  struct proc procs[nProcs];

  std::priority_queue<struct proc*, std::vector<struct proc*>, procCompare > procsQueue;

  // Give loads indices and costs
  for ( std::size_t idx = 0; idx < nLoads; idx++ ){
    loads[idx] = { idx, costs[idx] };
  }
  // Initialize procs
  for ( std::size_t idx = 0; idx < nProcs; idx++ ){
    procs[idx] = { idx, 0, std::vector<std::size_t>() };
    procsQueue.push( procs + idx );
  }

  std::sort( loads, loads + nLoads );

  for ( std::size_t idx = 0; idx < nLoads; idx++ ){
    struct proc *earliest = procsQueue.top();
    earliest->loads.push_back(loads[idx].index);
    earliest->cost += loads[idx].cost;
    procsQueue.pop();
    procsQueue.push( earliest );
  }

  // Initial start index
  starts[0] = 0;
  // Intermediate start and end indices
  for ( int i = 0; i < nProcs - 1; i++ ){
    ends[i] = starts[i] + procs[i].loads.size();
    starts[i+1] = ends[i];
  }
  // Final end index
  ends[nProcs-1] = starts[nProcs-1] + procs[nProcs-1].loads.size();

  int loadCount = 0;
  // Populate permutation index array
  for ( std::size_t i = 0; i < nProcs; i++ ) {
    for ( std::size_t j = 0; j < procs[i].loads.size(); j ++ ){
      permutation[loadCount] = procs[i].loads[j];
      loadCount++;
    }
  }

  delete[] loads;

}
