#ifndef MBPT2Corr_hpp_
#define MBPT2Corr_hpp_

#include "Chain.hpp"
#include "SPBasis.hpp"

double MBPT2Corr(std::size_t startChannel, std::size_t endChannel, Chain<double> *vnn_hhpp, SPBasis * basis);
double MBPT2GEMMCorr(std::size_t startChannel, std::size_t endChannel, Chain<double> *vnn_hhpp, SPBasis * basis);
double MBPT2FreeCorr(std::size_t startChannel, std::size_t endChannel, SPBasis * basis);

#endif
