// Copyright (c) 2015-2016, Justin Gage Lietz
// All rights reserved.

#ifndef PairingBasis_hpp_
#define PairingBasis_hpp_

#include "SPBasis.hpp"


struct PairingBundle{
  int chanSz;
};

class PairingSPBasis: public SPBasis{
public:
  double xi;
  double g;
  PairingBundle * chanValue;
  PairingBundle * chanModValue;

  // int Nspstates;
  // int Nparticles;
  // int Nchannels;
  // int ** indexMap;
  // double * spEnergy;
  // channelBundle * chanValue;
  // channelBundle * chanModValue;
  // int nPairStates?
  PairingSPBasis(std::size_t basisIndicatorIn, double xiIn, double gIn, std::size_t nPairStatesIn, std::size_t nParticlesIn);
  void generateIndexMap();
  void generateBasis();
  int checkSympqrs(std::size_t p, std::size_t q, std::size_t r, std::size_t s);
  int checkModSympqrs(std::size_t p, std::size_t q, std::size_t r, std::size_t s);
  int checkChanSym(std::size_t p, std::size_t q, std::size_t ichan);
  int checkChanModSym(std::size_t p, std::size_t q, std::size_t ichan);
  void setUpTwoStateChannels();
  void setUpChannelValues();
  void printBasis();
  void deallocate();

  std::size_t TBchanIndexFunction(std::size_t p, std::size_t q);
  std::size_t TBmodChanIndexFunction(std::size_t p, std::size_t q);
  std::size_t spIndex_from3Body(std::size_t q_inv, std::size_t r, std::size_t s);
  int spIndexExists_from3Body(std::size_t q_inv, std::size_t r, std::size_t s);

  double calcTBME(std::size_t p, std::size_t q, std::size_t r, std::size_t s);
  double calc_TBME_not_antisym(std::size_t p, std::size_t q, std::size_t r, std::size_t s);
  int kronecker_del(int i, int j);
};

#endif /* PAIRING_HPP */
