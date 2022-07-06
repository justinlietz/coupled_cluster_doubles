// Copyright (c) 2015-2016, Justin Gage Lietz
// All rights reserved.

// CCD code for arbitrary basis
// Written by Justin Gage Lietz starting in June 2015

#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include <omp.h>
#include <mpi.h>
#include "parse.hpp"
#include "load.hpp"
#include "calc.hpp"
#include "Chain.hpp"
#include "SPBasis.hpp"
#include "InfMatterSPBasis.hpp"
#include "ElectronGasSPBasis.hpp"
#include "PairingSPBasis.hpp"
#include "MBPT2Corr.hpp"
#include "balance.hpp"


int main(int argc, char * argv[])
{

  MPI_Init(0,0);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	printf("rank: %d size: %d\n",rank, size);
  SPBasis *basis;
  double tolerance = 1.e-8;
  int saveMemory = 0;
  char *timeFile = NULL;
  char timeMode;
  char *modelFile = NULL;
  int verbose = 0;

  char *parseMem = parse(argc,
                         argv,
                         &basis,
                         &tolerance,
                         &saveMemory,
                         &timeFile,
                         &timeMode,
                         &modelFile,
                         &verbose,
                         rank);

  double totalTimeTic = omp_get_wtime();
  double setupTimeTic = omp_get_wtime();

  //Check build
  if(verbose >= 2){
    if( rank == 0){
      basis->printBasis();
    }
  }

  // calculate reference energy BEFORE
  // rotating energies
  double Eref = basis->referenceEnergy();

  // rotate sp energies to HF energies
  basis->rotateSpEnergiesToNormalOrdered();

  //Check rotation
  if(verbose >= 2){
    if( rank == 0 ){
      basis->printBasis();
    }
  }
  // |i,j> are the two body states
  // set up the indexing scheme for these
  //basis->setUpTwoStateChannels();
  basis->setUpChannelValues();
  basis->setUpChannelDims();

  std::size_t startChannels[size];
  std::size_t endChannels[size];
  std::size_t chans_per_proc;

  std::size_t startParticleChannels[size];
  std::size_t endParticleChannels[size];

  std::size_t startHoleChannels[size];
  std::size_t endHoleChannels[size];

  std::size_t startModChannels[size];
  std::size_t endModChannels[size];

  double *costs = new double[basis->nChannels];

  for(std::size_t ichan = 0; ichan < basis->nChannels; ichan++){
    double ppDim = basis->chanDims[ichan].ppDim;
    double hhDim = basis->chanDims[ichan].hhDim;
    double phDim = basis->chanDims[ichan].phDim;
    double hpDim = basis->chanDims[ichan].hpDim;
    double phModDim = basis->chanModDims[ichan].phDim;
    double hpModDim = basis->chanModDims[ichan].hpDim;
    costs[ichan] = 0;
    costs[ichan] += hhDim * hhDim;
    costs[ichan] += hhDim * hpDim;
    costs[ichan] += hhDim * ppDim;
    costs[ichan] += hpDim * phDim;
    costs[ichan] += hpDim * ppDim;
    costs[ichan] += ppDim * ppDim;
    costs[ichan] += hpDim * phDim;
    costs[ichan] += hpDim * hpDim;
  }

  ffdBalance( size, basis->nChannels, costs, basis->channelIndices, startChannels, endChannels);
  ffdBalance( size, basis->nChannels, costs, basis->modChannelIndices, startModChannels, endModChannels);
  //randBalance( size, basis->nChannels, basis->channelIndices, startChannels, endChannels);
  //clownBalance( size, basis->nChannels, basis->channelIndices, startChannels, endChannels);

  // for(int ichan=0; ichan < basis->nParticles; ichan++){
  //   printf("ichan: %zu, hppDim: %zu\n", ichan, basis->threeBodyChanDims[ichan].hppDim);
  // }
  // for(int ichan=basis->nParticles; ichan < basis->nSpstates; ichan++){
  //   printf("ichan: %zu, hppDim: %zu\n", ichan, basis->threeBodyChanDims[ichan].phhDim);
  // }

  //Particle balance
  //std::size_t * particleIndices = new std::size_t[basis->nSpstates-basis->nParticles];
  clownBalance( size, basis->nSpstates-basis->nParticles, basis->spChannelIndices, startParticleChannels, endParticleChannels );
  //Hole balance
  //std::size_t * holeIndices = new std::size_t[basis->nParticles];
  clownBalance( size, basis->nParticles, basis->spChannelIndices, startHoleChannels, endHoleChannels);

  delete[] costs;

  basis->setUpInverseChannelIndicesAndPermuteDims();
  basis->setUpChannelMaps();
  if(rank == 0){
    if(verbose >= 1){
      printf("nChannels: %llu\n", basis->nChannels);
    }
  }

  std::size_t *ppDims = new std::size_t[basis->nChannels];
  std::size_t *hhDims = new std::size_t[basis->nChannels];
  std::size_t *phDims = new std::size_t[basis->nChannels];
  std::size_t *hpDims = new std::size_t[basis->nChannels];
  std::size_t *phhDims = new std::size_t[basis->nSpstates - basis->nParticles];
  std::size_t *hppDims = new std::size_t[basis->nParticles];
  std::size_t *oneDims = new std::size_t[basis->nSpstates];
  std::size_t *phModDims = new std::size_t[basis->nChannels];
  std::size_t *hpModDims = new std::size_t[basis->nChannels];

  for(std::size_t ichan = 0; ichan < basis->nChannels; ichan++){
    ppDims[ichan] = basis->chanDims[ichan].ppDim;
    hhDims[ichan] = basis->chanDims[ichan].hhDim;
    phDims[ichan] = basis->chanDims[ichan].phDim;
    hpDims[ichan] = basis->chanDims[ichan].hpDim;
    phModDims[ichan] = basis->chanModDims[ichan].phDim;
    hpModDims[ichan] = basis->chanModDims[ichan].hpDim;
  }
  for(std::size_t ichan = 0; ichan < basis->nParticles; ichan++){
    hppDims[ichan] = basis->threeBodyChanDims[ichan].hppDim;
  }
  for(std::size_t ichan = basis->nParticles; ichan < basis->nSpstates; ichan++){
    phhDims[ichan - basis->nParticles] = basis->threeBodyChanDims[ichan].phhDim;
  }
  for(std::size_t ichan = 0; ichan < basis->nSpstates; ichan++){
    oneDims[ichan] = 1;
  }

  if( verbose >= 2 ){
    MPI_Barrier(MPI_COMM_WORLD);
    if( rank == 0){
      printf("Setup time: %g\n", omp_get_wtime() - setupTimeTic);
    }
  }

  double loadTimeTic = omp_get_wtime();

  std::size_t t2Memory = 0;

  t2Memory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startModChannels[rank], endModChannels[rank], basis->nChannels, phModDims + startModChannels[rank], hpModDims + startModChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startParticleChannels[rank], endParticleChannels[rank], basis->nSpstates - basis->nParticles, oneDims + startParticleChannels[rank], phhDims + startParticleChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startHoleChannels[rank], endHoleChannels[rank], basis->nParticles, hppDims + startHoleChannels[rank], oneDims + startHoleChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startModChannels[rank], endModChannels[rank], basis->nChannels, phModDims + startModChannels[rank], hpModDims + startModChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startParticleChannels[rank], endParticleChannels[rank], basis->nSpstates - basis->nParticles, oneDims + startParticleChannels[rank], phhDims + startParticleChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startParticleChannels[rank], endParticleChannels[rank], basis->nSpstates - basis->nParticles, oneDims + startParticleChannels[rank], phhDims + startParticleChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startHoleChannels[rank], endHoleChannels[rank], basis->nParticles, hppDims + startHoleChannels[rank], oneDims + startHoleChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startHoleChannels[rank], endHoleChannels[rank], basis->nParticles, hppDims + startHoleChannels[rank], oneDims + startHoleChannels[rank]);

  t2Memory += Chain<ChainIndex>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<ChainIndex>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<ChainIndex>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<ChainIndex>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<ChainIndex>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<ChainIndex>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<ChainIndex>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<ChainIndex>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);

  t2Memory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  t2Memory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);

  if( verbose >= 0 ){
    printf("rank %d t2 chains require %g GB of memory\n", rank, t2Memory/1.e9);
    std::size_t globalT2Memory = 0;
    MPI_Reduce(&t2Memory, &globalT2Memory, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if( rank == 0){
      printf("all t2 chains require %g GB of memory\n", globalT2Memory/1.e9);
    }
  }


  Chain<double> t2_pphh(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<double> t2Diff_pphh(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<double> t2Diff_phhp(startModChannels[rank], endModChannels[rank], basis->nChannels, phModDims + startModChannels[rank], hpModDims + startModChannels[rank]);
  Chain<double> t2Diff_p2_p1hh(startParticleChannels[rank], endParticleChannels[rank], basis->nSpstates - basis->nParticles, oneDims + startParticleChannels[rank], phhDims + startParticleChannels[rank]);
  Chain<double> t2Diff_h2pp_h1(startHoleChannels[rank], endHoleChannels[rank], basis->nParticles, hppDims + startHoleChannels[rank], oneDims + startHoleChannels[rank]);
  Chain<double> t2Old_pphh(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<double> t2Old_phhp(startModChannels[rank], endModChannels[rank], basis->nChannels, phModDims + startModChannels[rank], hpModDims + startModChannels[rank]);
  Chain<double> t2Old_p1_p2hh(startParticleChannels[rank], endParticleChannels[rank], basis->nSpstates - basis->nParticles, oneDims + startParticleChannels[rank], phhDims + startParticleChannels[rank]);
  Chain<double> t2Old_p2_p1hh(startParticleChannels[rank], endParticleChannels[rank], basis->nSpstates - basis->nParticles, oneDims + startParticleChannels[rank], phhDims + startParticleChannels[rank]);
  Chain<double> t2Old_h1pp_h2(startHoleChannels[rank], endHoleChannels[rank], basis->nParticles, hppDims + startHoleChannels[rank], oneDims + startHoleChannels[rank]);
  Chain<double> t2Old_h2pp_h1(startHoleChannels[rank], endHoleChannels[rank], basis->nParticles, hppDims + startHoleChannels[rank], oneDims + startHoleChannels[rank]);

  t2_pphh.zero();

  Chain<ChainIndex> map_pphh_iab_j_hpp_h(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<ChainIndex> map_pphh_jab_i_hpp_h(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<ChainIndex> map_pphh_a_bij_p_phh(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<ChainIndex> map_pphh_b_aij_p_phh(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<ChainIndex> map_pphh_ai_jb_phhp(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<ChainIndex> map_pphh_aj_ib_phhp(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<ChainIndex> map_pphh_bi_ja_phhp(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<ChainIndex> map_pphh_bj_ia_phhp(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);

  Chain<double> t2Diff_h2pp_h1_iab_j_pphh(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<double> t2Diff_h2pp_h1_jab_i_pphh(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<double> t2Diff_p2_p1hh_a_bij_pphh(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<double> t2Diff_p2_p1hh_b_aij_pphh(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<double> t2Diff_phhp_ai_jb_pphh(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<double> t2Diff_phhp_aj_ib_pphh(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<double> t2Diff_phhp_bi_ja_pphh(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);
  Chain<double> t2Diff_phhp_bj_ia_pphh(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], hhDims + startChannels[rank]);

  Chain<double> *vnn_hhhh;
  Chain<double> *vnn_hhhp;
  Chain<double> *vnn_hhpp;
  Chain<double> *vnn_hpph;
  Chain<double> *vnn_hppp;
  Chain<double> *vnn_pppp;

  Chain<double> *vnn_phh_p;
  Chain<double> *vnn_h_hpp;

  Chain<double> *vnn_hhpp_hpph_mod;
  Chain<double> *vnn_hpph_hphp_mod;

  if(!saveMemory){
    std::size_t vnnMemory = 0;

    vnnMemory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, hhDims + startChannels[rank], hhDims + startChannels[rank]);
    vnnMemory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, hhDims + startChannels[rank], hpDims + startChannels[rank]);
    vnnMemory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, hhDims + startChannels[rank], ppDims + startChannels[rank]);
    vnnMemory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, hpDims + startChannels[rank], phDims + startChannels[rank]);
    vnnMemory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, hpDims + startChannels[rank], ppDims + startChannels[rank]);
    vnnMemory += Chain<double>::ChainMemory(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], ppDims + startChannels[rank]);

    vnnMemory += Chain<double>::ChainMemory(startParticleChannels[rank], endParticleChannels[rank], basis->nSpstates - basis->nParticles, phhDims + startParticleChannels[rank], oneDims + startParticleChannels[rank]);
    vnnMemory += Chain<double>::ChainMemory(startHoleChannels[rank], endHoleChannels[rank], basis->nParticles, oneDims + startHoleChannels[rank], hppDims + startHoleChannels[rank]);

    vnnMemory += Chain<double>::ChainMemory(startModChannels[rank], endModChannels[rank], basis->nChannels, hpModDims + startModChannels[rank], phModDims + startModChannels[rank]);
    vnnMemory += Chain<double>::ChainMemory(startModChannels[rank], endModChannels[rank], basis->nChannels, hpModDims + startModChannels[rank], hpModDims + startModChannels[rank]);

    if( verbose >= 0 ){
      printf("rank %d vnn chains require %g GB of memory\n", rank, vnnMemory/1.e9);
      std::size_t globalVnnMemory = 0;
      MPI_Reduce(&vnnMemory, &globalVnnMemory, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      if( rank == 0){
        printf("all vnn chains require %g GB of memory\n", globalVnnMemory/1.e9);
      }
    }


    vnn_hhhh = new Chain<double>(startChannels[rank], endChannels[rank], basis->nChannels, hhDims + startChannels[rank], hhDims + startChannels[rank]);
    vnn_hhhp = new Chain<double>(startChannels[rank], endChannels[rank], basis->nChannels, hhDims + startChannels[rank], hpDims + startChannels[rank]);
    vnn_hhpp = new Chain<double>(startChannels[rank], endChannels[rank], basis->nChannels, hhDims + startChannels[rank], ppDims + startChannels[rank]);
    vnn_hpph = new Chain<double>(startChannels[rank], endChannels[rank], basis->nChannels, hpDims + startChannels[rank], phDims + startChannels[rank]);
    vnn_hppp = new Chain<double>(startChannels[rank], endChannels[rank], basis->nChannels, hpDims + startChannels[rank], ppDims + startChannels[rank]);
    vnn_pppp = new Chain<double>(startChannels[rank], endChannels[rank], basis->nChannels, ppDims + startChannels[rank], ppDims + startChannels[rank]);

    vnn_phh_p = new Chain<double>(startParticleChannels[rank], endParticleChannels[rank], basis->nSpstates - basis->nParticles, phhDims + startParticleChannels[rank], oneDims + startParticleChannels[rank]);
    vnn_h_hpp = new Chain<double>(startHoleChannels[rank], endHoleChannels[rank], basis->nParticles, oneDims + startHoleChannels[rank], hppDims + startHoleChannels[rank]);

    vnn_hhpp_hpph_mod = new Chain<double>(startModChannels[rank], endModChannels[rank], basis->nChannels, hpModDims + startModChannels[rank], phModDims + startModChannels[rank]);
    vnn_hpph_hphp_mod = new Chain<double>(startModChannels[rank], endModChannels[rank], basis->nChannels, hpModDims + startModChannels[rank], hpModDims + startModChannels[rank]);

    loadVnn_hhhh(startChannels[rank],
                 endChannels[rank],
                 vnn_hhhh,
                 basis);
    loadVnn_hhhp(startChannels[rank],
                 endChannels[rank],
                 vnn_hhhp,
                 basis);
    loadVnn_hhpp(startChannels[rank],
                 endChannels[rank],
                 vnn_hhpp,
                 basis);
    loadVnn_hpph(startChannels[rank],
                 endChannels[rank],
                 vnn_hpph,
                 basis);
    loadVnn_hppp(startChannels[rank],
                 endChannels[rank],
                 vnn_hppp,
                 basis);
    loadVnn_pppp(startChannels[rank],
                 endChannels[rank],
                 vnn_pppp,
                 basis);

// 	printf("pppp mat elems:\n");
// 	size_t a,b,c,d;
// 	double matEl;
// 	for(size_t iBlock = 0; iBlock < basis->nChannels; iBlock++){
// //		printf("iBlock: %zu\n", iBlock);
// 		for(size_t iBra = 0; iBra < basis->chanDims[iBlock].ppDim; iBra++){
// 			for(size_t iKet = 0; iKet < basis->chanDims[iBlock].ppDim; iKet++){
// 				a = basis->chanMaps[iBlock].ppMap[iBra].p;
// 				b = basis->chanMaps[iBlock].ppMap[iBra].q;
// 				c = basis->chanMaps[iBlock].ppMap[iKet].p;
// 				d = basis->chanMaps[iBlock].ppMap[iKet].q;
// 				matEl = vnn_pppp->get(iBlock,iBra,iKet);
//
// 				printf("%zu %zu %zu %zu %f\n", a,b,c,d,matEl);
// 			}
// 		}
// 	}


    loadVnn_phh_p(startParticleChannels[rank],
                  endParticleChannels[rank],
                  vnn_phh_p,
                  basis);
    loadVnn_h_hpp(startHoleChannels[rank],
                  endHoleChannels[rank],
                  vnn_h_hpp,
                  basis);

    loadVnn_hhpp_hpph_mod(startModChannels[rank],
                          endModChannels[rank],
                          vnn_hhpp_hpph_mod,
                          basis);
    loadVnn_hpph_hphp_mod(startModChannels[rank],
                          endModChannels[rank],
                          vnn_hpph_hphp_mod,
                          basis);
  }


  delete[] ppDims;
  delete[] hhDims;
  delete[] phDims;
  delete[] hpDims;
  delete[] phhDims;
  delete[] hppDims;
  delete[] oneDims;
  delete[] phModDims;
  delete[] hpModDims;


  loadMaps(startChannels[rank],
           endChannels[rank],
           map_pphh_iab_j_hpp_h,
           map_pphh_jab_i_hpp_h,
           map_pphh_a_bij_p_phh,
           map_pphh_b_aij_p_phh,
           map_pphh_ai_jb_phhp,
           map_pphh_aj_ib_phhp,
           map_pphh_bi_ja_phhp,
           map_pphh_bj_ia_phhp,
           basis);

  MapDistributedData *mapFromDistributed_pphh_ai_jb_phhp;
  MapDistributedData *mapFromDistributed_pphh_aj_ib_phhp;
  MapDistributedData *mapFromDistributed_pphh_bi_ja_phhp;
  MapDistributedData *mapFromDistributed_pphh_bj_ia_phhp;
  MapDistributedData *mapFromDistributed_pphh_iab_j_hpp_h;
  MapDistributedData *mapFromDistributed_pphh_jab_i_hpp_h;
  MapDistributedData *mapFromDistributed_pphh_a_bij_p_phh;
  MapDistributedData *mapFromDistributed_pphh_b_aij_p_phh;
  mapFromDistributed_pphh_ai_jb_phhp = map_pphh_ai_jb_phhp.mapFromDistributedCreate(startChannels[rank], endChannels[rank], startChannels, endChannels, MPI_COMM_WORLD);
  mapFromDistributed_pphh_aj_ib_phhp = map_pphh_aj_ib_phhp.mapFromDistributedCreate(startChannels[rank], endChannels[rank], startChannels, endChannels, MPI_COMM_WORLD);
  mapFromDistributed_pphh_bi_ja_phhp = map_pphh_bi_ja_phhp.mapFromDistributedCreate(startChannels[rank], endChannels[rank], startChannels, endChannels, MPI_COMM_WORLD);
  mapFromDistributed_pphh_bj_ia_phhp = map_pphh_bj_ia_phhp.mapFromDistributedCreate(startChannels[rank], endChannels[rank], startChannels, endChannels, MPI_COMM_WORLD);
  mapFromDistributed_pphh_iab_j_hpp_h = map_pphh_iab_j_hpp_h.mapFromDistributedCreate(startChannels[rank], endChannels[rank], startHoleChannels, endHoleChannels, MPI_COMM_WORLD);
  mapFromDistributed_pphh_jab_i_hpp_h = map_pphh_jab_i_hpp_h.mapFromDistributedCreate(startChannels[rank], endChannels[rank], startHoleChannels, endHoleChannels, MPI_COMM_WORLD);
  mapFromDistributed_pphh_a_bij_p_phh = map_pphh_a_bij_p_phh.mapFromDistributedCreate(startChannels[rank], endChannels[rank], startParticleChannels, endParticleChannels, MPI_COMM_WORLD);
  mapFromDistributed_pphh_b_aij_p_phh = map_pphh_b_aij_p_phh.mapFromDistributedCreate(startChannels[rank], endChannels[rank], startParticleChannels, endParticleChannels, MPI_COMM_WORLD);

  std::size_t mapFromBytes = 0;
  mapFromBytes += MDD_Memory(mapFromDistributed_pphh_ai_jb_phhp);
  mapFromBytes += MDD_Memory(mapFromDistributed_pphh_aj_ib_phhp);
  mapFromBytes += MDD_Memory(mapFromDistributed_pphh_bi_ja_phhp);
  mapFromBytes += MDD_Memory(mapFromDistributed_pphh_bj_ia_phhp);
  mapFromBytes += MDD_Memory(mapFromDistributed_pphh_iab_j_hpp_h);
  mapFromBytes += MDD_Memory(mapFromDistributed_pphh_jab_i_hpp_h);
  mapFromBytes += MDD_Memory(mapFromDistributed_pphh_a_bij_p_phh);
  mapFromBytes += MDD_Memory(mapFromDistributed_pphh_b_aij_p_phh);

  if( verbose >= 0 ){
    printf("rank %d mapFromDistributedMaps require %g GB of memory\n", rank, mapFromBytes/1.e9);
    std::size_t globalMapFromMemory = 0;
    MPI_Reduce(&mapFromBytes, &globalMapFromMemory, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if( rank == 0){
      printf("all mapFromDistributedMaps require %g GB of memory\n", globalMapFromMemory/1.e9);
    }
  }

  MapDistributedData *mapToDistributed_pphh_iab_j_hpp_h;
  MapDistributedData *mapToDistributed_pphh_jab_i_hpp_h;
  MapDistributedData *mapToDistributed_pphh_a_bij_p_phh;
  MapDistributedData *mapToDistributed_pphh_b_aij_p_phh;
  MapDistributedData *mapToDistributed_pphh_ai_jb_phhp;

  mapToDistributed_pphh_iab_j_hpp_h = map_pphh_iab_j_hpp_h.mapToDistributedCreate(startChannels[rank], endChannels[rank], startHoleChannels, endHoleChannels, MPI_COMM_WORLD);
  mapToDistributed_pphh_jab_i_hpp_h = map_pphh_jab_i_hpp_h.mapToDistributedCreate(startChannels[rank], endChannels[rank], startHoleChannels, endHoleChannels, MPI_COMM_WORLD);
  mapToDistributed_pphh_a_bij_p_phh = map_pphh_a_bij_p_phh.mapToDistributedCreate(startChannels[rank], endChannels[rank], startParticleChannels, endParticleChannels, MPI_COMM_WORLD);
  mapToDistributed_pphh_b_aij_p_phh = map_pphh_b_aij_p_phh.mapToDistributedCreate(startChannels[rank], endChannels[rank], startParticleChannels, endParticleChannels, MPI_COMM_WORLD);
  mapToDistributed_pphh_ai_jb_phhp = map_pphh_ai_jb_phhp.mapToDistributedCreate(startChannels[rank], endChannels[rank], startChannels, endChannels, MPI_COMM_WORLD);

  std::size_t mapToBytes = 0;
  mapToBytes += MDD_Memory(mapToDistributed_pphh_iab_j_hpp_h);
  mapToBytes += MDD_Memory(mapToDistributed_pphh_jab_i_hpp_h);
  mapToBytes += MDD_Memory(mapToDistributed_pphh_a_bij_p_phh);
  mapToBytes += MDD_Memory(mapToDistributed_pphh_b_aij_p_phh);
  mapToBytes += MDD_Memory(mapToDistributed_pphh_ai_jb_phhp);

  if( verbose >= 0 ){
    printf("rank %d mapToDistributedMaps require %g GB of memory\n", rank, mapToBytes/1.e9);
    std::size_t globalMapToMemory = 0;
    MPI_Reduce(&mapToBytes, &globalMapToMemory, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if( rank == 0){
      printf("all mapToDistributedMaps require %g GB of memory\n", globalMapToMemory/1.e9);
    }
  }

  if( verbose >= 2 ){
    MPI_Barrier(MPI_COMM_WORLD);
    if( rank == 0){
      printf("Load time: %g s\n", omp_get_wtime() - loadTimeTic);
    }
  }

  double iterTimeTic = omp_get_wtime();

  // Calculate MBPT2 energy
  double myCorrMBPT2;
  double corrMBPT2 = 0.;
  double mbpt_time_0, mbpt_time_1;
  mbpt_time_0 = omp_get_wtime();
  if(saveMemory){
    myCorrMBPT2 = MBPT2FreeCorr(startChannels[rank],endChannels[rank], basis);
  }else{
    myCorrMBPT2 = MBPT2Corr(startChannels[rank],endChannels[rank],vnn_hhpp, basis);
  }
  MPI_Reduce(&myCorrMBPT2, &corrMBPT2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  mbpt_time_1 = omp_get_wtime();
  printf("mbpt loops time: %f\n", mbpt_time_1 - mbpt_time_0);

  if( rank == 0 ){
    if(verbose >=0){
      printf("Reference Energy: %.17g\n", Eref);
      printf("Reference Energy/A: %.17g\n", Eref/basis->nParticles);
      printf("MBPT2Corr: %.17g\n", corrMBPT2);
      printf("MBPT2Corr/A: %.17g\n", corrMBPT2/basis->nParticles);
      printf("E_mbpt2: %.17g\n", (Eref + corrMBPT2));
      printf("E_mbpt2/A: %.17g\n", (Eref + corrMBPT2)/basis->nParticles);
    }
  }

  double energyDiff = 1.0;
  double myCorrCCD;
  double corrCCD = 100.0;
  double corrEnergyPrev = 100.0;
  unsigned long iterations = 0;

  t2_pphh.zero();
  t2Old_pphh.zero();
  t2Old_h2pp_h1.zero();
  t2Old_h1pp_h2.zero();
  t2Old_p1_p2hh.zero();
  t2Old_p2_p1hh.zero();
  t2Old_phhp.zero();

  double cc_time_0, cc_time_1;
  cc_time_0 = omp_get_wtime();

  double mixing = 0.5;
  int max_iters = 100;
  // Begin CCD iterations. Iterate until energy converged under tolerance
  while(std::abs(energyDiff) > tolerance){
    if(iterations > max_iters){
      if( rank == 0 ){
        if( verbose >= 0 ){
          printf("\n\n\nMax iterations reached. Halting.\n\n\n");
        }
      }
      break;
    }

    if(saveMemory){
      // 3BodyChans calculated on all procs
      calcFreeT2Diff_p2_p1hh(startParticleChannels[rank],
                             endParticleChannels[rank],
                             t2Old_p1_p2hh,
                             t2Old_p2_p1hh,
                             t2Diff_p2_p1hh,
                             basis);
      calcFreeT2Diff_h2pp_h1(startHoleChannels[rank],
                             endHoleChannels[rank],
                             t2Old_h1pp_h2,
                             t2Old_h2pp_h1,
                             t2Diff_h2pp_h1,
                             basis);
      calcFreeT2Diff_phhp(startChannels[rank],
                          endChannels[rank],
                          t2Old_phhp,
                          t2Diff_phhp,
                          basis);

      //t2Diff_phhp.updateDistributed(startChannels, endChannels, MPI_COMM_WORLD);

      calcFreeT2Diff_pphh(startChannels[rank],
                          endChannels[rank],
                          t2Old_pphh,
                          t2Diff_pphh,
                          basis);

      //t2Diff_pphh.updateDistributed(startChannels, endChannels, MPI_COMM_WORLD);

      t2Diff_h2pp_h1_iab_j_pphh.mapFromDistributed(&t2Diff_h2pp_h1, mapFromDistributed_pphh_iab_j_hpp_h);
      t2Diff_h2pp_h1_jab_i_pphh.mapFromDistributed(&t2Diff_h2pp_h1, mapFromDistributed_pphh_jab_i_hpp_h);
      t2Diff_p2_p1hh_a_bij_pphh.mapFromDistributed(&t2Diff_p2_p1hh, mapFromDistributed_pphh_a_bij_p_phh);
      t2Diff_p2_p1hh_b_aij_pphh.mapFromDistributed(&t2Diff_p2_p1hh, mapFromDistributed_pphh_b_aij_p_phh);
      t2Diff_phhp_ai_jb_pphh.mapFromDistributed(&t2Diff_phhp, mapFromDistributed_pphh_ai_jb_phhp);
      t2Diff_phhp_aj_ib_pphh.mapFromDistributed(&t2Diff_phhp, mapFromDistributed_pphh_aj_ib_phhp);
      t2Diff_phhp_bi_ja_pphh.mapFromDistributed(&t2Diff_phhp, mapFromDistributed_pphh_bi_ja_phhp);
      t2Diff_phhp_bj_ia_pphh.mapFromDistributed(&t2Diff_phhp, mapFromDistributed_pphh_bj_ia_phhp);

      myCorrCCD = calcFreeT2_pphh(startChannels[rank],
                                  endChannels[rank],
                                  t2_pphh,
                                  t2Diff_pphh,
                                  t2Old_pphh,
                                  t2Diff_h2pp_h1_iab_j_pphh,
                                  t2Diff_h2pp_h1_jab_i_pphh,
                                  t2Diff_p2_p1hh_a_bij_pphh,
                                  t2Diff_p2_p1hh_b_aij_pphh,
                                  t2Diff_phhp_ai_jb_pphh,
                                  t2Diff_phhp_aj_ib_pphh,
                                  t2Diff_phhp_bi_ja_pphh,
                                  t2Diff_phhp_bj_ia_pphh,
                                  basis,
                                  mixing);

      //t2_pphh.updateDistributed(startChannels, endChannels, MPI_COMM_WORLD);

      t2Old_pphh.set(&t2_pphh);
      t2_pphh.mapToDistributed(&t2Old_h2pp_h1, mapToDistributed_pphh_jab_i_hpp_h);
      t2_pphh.mapToDistributed(&t2Old_h1pp_h2, mapToDistributed_pphh_iab_j_hpp_h);
      t2_pphh.mapToDistributed(&t2Old_p1_p2hh, mapToDistributed_pphh_a_bij_p_phh);
      t2_pphh.mapToDistributed(&t2Old_p2_p1hh, mapToDistributed_pphh_b_aij_p_phh);
      t2_pphh.mapToDistributed(&t2Old_phhp,    mapToDistributed_pphh_ai_jb_phhp);

      corrCCD = 0.;
      MPI_Allreduce(&myCorrCCD, &corrCCD, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      energyDiff = corrCCD - corrEnergyPrev;
      corrEnergyPrev = corrCCD;
    }else{
      // 3BodyChans calculated on all procs
      calcT2Diff_p2_p1hh(startParticleChannels[rank],
                         endParticleChannels[rank],
                         vnn_phh_p,
                         t2Old_p1_p2hh,
                         t2Old_p2_p1hh,
                         t2Diff_p2_p1hh,
                         basis);

      //t2Diff_p2_p1hh.updateDistributed(startParticleChannels, endParticleChannels, MPI_COMM_WORLD);

      calcT2Diff_h2pp_h1(startHoleChannels[rank],
                         endHoleChannels[rank],
                         vnn_h_hpp,
                         t2Old_h1pp_h2,
                         t2Old_h2pp_h1,
                         t2Diff_h2pp_h1,
                         basis);

      //t2Diff_h2pp_h1.updateDistributed(startHoleChannels, endHoleChannels, MPI_COMM_WORLD);

      calcT2Diff_phhp(startModChannels[rank],
                      endModChannels[rank],
                      vnn_hhpp_hpph_mod,
                      vnn_hpph_hphp_mod,
                      t2Old_phhp,
                      t2Diff_phhp,
                      basis);

      //t2Diff_phhp.updateDistributed(startChannels, endChannels, MPI_COMM_WORLD);

      calcT2Diff_pphh(startChannels[rank],
                      endChannels[rank],
                      vnn_hhhh,
                      vnn_hhpp,
                      vnn_pppp,
                      t2Old_pphh,
                      t2Diff_pphh,
                      basis);

      //t2Diff_pphh.updateDistributed(startChannels, endChannels, MPI_COMM_WORLD);

      t2Diff_h2pp_h1_iab_j_pphh.mapFromDistributed(&t2Diff_h2pp_h1, mapFromDistributed_pphh_iab_j_hpp_h);
      t2Diff_h2pp_h1_jab_i_pphh.mapFromDistributed(&t2Diff_h2pp_h1, mapFromDistributed_pphh_jab_i_hpp_h);
      t2Diff_p2_p1hh_a_bij_pphh.mapFromDistributed(&t2Diff_p2_p1hh, mapFromDistributed_pphh_a_bij_p_phh);
      t2Diff_p2_p1hh_b_aij_pphh.mapFromDistributed(&t2Diff_p2_p1hh, mapFromDistributed_pphh_b_aij_p_phh);
      t2Diff_phhp_ai_jb_pphh.mapFromDistributed(&t2Diff_phhp, mapFromDistributed_pphh_ai_jb_phhp);
      t2Diff_phhp_aj_ib_pphh.mapFromDistributed(&t2Diff_phhp, mapFromDistributed_pphh_aj_ib_phhp);
      t2Diff_phhp_bi_ja_pphh.mapFromDistributed(&t2Diff_phhp, mapFromDistributed_pphh_bi_ja_phhp);
      t2Diff_phhp_bj_ia_pphh.mapFromDistributed(&t2Diff_phhp, mapFromDistributed_pphh_bj_ia_phhp);

      myCorrCCD = calcT2_pphh(startChannels[rank],
                              endChannels[rank],
                              vnn_hhpp,
                              t2_pphh,
                              t2Diff_pphh,
                              t2Old_pphh,
                              t2Diff_h2pp_h1_iab_j_pphh,
                              t2Diff_h2pp_h1_jab_i_pphh,
                              t2Diff_p2_p1hh_a_bij_pphh,
                              t2Diff_p2_p1hh_b_aij_pphh,
                              t2Diff_phhp_ai_jb_pphh,
                              t2Diff_phhp_aj_ib_pphh,
                              t2Diff_phhp_bi_ja_pphh,
                              t2Diff_phhp_bj_ia_pphh,
                              basis,
                              mixing);

      // t2_pphh.updateDistributed(startChannels, endChannels, MPI_COMM_WORLD);

      t2Old_pphh.set(&t2_pphh);
      t2_pphh.mapToDistributed(&t2Old_h2pp_h1, mapToDistributed_pphh_jab_i_hpp_h);
      t2_pphh.mapToDistributed(&t2Old_h1pp_h2, mapToDistributed_pphh_iab_j_hpp_h);
      t2_pphh.mapToDistributed(&t2Old_p1_p2hh, mapToDistributed_pphh_a_bij_p_phh);
      t2_pphh.mapToDistributed(&t2Old_p2_p1hh, mapToDistributed_pphh_b_aij_p_phh);
      t2_pphh.mapToDistributed(&t2Old_phhp,    mapToDistributed_pphh_ai_jb_phhp);

      corrCCD = 0.;
      MPI_Allreduce(&myCorrCCD, &corrCCD, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      energyDiff = corrCCD - corrEnergyPrev;
      corrEnergyPrev = corrCCD;
    }
    if (rank==0){
      if(verbose >= 1){
        printf("iterations: %d\n", iterations);
        printf("corrCCD: %.17g\n", corrCCD);
        printf("energy diff: %.17g\n", energyDiff);
        printf("corrCCD/A: %.17g\n", corrCCD/basis->nParticles);
        printf("energy diff/A: %.17g\n", energyDiff/basis->nParticles);
      }
    }
    iterations++;
  }

  if( verbose >= 2 ){
    MPI_Barrier(MPI_COMM_WORLD);
    if( rank == 0){
      printf("Iter time: %g s\n", omp_get_wtime() - iterTimeTic);
    }
  }

  cc_time_1 = omp_get_wtime();
  if( rank == 0 ){
    if( verbose >= 0 ){
      printf("iterations: %d\n", iterations);
      printf("CCDCorr: %.17g\n", corrCCD);
      printf("CCDCorr/A: %.17g\n", corrCCD/basis->nParticles);
      printf("E_CCD: %.17g\n", (Eref + corrCCD));
      printf("E_CCD/A: %.17g\n", (Eref + corrCCD)/basis->nParticles);
      printf("Final time: %g s\n", omp_get_wtime() - totalTimeTic);
      printf("cc_diagrams_time: %f\n", cc_time_1 - cc_time_0);
    }
  }

  // Free all that memory
  // valgrind doesnt really like conditional delete
  if(!saveMemory){
    delete vnn_hhhh;
    delete vnn_hhhp;
    delete vnn_hhpp;
    delete vnn_hpph;
    delete vnn_hppp;
    delete vnn_pppp;

    delete vnn_phh_p;
    delete vnn_h_hpp;

    delete vnn_hhpp_hpph_mod;
    delete vnn_hpph_hphp_mod;
  }
  basis->deallocate();
  basis->deallocate_sp_basis();

  delete mapFromDistributed_pphh_ai_jb_phhp;
  delete mapFromDistributed_pphh_aj_ib_phhp;
  delete mapFromDistributed_pphh_bi_ja_phhp;
  delete mapFromDistributed_pphh_bj_ia_phhp;
  delete mapFromDistributed_pphh_iab_j_hpp_h;
  delete mapFromDistributed_pphh_jab_i_hpp_h;
  delete mapFromDistributed_pphh_a_bij_p_phh;
  delete mapFromDistributed_pphh_b_aij_p_phh;
  delete mapToDistributed_pphh_iab_j_hpp_h;
  delete mapToDistributed_pphh_jab_i_hpp_h;
  delete mapToDistributed_pphh_a_bij_p_phh;
  delete mapToDistributed_pphh_b_aij_p_phh;
  delete mapToDistributed_pphh_ai_jb_phhp;

  parseFree(parseMem);

  MPI_Finalize();

  return 0;
}
