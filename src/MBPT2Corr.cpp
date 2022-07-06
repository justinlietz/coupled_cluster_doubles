#include "MBPT2Corr.hpp"
#include "load.hpp"
#include <cblas.h>

// calculate the 2nd Order Many-Body Perturbation Theory correlation energy
static double MBPT2ChannelCorr(std::size_t channel, double *vnn_hhpp, SPBasis * basis)
{
  double corr = 0.;
  double energyDenom;
  double v;
  std::size_t h0,h1,p0,p1;
  for(std::size_t hhIndex = 0; hhIndex < basis->chanDims[channel].hhDim; hhIndex++){
    for(std::size_t ppIndex = 0; ppIndex < basis->chanDims[channel].ppDim; ppIndex++){
      h0 = basis->chanMaps[channel].hhMap[hhIndex].p;
      h1 = basis->chanMaps[channel].hhMap[hhIndex].q;
      p0 = basis->chanMaps[channel].ppMap[ppIndex].p;
      p1 = basis->chanMaps[channel].ppMap[ppIndex].q;
      energyDenom = basis->spEnergy[h0] + basis->spEnergy[h1] - basis->spEnergy[p0] - basis->spEnergy[p1];
      v = vnn_hhpp[hhIndex * basis->chanDims[channel].ppDim + ppIndex];
	  // if(h0 == 12 && h1 == 13 && p0 == 39 && p1 == 48){
		//   printf("%zu %zu %zu %zu %f\n", h0, h1, p0, p1, v);
	  // }
      corr += v * v / energyDenom;
    }
  }
  corr = 0.25*corr;
  return corr;
} // end MBPT2Corr

// calculate the 2nd Order Many-Body Perturbation Theory correlation energy
double MBPT2Corr(std::size_t startChannel, std::size_t endChannel, Chain<double> *vnn_hhpp, SPBasis * basis)
{
  double corr = 0.;
  for(std::size_t channel = startChannel; channel < endChannel; channel++){
    corr += MBPT2ChannelCorr(channel, vnn_hhpp->get(channel), basis);
  }
  return corr;
} // end MBPT2Corr

// calculate the 2nd Order Many-Body Perturbation Theory correlation energy
double MBPT2FreeCorr(std::size_t startChannel, std::size_t endChannel, SPBasis * basis)
{
  double corr = 0.;

  // why not just allocate the size of the channel here?
  //
  double *vnn_hhpp = new double[basis->maxChannelSize_hhpp];
  for(std::size_t channel = startChannel; channel < endChannel; channel++){
    loadFreeVnn_hhpp(channel, vnn_hhpp, basis);
    corr += MBPT2ChannelCorr(channel, vnn_hhpp, basis);
  }
  delete vnn_hhpp;
  return corr;
} // end MBPT2Corr

// calculate the 2nd Order Many-Body Perturbation Theory correlation energy with gemms
static double MBPT2GEMMChannelCorr(std::size_t channel, double *vnn_hhpp, double *vnn_pphh, double *xnn_hhhh, SPBasis * basis)
{
  double corr = 0.;
  double energyDenom;
  double v;
  std::size_t h0,h1,p0,p1,M,N,K;

  for(std::size_t hhIndex = 0; hhIndex < basis->chanDims[channel].hhDim; hhIndex++){
    for(std::size_t ppIndex = 0; ppIndex < basis->chanDims[channel].ppDim; ppIndex++){
      h0 = basis->chanMaps[channel].hhMap[hhIndex].p;
      h1 = basis->chanMaps[channel].hhMap[hhIndex].q;
      p0 = basis->chanMaps[channel].ppMap[ppIndex].p;
      p1 = basis->chanMaps[channel].ppMap[ppIndex].q;
      energyDenom = basis->spEnergy[h0] + basis->spEnergy[h1] - basis->spEnergy[p0] - basis->spEnergy[p1];
      // load v^T/e into v_pphh
      v = vnn_hhpp[hhIndex * basis->chanDims[channel].ppDim + ppIndex] / energyDenom;
      vnn_pphh[ppIndex * basis->chanDims[channel].hhDim + hhIndex] = v;

	  // if(h0 == 12 && h1 == 13 && p0 == 39 && p1 == 48){
		//   printf("%zu %zu %zu %zu %f\n", h0, h1, p0, p1, v);
	  // }
    }
  }

  // <ij|x|ij> = <ij|v|ab>*(<ab|v|ij>/e)
  M = basis->chanDims[channel].hhDim;
  K = basis->chanDims[channel].ppDim;
  N = basis->chanDims[channel].hhDim;
  if( M != 0 && K != 0 && N !=0 ){
    cblas_dgemm(/* ORDER  */ CblasRowMajor,
                /* TRANSA */ CblasNoTrans,
                /* TRANSB */ CblasNoTrans,
                /* M      */ M,
                /* N      */ N,
                /* K      */ K,
                /* ALPHA  */ 0.25,
                /* A      */ vnn_hhpp,
                /* LDA    */ K,
                /* B      */ vnn_pphh,
                /* LDB    */ N,
                /* BETA   */ 0.0,
                /* C      */ xnn_hhhh,
                /* LDC    */ N);
  }

  // trace over <ij|x|ij>
  for(std::size_t hhIndex = 0; hhIndex < basis->chanDims[channel].hhDim; hhIndex++){
    corr += xnn_hhhh[hhIndex * basis->chanDims[channel].hhDim + hhIndex];
  }

  // done in dgemm now
  // corr = 0.25*corr;
  return corr;
} // end MBPT2Corr

// calculate the 2nd Order Many-Body Perturbation Theory correlation energy with gemms
double MBPT2GEMMCorr(std::size_t startChannel, std::size_t endChannel, Chain<double> *vnn_hhpp, SPBasis * basis)
{
  double corr = 0.;

  // why not just allocate the size of the channel here?
  //
  double *xnn_hhhh = new double[basis->maxChannelSize_hhhh];
  double *vnn_pphh = new double[basis->maxChannelSize_hhpp];
  for(std::size_t channel = startChannel; channel < endChannel; channel++){
    corr += MBPT2GEMMChannelCorr(channel, vnn_hhpp->get(channel), vnn_pphh, xnn_hhhh, basis);
  }
  delete xnn_hhhh;
  delete vnn_pphh;
  return corr;
} // end MBPT2Corr
