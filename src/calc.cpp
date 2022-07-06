#include <cstddef>
#include <omp.h>
#include <cblas.h>
#include "Chain.hpp"
#include "SPBasis.hpp"
#include "calc.hpp"
#include "load.hpp"

static void calcChannelT2Diff_p2_p1hh(std::size_t    channel,
                                      double        *vnn_phh_p,
                                      Chain<double> &t2Old_p1_p2hh,
                                      Chain<double> &t2Old_p2_p1hh,
                                      Chain<double> &t2Diff_p2_p1hh,
                                      SPBasis       *basis){
  double value = 0.;
  // Calculate RHS piece of <b|X|c> = <b|X|b>, as this is diagonal.
  // so c = b.
  for(std::size_t DKL = 0; DKL < basis->threeBodyChanDims[channel+basis->nParticles].phhDim; DKL++){
    value += t2Old_p1_p2hh.get(channel, 0,DKL) * vnn_phh_p[DKL];
  }
  // Calculate C sum part of <ab|t|ij> as <b|t|a^{-1}ij>
  for(std::size_t AIJ = 0; AIJ < basis->threeBodyChanDims[channel+basis->nParticles].phhDim; AIJ++){
    t2Diff_p2_p1hh.set(channel, 0, AIJ, -0.5 * value * t2Old_p2_p1hh.get(channel, 0,AIJ));
  }
} // end calcChannelT2Diff_p2_p1hh

void calcT2Diff_p2_p1hh(std::size_t    startChannel,
                        std::size_t    endChannel,
                        Chain<double> *vnn_phh_p,
                        Chain<double> &t2Old_p1_p2hh,
                        Chain<double> &t2Old_p2_p1hh,
                        Chain<double> &t2Diff_p2_p1hh,
                        SPBasis       *basis){
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t channel = startChannel; channel < endChannel; channel++){
    calcChannelT2Diff_p2_p1hh(channel,
                              vnn_phh_p->get(channel),
                              t2Old_p1_p2hh,
                              t2Old_p2_p1hh,
                              t2Diff_p2_p1hh,
                              basis);
  }
} // end calcT2Diff_p2_p1hh

void calcFreeT2Diff_p2_p1hh(std::size_t    startChannel,
                            std::size_t    endChannel,
                            Chain<double> &t2Old_p1_p2hh,
                            Chain<double> &t2Old_p2_p1hh,
                            Chain<double> &t2Diff_p2_p1hh,
                            SPBasis       *basis){
  #pragma omp parallel
  {
    double *vnn_phh_p = new double[basis->maxChannelSize_phh_p];
    #pragma omp for schedule(dynamic)
    for(std::size_t channel = startChannel; channel < endChannel; channel++){
      loadFreeVnn_phh_p(channel, vnn_phh_p, basis);
      calcChannelT2Diff_p2_p1hh(channel,
                                vnn_phh_p,
                                t2Old_p1_p2hh,
                                t2Old_p2_p1hh,
                                t2Diff_p2_p1hh,
                                basis);
    }
    delete[] vnn_phh_p;
  }

} // end calcFreeT2Diff_p2_p1hh

// A little confusing at first, here we are calculating
// the <ab|t|ij> as <j^{-1}ab|t|i> term first.
// this is the term that already has the permutation
// operator acted on it. So (i<->j).
static void calcChannelT2Diff_h2pp_h1(std::size_t    channel,
                                      double        *vnn_h_hpp,
                                      Chain<double> &t2Old_h2pp_h1,
                                      Chain<double> &t2Old_h1pp_h2,
                                      Chain<double> &t2Diff_h2pp_h1,
                                      SPBasis       *basis){
  // Calculate RHS piece of <k|X|i> = <i|X|i>, as this is diagonal.
  // so k = i.
  double value = 0.;
  for(std::size_t LCD = 0; LCD < basis->threeBodyChanDims[channel].hppDim; LCD++){
    value += vnn_h_hpp[LCD]*t2Old_h1pp_h2.get(channel,LCD,0);
  }
  // Calculate K sum part of <ab|t|ij> as <j^{-1}ab|t|i>
  for(std::size_t JAB = 0; JAB < basis->threeBodyChanDims[channel].hppDim; JAB++){
    t2Diff_h2pp_h1.set(channel, JAB, 0, 0.5 * value * t2Old_h2pp_h1.get(channel, JAB,0));
  }
} // end calcChannelT2Diff_h2pp_h1

void calcT2Diff_h2pp_h1(std::size_t    startChannel,
                        std::size_t    endChannel,
                        Chain<double> *vnn_h_hpp,
                        Chain<double> &t2Old_h2pp_h1,
                        Chain<double> &t2Old_h1pp_h2,
                        Chain<double> &t2Diff_h2pp_h1,
                        SPBasis       *basis){
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t channel = startChannel; channel < endChannel; channel++){
    calcChannelT2Diff_h2pp_h1(channel,
                              vnn_h_hpp->get(channel),
                              t2Old_h2pp_h1,
                              t2Old_h1pp_h2,
                              t2Diff_h2pp_h1,
                              basis);
  }
} // end calcT2Diff_h2pp_h1

void calcFreeT2Diff_h2pp_h1(std::size_t    startChannel,
                            std::size_t    endChannel,
                            Chain<double> &t2Old_h2pp_h1,
                            Chain<double> &t2Old_h1pp_h2,
                            Chain<double> &t2Diff_h2pp_h1,
                            SPBasis       *basis){
  #pragma omp parallel
  {
    double *vnn_h_hpp = new double[basis->maxChannelSize_h_hpp];
    #pragma omp for schedule(dynamic)
    for(std::size_t channel = startChannel; channel < endChannel; channel++){
      loadFreeVnn_h_hpp(channel, vnn_h_hpp, basis);
      calcChannelT2Diff_h2pp_h1(channel,
                                vnn_h_hpp,
                                t2Old_h2pp_h1,
                                t2Old_h1pp_h2,
                                t2Diff_h2pp_h1,
                                basis);
    }
    delete[] vnn_h_hpp;
  }
} // end calcFreeT2Diff_h2pp_h1

static void calcChannelT2Diff_pphh(std::size_t    channel,
                                   double        *xnn_hhhh,
                                   double        *vnn_hhpp,
                                   double        *vnn_pppp,
                                   Chain<double> &t2Old_pphh,
                                   Chain<double> &t2Diff_pphh,
                                   SPBasis       *basis){
  std::size_t M, K, N;

  // First, set up <kl|X|ij>, <kl|v|ij> part in function call before this
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
                /* ALPHA  */ 0.5,
                /* A      */ vnn_hhpp,
                /* LDA    */ K,
                /* B      */ t2Old_pphh.get(channel),
                /* LDB    */ N,
                /* BETA   */ 1.0,
                /* C      */ xnn_hhhh,
                /* LDC    */ N);
  }

  // Calculate KL sum part of <ab|t|ij> using <kl|X|ij>
  M = basis->chanDims[channel].ppDim;
  K = basis->chanDims[channel].hhDim;
  N = basis->chanDims[channel].hhDim;
  if( M != 0 && K != 0 && N !=0 ){
    cblas_dgemm(/* ORDER  */ CblasRowMajor,
                /* TRANSA */ CblasNoTrans,
                /* TRANSB */ CblasNoTrans,
                /* M      */ M,
                /* N      */ N,
                /* K      */ K,
                /* ALPHA  */ 0.5,
                /* A      */ t2Old_pphh.get(channel),
                /* LDA    */ K,
                /* B      */ xnn_hhhh,
                /* LDB    */ N,
                /* BETA   */ 0.0,
                /* C      */ t2Diff_pphh.get(channel),
                /* LDC    */ N);
  }

  // Calculate CD sum part of <ab|t|ij>
  M = basis->chanDims[channel].ppDim;
  K = basis->chanDims[channel].ppDim;
  N = basis->chanDims[channel].hhDim;
  if( M != 0 && K != 0 && N !=0 ){
    cblas_dgemm(/* ORDER  */ CblasRowMajor,
                /* TRANSA */ CblasNoTrans,
                /* TRANSB */ CblasNoTrans,
                /* M      */ M,
                /* N      */ N,
                /* K      */ K,
                /* ALPHA  */ 0.5,
                /* A      */ vnn_pppp,
                /* LDA    */ K,
                /* B      */ t2Old_pphh.get(channel),
                /* LDB    */ N,
                /* BETA   */ 1.0,
                /* C      */ t2Diff_pphh.get(channel),
                /* LDC    */ N);
  }
} // end calcChannelT2Diff_pphh

void calcT2Diff_pphh(std::size_t    startChannel,
                     std::size_t    endChannel,
                     Chain<double> *vnn_hhhh,
                     Chain<double> *vnn_hhpp,
                     Chain<double> *vnn_pppp,
                     Chain<double> &t2Old_pphh,
                     Chain<double> &t2Diff_pphh,
                     SPBasis       *basis){
  #pragma omp parallel
  {
    double *xnn_hhhh = new double[vnn_hhhh->maxBufferSize];

    #pragma omp for schedule(dynamic)
    for(std::size_t channel = startChannel; channel < endChannel; channel++){
      vnn_hhhh->copy(channel, xnn_hhhh);
      calcChannelT2Diff_pphh(channel,
                             xnn_hhhh,
                             vnn_hhpp->get(channel),
                             vnn_pppp->get(channel),
                             t2Old_pphh,
                             t2Diff_pphh,
                             basis);
    }

  delete[] xnn_hhhh;
  }
} // end calcT2Diff_pphh

void calcFreeT2Diff_pphh(std::size_t    startChannel,
                         std::size_t    endChannel,
                         Chain<double> &t2Old_pphh,
                         Chain<double> &t2Diff_pphh,
                         SPBasis       *basis){
  #pragma omp parallel
  {
    double *xnn_hhhh = new double[basis->maxChannelSize_hhhh];
    double *vnn_hhpp = new double[basis->maxChannelSize_hhpp];
    double *vnn_pppp = new double[basis->maxChannelSize_pppp];

    #pragma omp for schedule(dynamic)
    for(std::size_t channel = startChannel; channel < endChannel; channel++){
      loadFreeVnn_hhhh(channel, xnn_hhhh, basis);
      loadFreeVnn_hhpp(channel, vnn_hhpp, basis);
      loadFreeVnn_pppp(channel, vnn_pppp, basis);
      calcChannelT2Diff_pphh(channel,
                             xnn_hhhh,
                             vnn_hhpp,
                             vnn_pppp,
                             t2Old_pphh,
                             t2Diff_pphh,
                             basis);
    }

    delete[] xnn_hhhh;
    delete[] vnn_hhpp;
    delete[] vnn_pppp;
  }
} // end calcFreeT2Diff_pphh

static void calcChannelT2Diff_phhp(std::size_t    channel,
                                   double        *xnn_hpph_hphp_mod,
                                   double        *vnn_hhpp_hpph_mod,
                                   Chain<double> &t2Old_phhp,
                                   Chain<double> &t2Diff_phhp,
                                   SPBasis       *basis){
  std::size_t M, K, N;

  // Set up <kb|X|cj> as <kc^{-1}|X|jb^{-1}>
  M = basis->chanModDims[channel].hpDim;
  K = basis->chanModDims[channel].phDim;
  N = basis->chanModDims[channel].hpDim;
  if( M != 0 && K != 0 && N !=0 ){
    cblas_dgemm(/* ORDER  */ CblasRowMajor,
                /* TRANSA */ CblasNoTrans,
                /* TRANSB */ CblasNoTrans,
                /* M      */ M,
                /* N      */ N,
                /* K      */ K,
                /* ALPHA  */ 0.5,
                /* A      */ vnn_hhpp_hpph_mod,
                /* LDA    */ K,
                /* B      */ t2Old_phhp.get(channel),
                /* LDB    */ N,
                /* BETA   */ 1.0,
                /* C      */ xnn_hpph_hphp_mod,
                /* LDC    */ N);
  }

  // Calculate KC sum part of <ab|t|ij> as <ai^{-}|t|jb^{-1}>
  M = basis->chanModDims[channel].phDim;
  K = basis->chanModDims[channel].hpDim;
  N = basis->chanModDims[channel].hpDim;
  if( M != 0 && K != 0 && N !=0 ){
    cblas_dgemm(/* ORDER  */ CblasRowMajor,
                /* TRANSA */ CblasNoTrans,
                /* TRANSB */ CblasNoTrans,
                /* M      */ M,
                /* N      */ N,
                /* K      */ K,
                /* ALPHA  */ 1.0,
                /* A      */ t2Old_phhp.get(channel),
                /* LDA    */ K,
                /* B      */ xnn_hpph_hphp_mod,
                /* LDB    */ N,
                /* BETA   */ 0.0,
                /* C      */ t2Diff_phhp.get(channel),
                /* LDC    */ N);
  }
} // end calcChannelT2Diff_phhp

void calcT2Diff_phhp(std::size_t    startChannel,
                     std::size_t    endChannel,
                     Chain<double> *vnn_hhpp_hpph_mod,
                     Chain<double> *vnn_hpph_hphp_mod,
                     Chain<double> &t2Old_phhp,
                     Chain<double> &t2Diff_phhp,
                     SPBasis       *basis){
  #pragma omp parallel
  {
    double *xnn_hpph_hphp_mod = new double[vnn_hpph_hphp_mod->maxBufferSize];

    #pragma omp for schedule(dynamic)
    for(std::size_t channel = startChannel; channel < endChannel; channel++){
      vnn_hpph_hphp_mod->copy(channel, xnn_hpph_hphp_mod);
      calcChannelT2Diff_phhp(channel,
                             xnn_hpph_hphp_mod,
                             vnn_hhpp_hpph_mod->get(channel),
                             t2Old_phhp,
                             t2Diff_phhp,
                             basis);
    }

    delete[] xnn_hpph_hphp_mod;
  }
} // end calcT2Diff_phhp

void calcFreeT2Diff_phhp(std::size_t    startChannel,
                         std::size_t    endChannel,
                         Chain<double> &t2Old_phhp,
                         Chain<double> &t2Diff_phhp,
                         SPBasis       *basis){
  #pragma omp parallel
  {
    double *xnn_hpph_hphp_mod = new double[basis->maxChannelSize_hphp];
    double *vnn_hhpp_hpph_mod = new double[basis->maxChannelSize_hpph];

    #pragma omp for schedule(dynamic)
    for(std::size_t channel = startChannel; channel < endChannel; channel++){
      loadFreeVnn_hpph_hphp_mod(channel, xnn_hpph_hphp_mod, basis);
      loadFreeVnn_hhpp_hpph_mod(channel, vnn_hhpp_hpph_mod, basis);
      calcChannelT2Diff_phhp(channel,
                             xnn_hpph_hphp_mod,
                             vnn_hhpp_hpph_mod,
                             t2Old_phhp,
                             t2Diff_phhp,
                             basis);
    }

    delete[] xnn_hpph_hphp_mod;
    delete[] vnn_hhpp_hpph_mod;
  }
} // end calcFreeT2Diff_phhp

static double calcChannelT2_pphh(std::size_t        channel,
                                 double            *vnn_hhpp,
                                 Chain<double>     &t2_pphh,
                                 Chain<double>     &t2Diff_pphh,
                                 Chain<double>     &t2Old_pphh,
                                 Chain<double>     &t2Diff_h2pp_h1_iab_j_pphh,
                                 Chain<double>     &t2Diff_h2pp_h1_jab_i_pphh,
                                 Chain<double>     &t2Diff_p2_p1hh_a_bij_pphh,
                                 Chain<double>     &t2Diff_p2_p1hh_b_aij_pphh,
                                 Chain<double>     &t2Diff_phhp_ai_jb_pphh,
                                 Chain<double>     &t2Diff_phhp_aj_ib_pphh,
                                 Chain<double>     &t2Diff_phhp_bi_ja_pphh,
                                 Chain<double>     &t2Diff_phhp_bj_ia_pphh,
                                 SPBasis           *basis,
                                 double            mixing){
  double corr = 0.;
  for(std::size_t AB = 0; AB < basis->chanDims[channel].ppDim; AB++){
    for(std::size_t IJ = 0; IJ < basis->chanDims[channel].hhDim; IJ++){
      std::size_t a = basis->chanMaps[channel].ppMap[AB].p;
      std::size_t b = basis->chanMaps[channel].ppMap[AB].q;
      std::size_t i = basis->chanMaps[channel].hhMap[IJ].p;
      std::size_t j = basis->chanMaps[channel].hhMap[IJ].q;

      // From my second committee meeting slides
      // To update <ab|t|ij>, start with <ab|v|ij>
      double value = vnn_hhpp[IJ * basis->chanDims[channel].ppDim + AB];
      // t2Diff_pphh has the cd and kl sum information
      value += t2Diff_pphh.get(channel, AB,IJ);
      // t2Diff_phhp has the information from the kc sum
      // and the associated P(ij)P(ab) information
      // P(ij) = 1 - (i<->j) is the permutation operator
      value += t2Diff_phhp_ai_jb_pphh.get(channel, AB, IJ);
      value -= t2Diff_phhp_aj_ib_pphh.get(channel, AB, IJ);
      value -= t2Diff_phhp_bi_ja_pphh.get(channel, AB, IJ);
      value += t2Diff_phhp_bj_ia_pphh.get(channel, AB, IJ);
      // (i<->j) K sum Ter      m - K sum term
      value += t2Diff_h2pp_h1_jab_i_pphh.get(channel, AB, IJ);
      value -= t2Diff_h2pp_h1_iab_j_pphh.get(channel, AB, IJ);
      // C sum term  -  (a<->b) C sum Term
      value += t2Diff_p2_p1hh_b_aij_pphh.get(channel, AB, IJ);
      value -= t2Diff_p2_p1hh_a_bij_pphh.get(channel, AB, IJ);

      // Divide by the energy denominator, this code requires Hartree-Fock basis,
      // the one-body terms are then diagonal, and moved make the LHS of the equation
      value = value/(basis->spEnergy[i] + basis->spEnergy[j] - basis->spEnergy[a] - basis->spEnergy[b]);

      // Mixing can improve convergence.
      value = mixing*value + (1.-mixing)*t2Old_pphh.get(channel, AB, IJ);
      t2_pphh.set(channel, AB, IJ, value);
      corr += vnn_hhpp[IJ * basis->chanDims[channel].ppDim + AB]*t2_pphh.get(channel,AB,IJ);
    }
  }
  return 0.25*corr;
} // end calcChannelT2_pphh

double calcT2_pphh(std::size_t        startChannel,
                   std::size_t        endChannel,
                   Chain<double>     *vnn_hhpp,
                   Chain<double>     &t2_pphh,
                   Chain<double>     &t2Diff_pphh,
                   Chain<double>     &t2Old_pphh,
                   Chain<double>     &t2Diff_h2pp_h1_iab_j_pphh,
                   Chain<double>     &t2Diff_h2pp_h1_jab_i_pphh,
                   Chain<double>     &t2Diff_p2_p1hh_a_bij_pphh,
                   Chain<double>     &t2Diff_p2_p1hh_b_aij_pphh,
                   Chain<double>     &t2Diff_phhp_ai_jb_pphh,
                   Chain<double>     &t2Diff_phhp_aj_ib_pphh,
                   Chain<double>     &t2Diff_phhp_bi_ja_pphh,
                   Chain<double>     &t2Diff_phhp_bj_ia_pphh,
                   SPBasis           *basis,
                   double            mixing){
  double corr = 0.;
  #pragma omp parallel for reduction(+:corr) schedule(dynamic)
  for(std::size_t channel = startChannel; channel < endChannel; channel++){
    corr += calcChannelT2_pphh(channel,
                               vnn_hhpp->get(channel),
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
  }
  return corr;
} // end calcT2_pphh

double calcFreeT2_pphh(std::size_t        startChannel,
                       std::size_t        endChannel,
                       Chain<double>     &t2_pphh,
                       Chain<double>     &t2Diff_pphh,
                       Chain<double>     &t2Old_pphh,
                       Chain<double>     &t2Diff_h2pp_h1_iab_j_pphh,
                       Chain<double>     &t2Diff_h2pp_h1_jab_i_pphh,
                       Chain<double>     &t2Diff_p2_p1hh_a_bij_pphh,
                       Chain<double>     &t2Diff_p2_p1hh_b_aij_pphh,
                       Chain<double>     &t2Diff_phhp_ai_jb_pphh,
                       Chain<double>     &t2Diff_phhp_aj_ib_pphh,
                       Chain<double>     &t2Diff_phhp_bi_ja_pphh,
                       Chain<double>     &t2Diff_phhp_bj_ia_pphh,
                       SPBasis           *basis,
                       double mixing){
  double corr = 0.;
  #pragma omp parallel reduction(+:corr)
  {
    double *vnn_hhpp = new double[basis->maxChannelSize_hhpp];

    #pragma omp for schedule(dynamic)
    for(std::size_t channel = startChannel; channel < endChannel; channel++){
      loadFreeVnn_hhpp(channel, vnn_hhpp, basis);
      corr += calcChannelT2_pphh(channel,
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
    }

    delete[] vnn_hhpp;
  }
  return corr;
} // end calcFreeT2_pphh


/*
  double corr = 0.;
  for(std::size_t ichan = startChannel; ichan < endChannel; ichan++){
    for(std::size_t IJ = 0; IJ < basis->chanDims[ichan].hhDim; IJ++){
      for(std::size_t AB = 0; AB < basis->chanDims[ichan].ppDim; AB++){
        corr += vnn_hhpp->get(ichan,IJ,AB)*t2_pphh.get(ichan,AB,IJ);
      }
    }
  }
  corr = 0.25*corr;
  return corr;
*/
