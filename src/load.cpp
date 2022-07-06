#include <unordered_map>

#include "SPBasis.hpp"
#include "Chain.hpp"
#include <omp.h>

#include "load.hpp"
void loadVnn_hhhh(std::size_t    startChannel,
                  std::size_t    endChannel,
                  Chain<double> *vnn_hhhh,
                  SPBasis       *basis){
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t ichan = startChannel; ichan < endChannel; ichan++){
    loadFreeVnn_hhhh(ichan, vnn_hhhh->get(ichan), basis);
  }
} // end loadVnn_hhhh

void loadVnn_hhhp(std::size_t    startChannel,
                  std::size_t    endChannel,
                  Chain<double> *vnn_hhhp,
                  SPBasis       *basis){
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t ichan = startChannel; ichan < endChannel; ichan++){
    loadFreeVnn_hhhp(ichan, vnn_hhhp->get(ichan), basis);
  }
} // end loadVnn_hhhp

void loadVnn_hhpp(std::size_t    startChannel,
                  std::size_t    endChannel,
                  Chain<double> *vnn_hhpp,
                  SPBasis       *basis){
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t ichan = startChannel; ichan < endChannel; ichan++){
    loadFreeVnn_hhpp(ichan, vnn_hhpp->get(ichan), basis);
  }
} // end loadVnn_hhpp

void loadVnn_hpph(std::size_t    startChannel,
                  std::size_t    endChannel,
                  Chain<double> *vnn_hpph,
                  SPBasis       *basis){
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t ichan = startChannel; ichan < endChannel; ichan++){
    loadFreeVnn_hpph(ichan, vnn_hpph->get(ichan), basis);
  }
} // end loadVnn_hpph

void loadVnn_hppp(std::size_t    startChannel,
                  std::size_t    endChannel,
                  Chain<double> *vnn_hppp,
                  SPBasis       *basis){
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t ichan = startChannel; ichan < endChannel; ichan++){
    loadFreeVnn_hppp(ichan, vnn_hppp->get(ichan), basis);
  }
} // end loadVnn_hppp

void loadVnn_pppp(std::size_t    startChannel,
                  std::size_t    endChannel,
                  Chain<double> *vnn_pppp,
                  SPBasis       *basis){
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t ichan = startChannel; ichan < endChannel; ichan++){
    loadFreeVnn_pppp(ichan, vnn_pppp->get(ichan), basis);
  }
} // end loadVnn_pppp

void loadVnn_phh_p(std::size_t    startChannel,
                   std::size_t    endChannel,
                   Chain<double> *vnn_phh_p,
                   SPBasis       *basis){
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t a = startChannel; a < endChannel; a++){
    loadFreeVnn_phh_p(a, vnn_phh_p->get(a), basis);
  }
} // end loadVnn_phh_p

void loadVnn_h_hpp(std::size_t    startChannel,
                   std::size_t    endChannel,
                   Chain<double> *vnn_h_hpp,
                   SPBasis       *basis){
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t i = startChannel; i < endChannel; i++){
    loadFreeVnn_h_hpp(i, vnn_h_hpp->get(i), basis);
  }
} // end loadVnn_h_hpp

void loadVnn_hhpp_hpph_mod(std::size_t    startChannel,
                           std::size_t    endChannel,
                           Chain<double> *vnn_hhpp_hpph_mod,
                           SPBasis       *basis){
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t ichan = startChannel; ichan < endChannel; ichan++){
    loadFreeVnn_hhpp_hpph_mod(ichan, vnn_hhpp_hpph_mod->get(ichan), basis);
  }
} // end loadVnn_hhpp_hpph_mod

void loadVnn_hpph_hphp_mod(std::size_t    startChannel,
                           std::size_t    endChannel,
                           Chain<double> *vnn_hpph_hphp_mod,
                           SPBasis       *basis){
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t ichan = startChannel; ichan < endChannel; ichan++){
    loadFreeVnn_hpph_hphp_mod(ichan, vnn_hpph_hphp_mod->get(ichan), basis);
  }
} // end loadVnn_hpph_hphp_mod

/**
 * Here begins matrix-free versions of load methods
 */

void loadFreeVnn_hhhh(std::size_t    channel,
                      double        *vnn_hhhh,
                      SPBasis       *basis){
  std::size_t hhDim;
  hhDim = basis->chanDims[channel].hhDim;

  if( hhDim > 0 ){
    for(std::size_t ihh = 0; ihh < hhDim; ihh++){
      for(std::size_t ihh2 = 0; ihh2 < hhDim; ihh2++){
        std::size_t p,q,r,s;
        p = basis->chanMaps[channel].hhMap[ihh].p;
        q = basis->chanMaps[channel].hhMap[ihh].q;
        r = basis->chanMaps[channel].hhMap[ihh2].p;
        s = basis->chanMaps[channel].hhMap[ihh2].q;
        vnn_hhhh[ihh * hhDim + ihh2] = basis->calcTBME(p,q,r,s);
      }
    }
  }
} // end loadFreeVnn_hhhh

void loadFreeVnn_hhhp(std::size_t    channel,
                      double        *vnn_hhhp,
                      SPBasis       *basis){
  std::size_t hhDim,hpDim;

  hhDim = basis->chanDims[channel].hhDim;
  hpDim = basis->chanDims[channel].hpDim;

  if( hhDim > 0 && hpDim > 0){
    for(std::size_t ihh = 0; ihh < hhDim; ihh++){
      for(std::size_t ihp = 0; ihp < hpDim; ihp++){
        std::size_t p,q,r,s;
        p = basis->chanMaps[channel].hhMap[ihh].p;
        q = basis->chanMaps[channel].hhMap[ihh].q;
        r = basis->chanMaps[channel].hpMap[ihp].p;
        s = basis->chanMaps[channel].hpMap[ihp].q;
        vnn_hhhp[ihh * hpDim + ihp] = basis->calcTBME(p,q,r,s);
      }
    }
  }
} // end loadFreeVnn_hhhp

void loadFreeVnn_hhpp(std::size_t    channel,
                      double        *vnn_hhpp,
                      SPBasis       *basis){
  std::size_t ppDim,hhDim;

  ppDim = basis->chanDims[channel].ppDim;
  hhDim = basis->chanDims[channel].hhDim;

  if( ppDim > 0 && hhDim > 0){
    for(std::size_t ihh = 0; ihh < hhDim; ihh++){
      for(std::size_t ipp = 0; ipp < ppDim; ipp++){
        std::size_t p,q,r,s;
        p = basis->chanMaps[channel].hhMap[ihh].p;
        q = basis->chanMaps[channel].hhMap[ihh].q;
        r = basis->chanMaps[channel].ppMap[ipp].p;
        s = basis->chanMaps[channel].ppMap[ipp].q;
        vnn_hhpp[ihh * ppDim + ipp] = basis->calcTBME(p,q,r,s);
      }
    }
  }
} // end loadFreeVnn_hhpp

void loadFreeVnn_hpph(std::size_t    channel,
                      double        *vnn_hpph,
                      SPBasis       *basis){
  std::size_t hpDim,phDim;

  hpDim = basis->chanDims[channel].hpDim;
  phDim = basis->chanDims[channel].phDim;

  if( hpDim > 0 && phDim > 0){
    for(std::size_t ihp = 0; ihp < hpDim; ihp++){
      for(std::size_t iph = 0; iph < phDim; iph++){
        std::size_t p,q,r,s;
        p = basis->chanMaps[channel].hpMap[ihp].p;
        q = basis->chanMaps[channel].hpMap[ihp].q;
        r = basis->chanMaps[channel].phMap[iph].p;
        s = basis->chanMaps[channel].phMap[iph].q;
        vnn_hpph[ihp * phDim + iph] = basis->calcTBME(p,q,r,s);
      }
    }
  }
} // end loadFreeVnn_hpph

void loadFreeVnn_hppp(std::size_t    channel,
                      double        *vnn_hppp,
                      SPBasis       *basis){
  std::size_t ppDim,hpDim;

  ppDim = basis->chanDims[channel].ppDim;
  hpDim = basis->chanDims[channel].hpDim;

  if( ppDim > 0 && hpDim > 0 ){
    for(std::size_t ihp = 0; ihp < hpDim; ihp++){
      for(std::size_t ipp = 0; ipp < ppDim; ipp++){
        std::size_t p,q,r,s;
        p = basis->chanMaps[channel].hpMap[ihp].p;
        q = basis->chanMaps[channel].hpMap[ihp].q;
        r = basis->chanMaps[channel].ppMap[ipp].p;
        s = basis->chanMaps[channel].ppMap[ipp].q;
        vnn_hppp[ihp * ppDim + ipp] = basis->calcTBME(p,q,r,s);
      }
    }
  }
} // end loadFreeVnn_hppp

void loadFreeVnn_pppp(std::size_t    channel,
                      double        *vnn_pppp,
                      SPBasis       *basis){
  std::size_t ppDim;

  ppDim = basis->chanDims[channel].ppDim;

  if( ppDim > 0 ){
    for(std::size_t ipp = 0; ipp < ppDim; ipp++){
      for(std::size_t ipp2 = 0; ipp2 < ppDim; ipp2++){
        std::size_t p,q,r,s;
        p = basis->chanMaps[channel].ppMap[ipp].p;
        q = basis->chanMaps[channel].ppMap[ipp].q;
        r = basis->chanMaps[channel].ppMap[ipp2].p;
        s = basis->chanMaps[channel].ppMap[ipp2].q;
        vnn_pppp[ipp * ppDim + ipp2] = basis->calcTBME(p,q,r,s);
      }
    }
  }
} // end loadFreeVnn_pppp

void loadFreeVnn_phh_p(std::size_t    channel,
                       double        *vnn_phh_p,
                       SPBasis       *basis){
  std::size_t phhDim = basis->threeBodyChanDims[channel + basis->nParticles].phhDim;
  std::size_t pDim = 1;

  if( phhDim > 0 ){
    for(std::size_t bij = 0; bij < phhDim; bij++){
      std::size_t b,i,j;
      b = basis->threeBodyChanMaps[channel + basis->nParticles].phhMap[bij].p;
      i = basis->threeBodyChanMaps[channel + basis->nParticles].phhMap[bij].q;
      j = basis->threeBodyChanMaps[channel + basis->nParticles].phhMap[bij].r;
      vnn_phh_p[bij] = basis->calcTBME(i,j,channel + basis->nParticles,b);
    }
  }
} // end loadFreeVnn_phh_p

void loadFreeVnn_h_hpp(std::size_t    channel,
                       double        *vnn_h_hpp,
                       SPBasis       *basis){
  std::size_t hDim = 1;
  std::size_t hppDim = basis->threeBodyChanDims[channel].hppDim;

  if( hppDim > 0 ){
    for(std::size_t jab = 0; jab < hppDim; jab++){
      std::size_t j,a,b;
      j = basis->threeBodyChanMaps[channel].hppMap[jab].p;
      a = basis->threeBodyChanMaps[channel].hppMap[jab].q;
      b = basis->threeBodyChanMaps[channel].hppMap[jab].r;
      vnn_h_hpp[jab] = basis->calcTBME(channel,j,a,b);
    }
  }
} // end loadFreeVnn_h_hpp

void loadFreeVnn_hhpp_hpph_mod(std::size_t    channel,
                               double        *vnn_hhpp_hpph_mod,
                               SPBasis       *basis){
  std::size_t hpModDim,phModDim;

  hpModDim = basis->chanModDims[channel].hpDim;
  phModDim = basis->chanModDims[channel].phDim;

  if( hpModDim > 0 && phModDim > 0 ){
    for(std::size_t ihp = 0; ihp < hpModDim; ihp++){
      for(std::size_t iph = 0; iph < phModDim; iph++){
        std::size_t i,a,b,j;
        i = basis->chanModMaps[channel].hpMap[ihp].p;
        a = basis->chanModMaps[channel].hpMap[ihp].q;
        b = basis->chanModMaps[channel].phMap[iph].p;
        j = basis->chanModMaps[channel].phMap[iph].q;
        // note the flipped order in calc
        vnn_hhpp_hpph_mod[ihp * phModDim + iph] = basis->calcTBME(i,j,a,b);
      }
    }
  }
} // end loadFreeVnn_hhpp_hpph_mod

void loadFreeVnn_hpph_hphp_mod(std::size_t    channel,
                               double        *vnn_hpph_hphp_mod,
                               SPBasis       *basis){
  std::size_t hpModDim;

  hpModDim = basis->chanModDims[channel].hpDim;

  if( hpModDim > 0 ){
    for(std::size_t ihp = 0; ihp < hpModDim; ihp++){
      for(std::size_t ihp2 = 0; ihp2 < hpModDim; ihp2++){
        std::size_t i,a,b,j;
        i = basis->chanModMaps[channel].hpMap[ihp].p;
        b = basis->chanModMaps[channel].hpMap[ihp].q;
        j = basis->chanModMaps[channel].hpMap[ihp2].p;
        a = basis->chanModMaps[channel].hpMap[ihp2].q;
        // note the flipped order in calc
        vnn_hpph_hphp_mod[ihp * hpModDim + ihp2] = basis->calcTBME(i,a,b,j);
      }
    }
  }
} // end loadFreeVnn_hpph_hphp_mod

struct DoubleBundle{
  std::size_t p;
  std::size_t q;
  bool operator==(const DoubleBundle &other) const
  { return (p == other.p
            && q == other.q);
  }
};

struct DoubleBundleHasher
{
  std::size_t operator()(const DoubleBundle& bundle) const
  {
    size_t seed;
    seed  = bundle.p;
    seed ^= bundle.q + 0x9e3779b9 + (seed<<6) + (seed>>2);
    return seed;
  }
};

struct TripleBundle{
  std::size_t p;
  std::size_t q;
  std::size_t r;
  bool operator==(const TripleBundle &other) const
  { return (p == other.p
            && q == other.q
            && r == other.r);
  }
};

struct TripleBundleHasher
{
  std::size_t operator()(const TripleBundle& bundle) const
  {
    size_t seed;
    seed  = bundle.p;
    seed ^= bundle.q + 0x9e3779b9 + (seed<<6) + (seed>>2);
    seed ^= bundle.r + 0x9e3779b9 + (seed<<6) + (seed>>2);
    return seed;
  }
};

struct Bundle{
  std::size_t p;
  std::size_t q;
  std::size_t r;
  std::size_t s;
  bool operator==(const Bundle &other) const
  { return (p == other.p
            && q == other.q
            && r == other.r
            && s == other.s);
  }
};

struct BundleHasher
{
  std::size_t operator()(const Bundle& bundle) const
  {
    size_t seed;
    seed  = bundle.p;
    seed ^= bundle.q + 0x9e3779b9 + (seed<<6) + (seed>>2);
    seed ^= bundle.r + 0x9e3779b9 + (seed<<6) + (seed>>2);
    seed ^= bundle.s + 0x9e3779b9 + (seed<<6) + (seed>>2);
    return seed;
  }
};

void loadMaps(std::size_t startChannel,
              std::size_t endChannel,
              Chain<ChainIndex> &map_pphh_iab_j_hpp_h,
              Chain<ChainIndex> &map_pphh_jab_i_hpp_h,
              Chain<ChainIndex> &map_pphh_a_bij_p_phh,
              Chain<ChainIndex> &map_pphh_b_aij_p_phh,
              Chain<ChainIndex> &map_pphh_ai_jb_phhp,
              Chain<ChainIndex> &map_pphh_aj_ib_phhp,
              Chain<ChainIndex> &map_pphh_bi_ja_phhp,
              Chain<ChainIndex> &map_pphh_bj_ia_phhp,
              SPBasis *basis){

  std::unordered_map<struct Bundle, ChainIndex, struct BundleHasher> hash_phhp;
  std::unordered_map<struct Bundle, ChainIndex, struct BundleHasher> hash_p_phh;
  std::unordered_map<struct Bundle, ChainIndex, struct BundleHasher> hash_hpp_h;

  // delete these
  std::unordered_map<struct DoubleBundle, std::size_t, struct DoubleBundleHasher> * hash_ph
    = new std::unordered_map<struct DoubleBundle, std::size_t, struct DoubleBundleHasher>[basis->nChannels];

  std::unordered_map<struct DoubleBundle, std::size_t, struct DoubleBundleHasher> * hash_hp
    = new std::unordered_map<struct DoubleBundle, std::size_t, struct DoubleBundleHasher>[basis->nChannels];

  std::unordered_map<struct TripleBundle, std::size_t, struct TripleBundleHasher> * hash_hpp
    = new std::unordered_map<struct TripleBundle, std::size_t, struct TripleBundleHasher>[basis->nParticles];

  std::unordered_map<struct TripleBundle, std::size_t, struct TripleBundleHasher> * hash_phh
    = new std::unordered_map<struct TripleBundle, std::size_t, struct TripleBundleHasher>[basis->nSpstates - basis->nParticles];

  double wallTimeStart = omp_get_wtime();

  size_t elemSum = 0;
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t ichan = 0; ichan < basis->nChannels; ichan++){
    std::size_t a,b,i,j;
    std::size_t phModDim, hpModDim;
    phModDim = basis->chanModDims[ichan].phDim;
    hpModDim = basis->chanModDims[ichan].hpDim;
    elemSum += phModDim*hpModDim;
    for(std::size_t AI = 0; AI < phModDim; AI++){
      a = basis->chanModMaps[ichan].phMap[AI].p;
      i = basis->chanModMaps[ichan].phMap[AI].q;
      hash_ph[ichan][(struct DoubleBundle){a,i}] = AI;
    }
    for(std::size_t JB = 0; JB < hpModDim; JB++){
      j = basis->chanModMaps[ichan].hpMap[JB].p;
      b = basis->chanModMaps[ichan].hpMap[JB].q;
      hash_hp[ichan][(struct DoubleBundle){j,b}] = JB;
    }
  }


  size_t hppSum = 0;
  #pragma omp parallel for schedule(dynamic)
  for(std::size_t j = 0; j < basis->nParticles; j++){
    std::size_t a,b,i;
    std::size_t hppDim;
    hppDim = basis->threeBodyChanDims[j].hppDim;
    hppSum += hppDim;
    for(std::size_t iab = 0; iab < hppDim; iab++){
      i = basis->threeBodyChanMaps[j].hppMap[iab].p;
      a = basis->threeBodyChanMaps[j].hppMap[iab].q;
      b = basis->threeBodyChanMaps[j].hppMap[iab].r;
      hash_hpp[j][(struct TripleBundle){i,a,b}] = iab;
    }
  }

  #pragma omp parallel for schedule(dynamic)
  for(std::size_t a = basis->nParticles; a < basis->nSpstates; a++){
    std::size_t b,i,j;
    std::size_t phhDim;
    phhDim = basis->threeBodyChanDims[a].phhDim;
    for(std::size_t bij = 0; bij < phhDim; bij++){
      b = basis->threeBodyChanMaps[a].phhMap[bij].p;
      i = basis->threeBodyChanMaps[a].phhMap[bij].q;
      j = basis->threeBodyChanMaps[a].phhMap[bij].r;
      hash_phh[a - basis->nParticles][(struct TripleBundle){b,i,j}] = bij;
    }
  }

  #pragma omp parallel for schedule(dynamic)
  for(std::size_t ichan = startChannel; ichan < endChannel; ichan++){
    for(std::size_t AB = 0; AB < basis->chanDims[ichan].ppDim; AB++){
      for(std::size_t IJ = 0; IJ < basis->chanDims[ichan].hhDim; IJ++){
        size_t spChan;
        size_t modChan;
        size_t ROW;
        size_t COL;
        std::size_t a,b,i,j;
        a = basis->chanMaps[ichan].ppMap[AB].p;
        b = basis->chanMaps[ichan].ppMap[AB].q;
        i = basis->chanMaps[ichan].hhMap[IJ].p;
        j = basis->chanMaps[ichan].hhMap[IJ].q;

        spChan = j;
        ROW = hash_hpp[j][(struct TripleBundle){i,a,b}];
        COL = 0;
        map_pphh_iab_j_hpp_h.set(ichan, AB, IJ, (ChainIndex){spChan,ROW,COL});

        spChan = i;
        ROW = hash_hpp[i][(struct TripleBundle){j,a,b}];
        COL = 0;
        map_pphh_jab_i_hpp_h.set(ichan, AB, IJ, (ChainIndex){spChan,ROW,COL});

        spChan = a - basis->nParticles;
        ROW = 0;
        COL = hash_phh[spChan][(struct TripleBundle){b,i,j}];
        map_pphh_a_bij_p_phh.set(ichan, AB, IJ, (ChainIndex){spChan,ROW,COL});

        spChan = b - basis->nParticles;
        ROW = 0;
        COL = hash_phh[spChan][(struct TripleBundle){a,i,j}];
        map_pphh_b_aij_p_phh.set(ichan, AB, IJ, (ChainIndex){spChan,ROW,COL});



        modChan = basis->TBmodChanIndexFunction(a,i);
        ROW = hash_ph[modChan][(struct DoubleBundle){a,i}];
        COL = hash_hp[modChan][(struct DoubleBundle){j,b}];
        map_pphh_ai_jb_phhp.set(ichan, AB, IJ, (ChainIndex){modChan,ROW,COL});

        modChan = basis->TBmodChanIndexFunction(a,j);
        ROW = hash_ph[modChan][(struct DoubleBundle){a,j}];
        COL = hash_hp[modChan][(struct DoubleBundle){i,b}];
        map_pphh_aj_ib_phhp.set(ichan, AB, IJ, (ChainIndex){modChan,ROW,COL});

        modChan = basis->TBmodChanIndexFunction(b,i);
        ROW = hash_ph[modChan][(struct DoubleBundle){b,i}];
        COL = hash_hp[modChan][(struct DoubleBundle){j,a}];
        map_pphh_bi_ja_phhp.set(ichan, AB, IJ, (ChainIndex){modChan,ROW,COL});

        modChan = basis->TBmodChanIndexFunction(b,j);
        ROW = hash_ph[modChan][(struct DoubleBundle){b,j}];
        COL = hash_hp[modChan][(struct DoubleBundle){i,a}];
        map_pphh_bj_ia_phhp.set(ichan, AB, IJ, (ChainIndex){modChan,ROW,COL});
      }
    }
  }

  delete [] hash_hp;
  delete [] hash_ph;
  delete [] hash_phh;
  delete [] hash_hpp;
} // end loadT2Maps
