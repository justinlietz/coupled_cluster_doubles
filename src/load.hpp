#ifndef load_hpp_
#define load_hpp_

#include "SPBasis.hpp"
#include "Chain.hpp"

void loadVnn_hhhh(std::size_t    startChannel,
                  std::size_t    endChannel,
                  Chain<double> *vnn_hhhh,
                  SPBasis       *basis);

void loadVnn_hhhp(std::size_t    startChannel,
                  std::size_t    endChannel,
                  Chain<double> *vnn_hhhp,
                  SPBasis       *basis);

void loadVnn_hhpp(std::size_t    startChannel,
                  std::size_t    endChannel,
                  Chain<double> *vnn_hhpp,
                  SPBasis       *basis);

void loadVnn_hpph(std::size_t    startChannel,
                  std::size_t    endChannel,
                  Chain<double> *vnn_hpph,
                  SPBasis       *basis);

void loadVnn_hppp(std::size_t    startChannel,
                  std::size_t    endChannel,
                  Chain<double> *vnn_hppp,
                  SPBasis       *basis);

void loadVnn_pppp(std::size_t    startChannel,
                  std::size_t    endChannel,
                  Chain<double> *vnn_pppp,
                  SPBasis       *basis);

void loadVnn_phh_p(std::size_t    startChannel,
                   std::size_t    endChannel,
                   Chain<double> *vnn_phh_p,
                   SPBasis       *basis);

void loadVnn_h_hpp(std::size_t    startChannel,
                   std::size_t    endChannel,
                   Chain<double> *vnn_h_hpp,
                   SPBasis       *basis);

void loadVnn_hhpp_hpph_mod(std::size_t    startChannel,
                           std::size_t    endChannel,
                           Chain<double> *vnn_hhpp_hpph_mod,
                           SPBasis       *basis);

void loadVnn_hpph_hphp_mod(std::size_t    startChannel,
                           std::size_t    endChannel,
                           Chain<double> *vnn_hpph_hphp_mod,
                           SPBasis       *basis);

void loadFreeVnn_hhhh(std::size_t    channel,
                      double        *vnn_hhhh,
                      SPBasis       *basis);

void loadFreeVnn_hhhp(std::size_t    channel,
                      double        *vnn_hhhp,
                      SPBasis       *basis);

void loadFreeVnn_hhpp(std::size_t    channel,
                      double        *vnn_hhpp,
                      SPBasis       *basis);

void loadFreeVnn_hpph(std::size_t    channel,
                      double        *vnn_hpph,
                      SPBasis       *basis);

void loadFreeVnn_hppp(std::size_t    channel,
                      double        *vnn_hppp,
                      SPBasis       *basis);

void loadFreeVnn_pppp(std::size_t    channel,
                      double        *vnn_pppp,
                      SPBasis       *basis);

void loadFreeVnn_phh_p(std::size_t    channel,
                       double        *vnn_phh_p,
                       SPBasis       *basis);

void loadFreeVnn_h_hpp(std::size_t    channel,
                       double        *vnn_h_hpp,
                       SPBasis       *basis);

void loadFreeVnn_hhpp_hpph_mod(std::size_t    channel,
                               double        *vnn_hhpp_hpph_mod,
                               SPBasis       *basis);

void loadFreeVnn_hpph_hphp_mod(std::size_t    channel,
                               double        *vnn_hpph_hphp_mod,
                               SPBasis       *basis);

void loadMaps(std::size_t startChannel,
              std::size_t endChannel,
              Chain<ChainIndex> &map_pphh_iab_j_hpp_h,
              Chain<ChainIndex> &map_pphh_jab_i_hpp_h,
              Chain<ChainIndex> &map_pphh_a_bij_hpp_h,
              Chain<ChainIndex> &map_pphh_b_aij_hpp_h,
              Chain<ChainIndex> &map_pphh_ai_jb_phhp,
              Chain<ChainIndex> &map_pphh_aj_ib_phhp,
              Chain<ChainIndex> &map_pphh_bi_ja_phhp,
              Chain<ChainIndex> &map_pphh_bj_ia_phhp,
              SPBasis *basis);
#endif
