#ifndef calc_hpp_
#define calc_hpp_

#include <cstddef>
#include "Chain.hpp"
#include "SPBasis.hpp"

/**
 * All of the pieces for a complete CCD step are
 * included in the file. The form of the equations
 * are in my second committee meeting slides. The
 * modified channels part is in my part of Ch.8
 * of the nuclear structure book. All of the intermediates
 * are only calculated block by block, and are stored in
 * an intermediate memory space.
 */

/**
 * Combine all of the T2_pphh information.
 * Requires all of the t2Diffs to be updated first.
 */
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
                   double            mixing);

/**
 * Calculate the t2 information from the non-permuted
 * C sum term. With the current distributed scheme,
 * every processor should calculate this piece entirely.
 * @param t2Diff_p2_p1hh is the output.
 */
void calcT2Diff_p2_p1hh(std::size_t    startChannel,
                        std::size_t    endChannel,
                        Chain<double> *vnn_phh_p,
                        Chain<double> &t2Old_p1_p2hh,
                        Chain<double> &t2Old_p2_p1hh,
                        Chain<double> &t2Diff_p2_p1hh,
                        SPBasis       *basis);

/**
 * Calculate the t2 information from the (i<->j) permuted
 * K sum term. With the current distributed scheme, every
 * processor should calculate this piece entirely.
 * @param t2Diff_h2pp_h1 is the output.
 */
void calcT2Diff_h2pp_h1(std::size_t   startChannel,
                        std::size_t   endChannel,
                        Chain<double> *vnn_h_hpp,
                        Chain<double> &t2Old_h2pp_h1,
                        Chain<double> &t2Old_h1pp_h2,
                        Chain<double> &t2Diff_h2pp_h1,
                        SPBasis       *basis);

/**
 * Calculate the t2 information from the CD and KL
 * sum terms. Only calculating myRank's blocks. These
 * are the blocks with "normal" symmetry.
 * @param t2Diff_pphh is the output.
 */
void calcT2Diff_pphh(std::size_t    startChannel,
                     std::size_t    endChannel,
                     Chain<double> *vnn_hhhh,
                     Chain<double> *vnn_hhpp,
                     Chain<double> *vnn_pppp,
                     Chain<double> &t2Old_pphh,
                     Chain<double> &t2Diff_pphh,
                     SPBasis       *basis);

/**
 * Calculate the t2 information from the non-permuted
 * KC sum term. Only calculating myRank's blocks. These
 * are the blocks with "modified" symmetry.
 * @param t2Diff_phhp is the output.
 */
void calcT2Diff_phhp(std::size_t    startChannel,
                     std::size_t    endChannel,
                     Chain<double> *vnn_hhpp_hpph_mod,
                     Chain<double> *vnn_hpph_hphp_mod,
                     Chain<double> &t2Old_phhp,
                     Chain<double> &t2Diff_phhp,
                     SPBasis       *basis);

/**
 * Calculate the CCD correlation energy.
 */
double calcFreeCorr(std::size_t    startChannel,
                    std::size_t    endChannel,
                    Chain<double> &t2_pphh,
                    SPBasis       *basis);

/**
 * The rest of these functions do the same as above
 * but now without needing the vnn chains as input.
 * They calculate each block of vnn as needed on the
 * fly.
 */

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
                       double            mixing);

void calcFreeT2Diff_h2pp_h1(std::size_t    startChannel,
                            std::size_t    endChannel,
                            Chain<double> &t2Old_h2pp_h1,
                            Chain<double> &t2Old_h1pp_h2,
                            Chain<double> &t2Diff_h2pp_h1,
                            SPBasis       *basis);

void calcFreeT2Diff_p2_p1hh(std::size_t    startChannel,
                            std::size_t    endChannel,
                            Chain<double> &t2Old_p1_p2hh,
                            Chain<double> &t2Old_p2_p1hh,
                            Chain<double> &t2Diff_p2_p1hh,
                            SPBasis       *basis);

void calcFreeT2Diff_pphh(std::size_t    startChannel,
                         std::size_t    endChannel,
                         Chain<double> &t2Old_pphh,
                         Chain<double> &t2Diff_pphh,
                         SPBasis       *basis);

void calcFreeT2Diff_phhp(std::size_t    startChannel,
                         std::size_t    endChannel,
                         Chain<double> &t2Old_phhp,
                         Chain<double> &t2Diff_phhp,
                         SPBasis       *basis);

#endif
