#ifndef POSTPROCESSING_HPP
#define POSTPROCESSING_HPP

#include "udf.hpp"
#include "nrs.hpp"

namespace postProcessing {
  
  inline void strainRate(nrs_t *nrs, bool useMeshVelocity, occa::memory o_Sij) {
      // 1. We call the new native NekRS v26 function.
      // The parameter 'false' disables smoothing (to keep strict gradients at the wall).
      occa::memory o_Sij_real = nrs->strainRate(false);
      
      // 2. Copy the real mathematical result into your pointer o_Sij
      // so that cf.okl can use it and compute Cf.
      o_Sij_real.copyTo(o_Sij);
  }

}

#endif