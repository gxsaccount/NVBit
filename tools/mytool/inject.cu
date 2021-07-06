#include <cstdint>
#include <cstdio>

#include "nvbit_reg_rw.h"
#include "utils/utils.h"

extern "C" __device__ __noinline__ void instrument_enter(int32_t sm) {
  // if (get_smid() != sm) {
  //   asm("exit;");
  // }
  printf("enter thread (smid=%d, warpid=%d, lineid=%d)\n", get_smid(), 
         get_warpid(), get_laneid()); 
}

extern "C" __device__ __noinline__ void instrument_exit(int32_t pred) {
  // printf("exit thread (smid=%d, warpid=%d, lineid=%d)\n", get_smid(), 
  //        get_warpid(), get_laneid());
  if (!pred) {
    return;
  }
  asm("exit;");
}

extern "C" __device__ __noinline__ void instrument_s2r(int32_t pred,
                                                       int32_t val,
                                                       int32_t nr) {
  if (!pred) {
    return;
  }
  nvbit_write_reg(nr, val);
}
