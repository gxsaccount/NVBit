
#include <stdint.h>
#include <stdio.h>
#include "nvbit_reg_rw.h"
#include "utils/utils.h"

extern "C" __device__ __noinline__ void count_instrs(int predicate,
                                                     int count_warp_level,
                                                     uint64_t pcounter) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot_sync(__activemask(), 1);

    /* compute the predicate mask */
    const int predicate_mask = __ballot_sync(__activemask(), predicate);

    /* each thread will get a lane id (get_lane_id is implemented in
     * utils/utils.h) */
    const int laneid = get_laneid();

    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;

    /* count all the active thread */
    const int num_threads = __popc(predicate_mask);

    /* only the first active thread will perform the atomic */
    if (first_laneid == laneid) {
        if (count_warp_level) {
            /* num threads can be zero when accounting for predicates off */
            if (num_threads > 0) {
                atomicAdd((unsigned long long*)pcounter, 1);
            }
        } else {
            atomicAdd((unsigned long long*)pcounter, num_threads);
        }
    }
}

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

// extern "C" __device__ __noinline__ void instrument_s2r(int32_t pred,
//                                                        int32_t val,
//                                                        int32_t nr) {
//   if (!pred) {
//     return;
//   }
//   nvbit_write_reg(nr, val);
// }
