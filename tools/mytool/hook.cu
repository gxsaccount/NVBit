#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <map>
#include <string>
#include <unordered_set>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* opcode to id map */
std::map<std::string, int> opcode_to_id_map;

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction f) {
  /* "recording" function was instrumented, if set insertion failed
   * we have already encountered this function */
  if (!already_instrumented.insert(f).second) {
    return;
  }

  const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
  printf("Inspecting function %s at address 0x%lx\n",
         nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f));

  for (auto ins : instrs) {
    ins->print("\t");
    if (strcmp(ins->getOpcode(), "EXIT") == 0) {
      nvbit_insert_call(ins, "instrument_exit", IPOINT_BEFORE);
      nvbit_add_call_arg_guard_pred_val(ins);
    } else if (strcmp(ins->getOpcode(), "S2R") == 0) {
      std::string src = ins->getOperand(1)->u.generic.array;
      assert(ins->getOperand(0)->type == InstrType::OperandType::REG);
      int nr = ins->getOperand(0)->u.reg.num;
      if (src == "SR_CTAID.X") {
        nvbit_insert_call(ins, "instrument_s2r", IPOINT_BEFORE);
        nvbit_remove_orig(ins);
        nvbit_add_call_arg_guard_pred_val(ins);
        nvbit_add_call_arg_launch_val32(ins, 4);
        nvbit_add_call_arg_const_val32(ins, nr);
      } else if (src == "SR_CTAID.Y") {
        nvbit_insert_call(ins, "instrument_s2r", IPOINT_BEFORE);
        nvbit_remove_orig(ins);
        nvbit_add_call_arg_guard_pred_val(ins);
        nvbit_add_call_arg_launch_val32(ins, 8);
        nvbit_add_call_arg_const_val32(ins, nr);
      } else if (src == "SR_CTAID.Z") {
        nvbit_insert_call(ins, "instrument_s2r", IPOINT_BEFORE);
        nvbit_remove_orig(ins);
        nvbit_add_call_arg_guard_pred_val(ins);
        nvbit_add_call_arg_launch_val32(ins, 12);
        nvbit_add_call_arg_const_val32(ins, nr);
      } else if (src == "SR_NCTAID.X") {
        nvbit_insert_call(ins, "instrument_s2r", IPOINT_BEFORE);
        nvbit_remove_orig(ins);
        nvbit_add_call_arg_guard_pred_val(ins);
        nvbit_add_call_arg_launch_val32(ins, 16);
        nvbit_add_call_arg_const_val32(ins, nr);
      } else if (src == "SR_NCTAID.Y") {
        nvbit_insert_call(ins, "instrument_s2r", IPOINT_BEFORE);
        nvbit_remove_orig(ins);
        nvbit_add_call_arg_guard_pred_val(ins);
        nvbit_add_call_arg_launch_val32(ins, 20);
        nvbit_add_call_arg_const_val32(ins, nr);
      } else if (src == "SR_NCTAID.Z") {
        nvbit_insert_call(ins, "instrument_s2r", IPOINT_BEFORE);
        nvbit_remove_orig(ins);
        nvbit_add_call_arg_guard_pred_val(ins);
        nvbit_add_call_arg_launch_val32(ins, 24);
        nvbit_add_call_arg_const_val32(ins, nr);
      }
    }
  }
  nvbit_insert_call(instrs[0], "instrument_enter", IPOINT_BEFORE);
  nvbit_add_call_arg_launch_val32(instrs[0], 0);
}

/* This call-back is triggered every time a CUDA driver call is encountered.
 * Here we can look for a particular CUDA driver call by checking at the
 * call back ids  which are defined in tools_cuda_api_meta.h.
 * This call back is triggered bith at entry and at exit of each CUDA driver
 * call, is_exit=0 is entry, is_exit=1 is exit.
 * */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
  static thread_local int3 grid;
  static thread_local int3 total;
  if (cbid == API_CUDA_cuLaunchKernel) {
    auto p = (cuLaunchKernel_params_st *)params;

    if (!is_exit) {
      if (p->gridDimX * p->gridDimY * p->gridDimZ > 1) {
        total.x = p->gridDimX;
        total.y = p->gridDimY;
        total.z = p->gridDimZ;
        for (int x = 0; x < p->gridDimX; x++)
          for (int y = 0; y < p->gridDimY; y++)
            for (int z = 0; z < p->gridDimY; z++) {
              grid.x = x;
              grid.y = y;
              grid.z = z;
              if (x == p->gridDimX - 1 && y == p->gridDimY - 1 &&
                  z == p->gridDimZ - 1) {
                continue;
              } else {
                auto r = cuLaunchKernel(
                    p->f, 1, 1, 1, p->blockDimX, p->blockDimY, p->blockDimZ,
                    p->sharedMemBytes, p->hStream, p->kernelParams, p->extra);
              }
            }
      } else {
        p->gridDimX = 1;
        p->gridDimY = 1;
        p->gridDimZ = 1;
      }

      instrument_function_if_needed(ctx, p->f);
      int32_t smid[7] = {1, grid.x, grid.y, grid.z, total.x, total.y, total.z};
      nvbit_set_at_launch(ctx, p->f, &smid[0], 7 * sizeof(int32_t));
      nvbit_enable_instrumented(ctx, p->f, true);
    }
  }
}
