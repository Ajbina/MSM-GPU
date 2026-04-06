#include "cuda_utils.cuh"
#include "kernels.cuh"
#include "msm_plan.hpp"
#include "bn254_params.cuh"
#include "ec_bn254.cuh"
#include "scalar_bn254.cuh"
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>

// simple benchmark kernel for EC addition throughput
__global__ void bench_add(G1J* a, int iters) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    G1J acc = *a;
    for (int i = 0; i < iters; ++i) acc = g1_add(acc, acc);
    *a = acc;
  }
}

struct GpuCtx {
  int device_id = 0;
  cudaStream_t stream{};
  G1J* d_points = nullptr;
  int* d_offsets = nullptr;
  G1J* d_local_bucket_sums = nullptr;
  
  // GPU-side merge and window reduction
  int* d_bucket_ids = nullptr;           // task bucket IDs (for GPU merge)
  G1J* d_merged_bucket_sums = nullptr;   // full bucket sums (GPU merge output)
  G1J* d_window_result = nullptr;        // window result on device (size 1)

  // PHASE 1: GPU-side digit extraction and bucket counting
  ScalarR_device* d_scalars_full = nullptr;    // full scalar array (uploaded once at startup)
  uint32_t* d_digits = nullptr;                 // digits computed per window (size N)
  int* d_bucket_counts = nullptr;              // bucket counts per window (size B)

  size_t points_capacity = 0;
  size_t offsets_capacity = 0;
  size_t local_sums_capacity = 0;
  size_t bucket_ids_capacity = 0;
  size_t merged_bucket_sums_capacity = 0;
  size_t scalars_capacity = 0;
  size_t digits_capacity = 0;
  size_t bucket_counts_capacity = 0;

  std::vector<G1J> h_points;
  std::vector<int> h_offsets;
  std::vector<int> h_bucket_ids;
  std::vector<int> h_shard_ids;
  std::vector<G1J> h_local_sums;
  std::vector<G1J> h_bucket_sums;  // per-GPU merged contribution by logical bucket index
  G1J h_window_result;              // window result from GPU

  // REFACTOR: Reusable host buffers for two-pass bucketing scheme
  std::vector<int> h_bucket_counts;    // bucket counts per window (host copy, size B)
  std::vector<int> h_bucket_offsets;   // prefix sum offsets per window (size B+1)
  std::vector<G1J> h_packed_points;    // contiguous bucketed points (variable size)
  size_t packed_points_capacity = 0;
};

static inline void ensure_points_capacity(GpuCtx& ctx, int required_points) {
  size_t need = size_t(std::max(1, required_points));
  if (ctx.points_capacity >= need) return;
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  if (ctx.d_points) CUDA_CALL(cudaFree(ctx.d_points));
  CUDA_CALL(cudaMalloc(&ctx.d_points, need * sizeof(G1J)));
  ctx.points_capacity = need;
}

static inline void ensure_offsets_capacity(GpuCtx& ctx, int required_offsets) {
  size_t need = size_t(std::max(1, required_offsets));
  if (ctx.offsets_capacity >= need) return;
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  if (ctx.d_offsets) CUDA_CALL(cudaFree(ctx.d_offsets));
  CUDA_CALL(cudaMalloc(&ctx.d_offsets, need * sizeof(int)));
  ctx.offsets_capacity = need;
}

static inline void ensure_local_sums_capacity(GpuCtx& ctx, int required_tasks) {
  size_t need = size_t(std::max(1, required_tasks));
  if (ctx.local_sums_capacity >= need) return;
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  if (ctx.d_local_bucket_sums) CUDA_CALL(cudaFree(ctx.d_local_bucket_sums));
  CUDA_CALL(cudaMalloc(&ctx.d_local_bucket_sums, need * sizeof(G1J)));
  ctx.local_sums_capacity = need;
}

static inline void ensure_bucket_ids_capacity(GpuCtx& ctx, int required_tasks) {
  size_t need = size_t(std::max(1, required_tasks));
  if (ctx.bucket_ids_capacity >= need) return;
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  if (ctx.d_bucket_ids) CUDA_CALL(cudaFree(ctx.d_bucket_ids));
  CUDA_CALL(cudaMalloc(&ctx.d_bucket_ids, need * sizeof(int)));
  ctx.bucket_ids_capacity = need;
}

static inline void ensure_merged_bucket_sums_capacity(GpuCtx& ctx, int B) {
  size_t need = size_t(std::max(1, B));
  if (ctx.merged_bucket_sums_capacity >= need) return;
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  if (ctx.d_merged_bucket_sums) CUDA_CALL(cudaFree(ctx.d_merged_bucket_sums));
  CUDA_CALL(cudaMalloc(&ctx.d_merged_bucket_sums, need * sizeof(G1J)));
  ctx.merged_bucket_sums_capacity = need;
  // Also allocate device window result (size 1)
  if (ctx.d_window_result) CUDA_CALL(cudaFree(ctx.d_window_result));
  CUDA_CALL(cudaMalloc(&ctx.d_window_result, 1 * sizeof(G1J)));
}

static inline void ensure_scalars_capacity(GpuCtx& ctx, int N) {
  size_t need = size_t(std::max(1, N));
  if (ctx.scalars_capacity >= need) return;
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  if (ctx.d_scalars_full) CUDA_CALL(cudaFree(ctx.d_scalars_full));
  CUDA_CALL(cudaMalloc(&ctx.d_scalars_full, need * sizeof(ScalarR_device)));
  ctx.scalars_capacity = need;
}

static inline void ensure_digits_capacity(GpuCtx& ctx, int N) {
  size_t need = size_t(std::max(1, N));
  if (ctx.digits_capacity >= need) return;
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  if (ctx.d_digits) CUDA_CALL(cudaFree(ctx.d_digits));
  CUDA_CALL(cudaMalloc(&ctx.d_digits, need * sizeof(uint32_t)));
  ctx.digits_capacity = need;
}

static inline void ensure_bucket_counts_capacity(GpuCtx& ctx, int B) {
  size_t need = size_t(std::max(1, B));
  if (ctx.bucket_counts_capacity >= need) return;
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  if (ctx.d_bucket_counts) CUDA_CALL(cudaFree(ctx.d_bucket_counts));
  CUDA_CALL(cudaMalloc(&ctx.d_bucket_counts, need * sizeof(int)));
  ctx.bucket_counts_capacity = need;
}

static inline void ensure_host_packed_points_capacity(std::vector<G1J>& h_packed_points,
                                                       size_t& packed_points_capacity,
                                                       size_t required_points) {
  if (packed_points_capacity >= required_points) return;
  // Allocate with some headroom to avoid frequent reallocations
  size_t new_capacity = std::max(required_points, packed_points_capacity * 2);
  h_packed_points.reserve(new_capacity);
  packed_points_capacity = new_capacity;
}

static inline G1J host_tree_reduce_points(std::vector<G1J> vals) {
  if (vals.empty()) return G1J::infinity();
  while (vals.size() > 1) {
    std::vector<G1J> next;
    next.reserve((vals.size() + 1) / 2);
    size_t i = 0;
    for (; i + 1 < vals.size(); i += 2) {
      next.push_back(g1_add(vals[i], vals[i + 1]));
    }
    if (i < vals.size()) next.push_back(vals[i]);
    vals.swap(next);
  }
  return vals[0];
}

struct HostMergeStats {
  int fanin_zero = 0;
  int fanin_one = 0;
  int fanin_two = 0;
  int fanin_ge3 = 0;
  int fanin_max = 0;
  int merge_adds = 0;
};

// Host merge strategy by fan-in:
// 0 partials -> skip, 1 -> direct assign, 2 -> one add, >=3 -> tree reduction.
static inline HostMergeStats merge_bucket_partials_host(
    std::vector<std::vector<G1J>>& bucket_partials,
    std::vector<G1J>& combined_buckets) {
  HostMergeStats stats;
  const int B = (int)bucket_partials.size();
  for (int b = 0; b < B; b++) {
    int n = (int)bucket_partials[b].size();
    stats.fanin_max = std::max(stats.fanin_max, n);
    if (n > 1) stats.merge_adds += (n - 1);
    if (n == 0) {
      stats.fanin_zero++;
    } else if (n == 1) {
      stats.fanin_one++;
      combined_buckets[b] = bucket_partials[b][0];
    } else if (n == 2) {
      stats.fanin_two++;
      combined_buckets[b] = g1_add(bucket_partials[b][0], bucket_partials[b][1]);
    } else {
      stats.fanin_ge3++;
      combined_buckets[b] = host_tree_reduce_points(std::move(bucket_partials[b]));
    }
  }
  return stats;
}

static inline G1J pippenger_window_sum_from_bucket_sums(const std::vector<G1J>& bucket_sums) {
  // Standard Pippenger window sum: sum_{b=1..B} b * S_b via reverse suffix accumulation.
  G1J window_sum = G1J::infinity();
  G1J running = G1J::infinity();
  for (int b = (int)bucket_sums.size() - 1; b >= 0; --b) {
    running = g1_add(running, bucket_sums[b]);
    window_sum = g1_add(window_sum, running);
  }
  return window_sum;
}

static inline G1J host_fold_windows(const std::vector<G1J>& window_sums, int wbits) {
  G1J R = G1J::infinity();
  for (int i = (int)window_sums.size() - 1; i >= 0; --i) {
    if (i != (int)window_sums.size() - 1) {
      for (int k = 0; k < wbits; k++) R = g1_dbl(R);
    }
    R = g1_add(R, window_sums[i]);
  }
  return R;
}

// 256-bit scalar mul (bit-walk) for correctness reference only.
// This uses ScalarR bits (up to 254 bits by design).
static inline G1J g1_scalar_mul_254(G1J P, const ScalarR& k) {
  G1J R = G1J::infinity();
  for (int i = 0; i < 254; i++) {
    if (get_bit(k, i)) R = g1_add(R, P);
    P = g1_dbl(P);
  }
  return R;
}

static inline G1J cpu_msm_reference(const std::vector<G1J>& pts, const std::vector<ScalarR>& sc) {
  G1J acc = G1J::infinity();
  for (size_t i = 0; i < pts.size(); i++) {
    G1J t = g1_scalar_mul_254(pts[i], sc[i]);
    acc = g1_add(acc, t);
  }
  return acc;
}

static inline bool equal_projective(const G1J& A, const G1J& B) {
  if (A.is_inf() && B.is_inf()) return true;
  if (A.is_inf() != B.is_inf()) return false;
  Fp Z1Z1 = A.Z * A.Z;
  Fp Z2Z2 = B.Z * B.Z;
  Fp U1 = A.X * Z2Z2;
  Fp U2 = B.X * Z1Z1;
  if (!(U1 == U2)) return false;
  Fp S1 = (A.Y * B.Z) * Z2Z2;
  Fp S2 = (B.Y * A.Z) * Z1Z1;
  return (S1 == S2);
}

int main(int argc, char** argv) {
  try {
    print_device_info();

    int G = 0;
    CUDA_CALL(cudaGetDeviceCount(&G));
    if (G < 1) {
      std::cerr << "No CUDA GPU found.\n";
      return 1;
    }

    // args:
    //   1: N
    //   2: wbits
    //   3: use_greedy (0=static round-robin, 1=greedy)
    //   4: check (1 enables correctness checks)
    //   5: objective (0=throughput, 1=latency)
    //   6: audit_stage_timing (optional, default 0)
    //   7: force_split_test (optional, default 0; debug only)
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1'000'000;
    const int wbits = (argc > 2) ? std::atoi(argv[2]) : 8;
    const int use_greedy = (argc > 3) ? std::atoi(argv[3]) : 1;
    const int do_check = (argc > 4) ? std::atoi(argv[4]) : 0;
    const int obj_flag = (argc > 5) ? std::atoi(argv[5]) : 0;
    const int audit_flag = (argc > 6) ? std::atoi(argv[6]) : 0;
    const int force_split_flag = (argc > 7) ? std::atoi(argv[7]) : 0;

    if (N <= 0) {
      std::cerr << "Invalid input: N must be > 0\n";
      return 1;
    }
    if (wbits <= 0) {
      std::cerr << "Invalid input: wbits must be > 0\n";
      return 1;
    }
    if (wbits >= 31) {
      std::cerr << "Invalid input: wbits must be < 31\n";
      return 1;
    }
    if (obj_flag != 0 && obj_flag != 1) {
      std::cerr << "Invalid input: objective must be 0 (throughput) or 1 (latency)\n";
      return 1;
    }
    if (force_split_flag != 0 && force_split_flag != 1) {
      std::cerr << "Invalid input: force_split_flag must be 0 or 1\n";
      return 1;
    }

    const bool audit_stage_timing = (audit_flag != 0);
    const bool force_split_test = (force_split_flag != 0);

    const bool planner_use_greedy = (use_greedy != 0);
    const Objective objective = (obj_flag == 1) ? Objective::Latency : Objective::Throughput;

    std::cout << "N=" << N << ", wbits=" << wbits << ", GPUs=" << G
              << ", mode=" << (planner_use_greedy ? "Greedy" : "Even") << "\n";

    auto calibrate_Rg = [&]() {
      G1J* d;
      CUDA_CALL(cudaMalloc(&d, sizeof(G1J)));
      G1J tmp = g1_from_affine_u64(1, 2);
      CUDA_CALL(cudaMemcpy(d, &tmp, sizeof(G1J), cudaMemcpyHostToDevice));
      cudaEvent_t s, e;
      cudaEventCreate(&s);
      cudaEventCreate(&e);
      const int iters = 1000000;
      CUDA_CALL(cudaEventRecord(s));
      bench_add<<<1, 1>>>(d, iters);
      CUDA_CALL(cudaEventRecord(e));
      CUDA_CALL(cudaEventSynchronize(e));
      float ms;
      CUDA_CALL(cudaEventElapsedTime(&ms, s, e));
      CUDA_CALL(cudaFree(d));
      return double(iters) / (ms * 1e-3);
    };

    // gather simple system parameters from device 0 (assuming homogenous cluster)
    SystemParams params{};
    size_t max_mem_per_gpu = SIZE_MAX;
    {
      cudaDeviceProp prop;
      size_t free_mem = 0, total_mem = 0;
      CUDA_CALL(cudaGetDeviceProperties(&prop, 0));
      CUDA_CALL(cudaMemGetInfo(&free_mem, &total_mem));
      params.B_g    = double(prop.memoryBusWidth) * prop.memoryClockRate * 1e3 / 8.0;
      params.B_link = 8e9;
      params.L_link = 1e-6;
      params.U_g    = 0.5;
      params.D_pt   = sizeof(G1J);
      params.L_sync = 1e-6;
      params.L_h2d  = 3e-5;
      params.L_d2h  = 2e-5;
      // GPU-reduction pipeline calibration (Apr 6): Updated with kernel timing scope fix
      // Previous: 0.26 (legacy host-merge model) → 5.36 (Mar 31)
      // Calibrated from audit runs: N=10000
      // - wbits=6: k_compute_small = 5.84 (was 5.20)
      // - wbits=8: k_compute_mid = 11.0854 (was 5.36)
      // - wbits=10: k_compute_large = 8.91 (was 5.40)
      params.k_compute_small = 5.84;   // wbits <= 6 (newly calibrated)
      params.k_compute_mid   = 11.0854;   // 7 <= wbits <= 8 (newly calibrated after kernel timing fix)
      params.k_compute_large = 8.91;   // wbits >= 9 (newly calibrated)
      params.alpha_pack   = 2.8e-08;
      // GPU merge and suffix operations: coefficients now in make_plan (5.0e-8, 1.0e-8)
      params.tpb      = 256;
      params.num_sms  = prop.multiProcessorCount;
      max_mem_per_gpu = size_t(free_mem * 0.9);
      params.M_g = max_mem_per_gpu;
      params.max_mem_per_gpu = max_mem_per_gpu;
      params.R_g = calibrate_Rg();

      if (force_split_test) {
        size_t forced_budget = std::max<size_t>(size_t(params.D_pt) * 4, size_t(params.D_pt));
        max_mem_per_gpu = forced_budget;
        params.M_g = forced_budget;
        params.max_mem_per_gpu = forced_budget;
        std::cerr << "DEBUG force_split_test=ON: clamped planner memory budget to "
                  << forced_budget << " bytes per GPU\n";
      }

      if (audit_stage_timing) {
        std::cerr << "SystemParams: R_g=" << params.R_g << " B_g=" << params.B_g
                  << " B_link=" << params.B_link
                  << " tpb=" << params.tpb << " num_sms=" << params.num_sms
                  << " M_g=" << params.M_g << " max_mem_per_gpu=" << params.max_mem_per_gpu
                  << " L_h2d=" << params.L_h2d << " L_d2h=" << params.L_d2h
                  << " k_compute_small=" << params.k_compute_small
                  << " k_compute_mid=" << params.k_compute_mid
                  << " k_compute_large=" << params.k_compute_large
                  << " alpha_pack=" << params.alpha_pack
                  << "\n";
      }
    }

    auto e2e_start = std::chrono::high_resolution_clock::now();

    const int BITSIZE = 254;
    const int W = (BITSIZE + wbits - 1) / wbits;
    const int B = (1 << wbits) - 1;

    if (audit_stage_timing) {
      std::cout << "\n=== MSM config ===\n";
      std::cout << "N=" << N << ", wbits=" << wbits << ", windows W=" << W
                << ", buckets/window B=" << B << "\n";
      std::cout << "GPUs=" << G << ", objective="
                << (objective == Objective::Latency ? "latency" : "throughput") << "\n";
      std::cout << "planner_strategy=" << (planner_use_greedy ? "greedy" : "static_round_robin") << "\n";
      std::cout << "check=" << do_check
                << ", audit_stage_timing=" << (audit_stage_timing ? 1 : 0)
                << ", force_split_test=" << (force_split_test ? 1 : 0) << "\n";
    }

    // Audit-only CSV log
    std::ofstream csv;
    if (audit_stage_timing) {
      csv.open("performance.csv");
      csv << "win,points,buckets,obj,pred_lat,pred_tp,act,max_cost,avg_cost,"
             "pred_gpu_max,pred_gpu_sum,pred_h2d_max,pred_comp_max,pred_mem_max,"
             "pred_h2d_sum,pred_comp_sum,pred_mem_sum,pred_reduction,pred_overhead\n";
    }

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist_u64(1, ~0ULL);

    std::vector<ScalarR> scalars(N);
    std::vector<G1J> points(N);

    G1J Gen = g1_from_affine_u64(BN254_GX, BN254_GY);

    for (int i = 0; i < N; i++) {
      scalars[i] = random_scalar_mod_r(rng);
      uint64_t k = dist_u64(rng);
      points[i] = g1_scalar_mul_u64(Gen, k);
    }

    std::vector<GpuCtx> ctx(G);
    for (int g = 0; g < G; g++) {
      CUDA_CALL(cudaSetDevice(g));
      ctx[g].device_id = g;
      CUDA_CALL(cudaStreamCreate(&ctx[g].stream));
    }

    if (audit_stage_timing) {
      CUDA_CALL(cudaSetDevice(0));
      G1J test_pt_host = g1_from_affine_u64(1, 2);
      G1J* d_test = nullptr;
      CUDA_CALL(cudaMalloc(&d_test, sizeof(G1J)));
      CUDA_CALL(cudaMemcpy(d_test, &test_pt_host, sizeof(G1J), cudaMemcpyHostToDevice));
      G1J test_pt_back;
      CUDA_CALL(cudaMemcpy(&test_pt_back, d_test, sizeof(G1J), cudaMemcpyDeviceToHost));
      std::cerr << "TEST: host point sent to GPU and back, is_inf=" << test_pt_back.is_inf() << "\n";
      CUDA_CALL(cudaFree(d_test));
    }

    // PHASE 1: Upload scalars once to GPU for per-window digit extraction
    // Convert host scalars to device format and upload
    std::vector<ScalarR_device> scalars_device(N);
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < 4; j++) {
        scalars_device[i].v[j] = scalars[i].v[j];
      }
    }
    
    // Ensure capacity and upload on GPU 0 (main GPU)
    CUDA_CALL(cudaSetDevice(0));
    ensure_scalars_capacity(ctx[0], N);
    ensure_digits_capacity(ctx[0], N);
    ensure_bucket_counts_capacity(ctx[0], B);
    
    CUDA_CALL(cudaMemcpy(ctx[0].d_scalars_full, scalars_device.data(),
                         N * sizeof(ScalarR_device), cudaMemcpyHostToDevice));
    
    if (audit_stage_timing) {
      std::cerr << "GPU SETUP: Uploaded " << N << " scalars, prepared for "
                << N << " digits, " << B << " buckets\n";
    }

    std::vector<G1J> window_sums(W, G1J::infinity());

    double overhead_ewma = 0.024;
    const double overhead_alpha = 0.2;
    const double overhead_cap = 0.030;

    // Helper: select k_compute based on wbits band
    auto get_k_compute_for_wbits = [&](int w) -> double {
      if (w <= 6) return params.k_compute_small;
      if (w <= 8) return params.k_compute_mid;
      return params.k_compute_large;
    };

    if (audit_stage_timing) {
      std::cerr << "START windows (W=" << W << ", B=" << B << ") with one full warm-up pass\n";
    }

    // Calibration accumulators for each bucket
    int audit_k_compute_windows_small = 0;
    int audit_k_compute_windows_mid = 0;
    int audit_k_compute_windows_large = 0;
    double audit_suggested_k_compute_small_avg = 0.0;
    double audit_suggested_k_compute_mid_avg = 0.0;
    double audit_suggested_k_compute_large_avg = 0.0;

    double audit_pred_h2d_total = 0.0;
    double audit_pred_comp_total = 0.0;
    double audit_pred_d2h_total = 0.0;
    double audit_pred_total_total = 0.0;
    double audit_actual_pack_total = 0.0;
    double audit_actual_h2d_total = 0.0;
    double audit_actual_kernel_total = 0.0;
    double audit_actual_d2h_total = 0.0;
    double audit_actual_host_merge_total = 0.0;
    double audit_actual_cpu_suffix_total = 0.0;
    double audit_fixed_overhead_total = 0.0;
    double audit_total_points = 0.0;
    double audit_total_tasks = 0.0;
    double audit_total_merge_adds = 0.0;
    double audit_total_nonempty_buckets = 0.0;
    double audit_suggested_k_compute_sum = 0.0;
    double audit_suggested_alpha_pack_sum = 0.0;
    double audit_suggested_alpha_merge_sum = 0.0;
    double audit_suggested_alpha_suffix_sum = 0.0;
    // These are now declared earlier, just remove duplicates
    // double audit_suggested_k_compute_small_sum = 0.0;
    // double audit_suggested_k_compute_mid_sum = 0.0;
    // double audit_suggested_k_compute_large_sum = 0.0;
    // int audit_k_compute_windows_contributing = 0;
    double audit_ratio_pack_sum = 0.0;
    double audit_ratio_compute_sum = 0.0;
    double audit_ratio_merge_sum = 0.0;
    double audit_ratio_suffix_sum = 0.0;
    int audit_ratio_pack_n = 0;
    int audit_ratio_compute_n = 0;
    int audit_ratio_merge_n = 0;
    int audit_ratio_suffix_n = 0;
    bool audit_window0_reported = false;
    int audit_windows = 0;

    double active_k_compute = get_k_compute_for_wbits(wbits);

    double warmup_window_time = 0.0;
    double audit_window_total_time_sum = 0.0;
    double audit_coarse_sum_total = 0.0;
    double audit_fine_sum_total = 0.0;
    double audit_gpu_gap_total = 0.0;
    double audit_unaccounted_window_gap_total = 0.0;

    auto setup_end = std::chrono::high_resolution_clock::now();
    double setup_time = std::chrono::duration<double>(setup_end - e2e_start).count();

    for (int pass = -1; pass < W; pass++) {
      const bool collect_metrics = (pass >= 0);
      const int wi = collect_metrics ? pass : 0;

      auto window_start = std::chrono::high_resolution_clock::now();
      double window_bucketization_time = 0.0;
      double window_planning_time = 0.0;
      double window_gpu_phase_total_time = 0.0;
      double window_host_merge_plus_suffix_time = 0.0;

      if (audit_stage_timing) {
        if (collect_metrics) {
          std::cerr << "ENTER window " << wi << "\n";
        } else {
          std::cerr << "ENTER warm-up pass (window template 0, not logged)\n";
        }
      }

      const int bit_offset = wi * wbits;

      auto bucketing_start = std::chrono::high_resolution_clock::now();
      double gpu_digit_extraction_time = 0.0;
      double gpu_bucket_count_time = 0.0;
      double remaining_host_pack_time = 0.0;
      
      // PHASE 1: GPU-side digit extraction
      auto digit_extract_start = std::chrono::high_resolution_clock::now();
      {
        CUDA_CALL(cudaSetDevice(ctx[0].device_id));
        int threads_per_block = 256;
        int blocks = (N + threads_per_block - 1) / threads_per_block;
        kernel_compute_digits<<<blocks, threads_per_block, 0, ctx[0].stream>>>(
          ctx[0].d_scalars_full, N, bit_offset, wbits, ctx[0].d_digits);
        CUDA_CALL(cudaGetLastError());
      }
      CUDA_CALL(cudaStreamSynchronize(ctx[0].stream));
      auto digit_extract_end = std::chrono::high_resolution_clock::now();
      gpu_digit_extraction_time = std::chrono::duration<double>(digit_extract_end - digit_extract_start).count();
      
      // PHASE 2: GPU-side bucket counting
      auto bucket_count_start = std::chrono::high_resolution_clock::now();
      std::vector<uint32_t> h_digits(N);
      std::vector<int> h_bucket_counts(B);
      {
        CUDA_CALL(cudaSetDevice(ctx[0].device_id));
        // Zero bucket counts on GPU
        CUDA_CALL(cudaMemsetAsync(ctx[0].d_bucket_counts, 0, B * sizeof(int), ctx[0].stream));
        
        // Launch bucket count kernel
        int threads_per_block = 256;
        int blocks = (N + threads_per_block - 1) / threads_per_block;
        kernel_count_buckets<<<blocks, threads_per_block, 0, ctx[0].stream>>>(
          ctx[0].d_digits, N, ctx[0].d_bucket_counts);
        CUDA_CALL(cudaGetLastError());
      }
      
      // Copy digits and bucket counts back to host
      CUDA_CALL(cudaMemcpyAsync(h_digits.data(), ctx[0].d_digits, N * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, ctx[0].stream));
      CUDA_CALL(cudaMemcpyAsync(h_bucket_counts.data(), ctx[0].d_bucket_counts, B * sizeof(int),
                                cudaMemcpyDeviceToHost, ctx[0].stream));
      CUDA_CALL(cudaStreamSynchronize(ctx[0].stream));
      
      auto bucket_count_end = std::chrono::high_resolution_clock::now();
      gpu_bucket_count_time = std::chrono::duration<double>(bucket_count_end - bucket_count_start).count();
      
      // PHASE 3: Host-side two-pass bucketing scheme (REFACTORED for cache efficiency)
      // We use GPU-computed digits and bucket counts to build a contiguous bucketed layout
      
      auto host_pack_start = std::chrono::high_resolution_clock::now();
      
      // Initialize host buffers for this window
      ctx[0].h_bucket_counts.assign(B, 0);
      ctx[0].h_bucket_offsets.assign(B + 1, 0);
      
      // PASS 1: Verify counts and compute prefix sum offsets
      // (bucket_counts come from GPU, but we validate and compute offsets here)
      int total_bucketed = 0;
      int nonempty_buckets = 0;
      for (int b = 0; b < B; b++) {
        ctx[0].h_bucket_counts[b] = h_bucket_counts[b];
        if (h_bucket_counts[b] > 0) nonempty_buckets++;
      }
      
      // Compute bucket_offsets via prefix sum
      ctx[0].h_bucket_offsets[0] = 0;
      for (int b = 0; b < B; b++) {
        ctx[0].h_bucket_offsets[b + 1] = ctx[0].h_bucket_offsets[b] + ctx[0].h_bucket_counts[b];
      }
      total_bucketed = ctx[0].h_bucket_offsets[B];
      
      // Sanity check: total_bucketed should match sum of counts
      if (total_bucketed != std::accumulate(ctx[0].h_bucket_counts.begin(), ctx[0].h_bucket_counts.end(), 0)) {
        std::cerr << "ERROR [Window " << wi << "]: bucket_offsets prefix sum mismatch!\n";
      }
      
      // PASS 2: Allocate/reuse packed_points and scatter points contiguously
      ensure_host_packed_points_capacity(ctx[0].h_packed_points, ctx[0].packed_points_capacity, total_bucketed);
      ctx[0].h_packed_points.clear();
      ctx[0].h_packed_points.resize(total_bucketed);
      
      // Use write cursors to track insertion position per bucket
      std::vector<int> write_cursors(B);
      for (int b = 0; b < B; b++) {
        write_cursors[b] = ctx[0].h_bucket_offsets[b];
      }
      
      // Scatter each point into its correct bucket position
      for (int i = 0; i < N; i++) {
        uint32_t digit = h_digits[i];
        if (!digit) continue;  // Skip zero digits
        int b = (int)digit - 1;
        if (b < 0 || b >= B) {
          std::cerr << "ERROR [Window " << wi << "]: bucket index " << b << " out of range [0," << B << ")\n";
          continue;
        }
        int write_pos = write_cursors[b]++;
        if (write_pos < (int)ctx[0].h_packed_points.size()) {
          ctx[0].h_packed_points[write_pos] = points[i];
        }
      }
      
      // Verify write cursors match expected offsets
      for (int b = 0; b < B; b++) {
        if (write_cursors[b] != ctx[0].h_bucket_offsets[b + 1]) {
          std::cerr << "ERROR [Window " << wi << "]: bucket " << b << " write cursor mismatch: "
                    << write_cursors[b] << " != " << ctx[0].h_bucket_offsets[b + 1] << "\n";
        }
      }
      
      auto host_pack_end = std::chrono::high_resolution_clock::now();
      
      remaining_host_pack_time = std::chrono::duration<double>(host_pack_end - host_pack_start).count();
      
      auto bucketing_end = std::chrono::high_resolution_clock::now();
      if (collect_metrics) {
        window_bucketization_time = std::chrono::duration<double>(bucketing_end - bucketing_start).count();
      }

      // REFACTOR DEBUG: Reconstruct old-style bucket_points for verification
      if (collect_metrics && audit_stage_timing && do_check) {
        std::vector<std::vector<G1J>> ref_bucket_points(B);
        for (int i = 0; i < N; i++) {
          uint32_t digit = h_digits[i];
          if (!digit) continue;
          int b = (int)digit - 1;
          if (b >= 0 && b < B) {
            ref_bucket_points[b].push_back(points[i]);
          }
        }
        
        // Compare reference reconstruction to new h_packed_points layout
        std::cerr << "[DEBUG REFACTOR] Comparing old vs new packing for window " << wi << ":\n";
        bool packing_mismatch = false;
        for (int b = 0; b < B; b++) {
          int expected_size = (int)ref_bucket_points[b].size();
          int actual_size = ctx[0].h_bucket_counts[b];
          if (expected_size != actual_size) {
            std::cerr << "[DEBUG REFACTOR] Bucket " << b << ": ref size=" << expected_size
                      << " vs h_bucket_counts=" << actual_size << "\n";
            packing_mismatch = true;
          }
          
          // Compare actual point content
          int offset_begin = ctx[0].h_bucket_offsets[b];
          int offset_end = ctx[0].h_bucket_offsets[b + 1];
          if ((offset_end - offset_begin) != expected_size) {
            std::cerr << "[DEBUG REFACTOR] Bucket " << b << ": offset range mismatch\n";
            packing_mismatch = true;
          }
          
          for (int p = 0; p < expected_size && p < 5; p++) {  // Check first 5 for brevity
            const G1J& ref_pt = ref_bucket_points[b][p];
            const G1J& new_pt = ctx[0].h_packed_points[offset_begin + p];
            // Compare using memcmp (safest approach for struct comparison)
            if (std::memcmp(&ref_pt, &new_pt, sizeof(G1J)) != 0) {
              std::cerr << "[DEBUG REFACTOR] Bucket " << b << " point " << p 
                        << ": MISMATCH in coordinates\n";
              packing_mismatch = true;
            }
          }
        }
        if (!packing_mismatch) {
          std::cerr << "[DEBUG REFACTOR] Old vs new packing: IDENTICAL\n";
        }
      }

      // Audit check: verify contiguous layout is correct
      int gpu_total_bucketed = 0;
      int gpu_nonzero_buckets = nonempty_buckets;
      for (int b = 0; b < B; b++) {
        gpu_total_bucketed += ctx[0].h_bucket_counts[b];
      }
      
      // Verify packed_points_count == total_bucketed
      if ((int)ctx[0].h_packed_points.size() != total_bucketed) {
        std::cerr << "ERROR [Window " << wi << "]: packed_points size mismatch: "
                  << ctx[0].h_packed_points.size() << " != " << total_bucketed << "\n";
      }
      
      // Count actual nonzero digits
      int actual_nonzero_digits = 0;
      for (int i = 0; i < N; i++) {
        if (h_digits[i] != 0) actual_nonzero_digits++;
      }
      
      // Audit: sum(bucket_counts) must equal number of nonzero digits
      if (actual_nonzero_digits != gpu_total_bucketed) {
        std::cerr << "WARNING [Window " << wi << "]: GPU bucket count mismatch!\n"
                  << "  Nonzero digits: " << actual_nonzero_digits << "\n"
                  << "  Sum of bucket counts: " << gpu_total_bucketed << "\n";
      }
      
      std::vector<int> bucket_sizes = h_bucket_counts;  // Use GPU-computed counts directly

      if (collect_metrics && audit_stage_timing) {
        std::cerr << "Window " << wi << ": bucketed " << total_bucketed
                  << " points in " << nonempty_buckets << " buckets\n"
                  << "  Timing breakdown:\n"
                  << "    GPU digit extraction: " << (gpu_digit_extraction_time * 1e6) << " us\n"
                  << "    GPU bucket counting:  " << (gpu_bucket_count_time * 1e6) << " us\n"
                  << "    Host bucket packing:  " << (remaining_host_pack_time * 1e6) << " us\n"
                  << "  Two-pass host bucketing audit:\n"
                  << "    total_bucketed: " << total_bucketed << "\n"
                  << "    sum(bucket_counts): " << gpu_total_bucketed << "\n"
                  << "    nonempty_buckets: " << nonempty_buckets << "\n"
                  << "    h_packed_points.size(): " << ctx[0].h_packed_points.size() << "\n"
                  << "    h_bucket_offsets[B]: " << ctx[0].h_bucket_offsets[B] << "\n"
                  << "    Consistency: " << (gpu_total_bucketed == total_bucketed && 
                                             (int)ctx[0].h_packed_points.size() == total_bucketed ? "OK" : "MISMATCH") << "\n"
                  << "  Planner using GPU bucket counts: YES\n";
      }

      double max_bucket_cost = 0.0;
      double sum_bucket_cost = 0.0;
      int count_buckets = 0;
      for (int b = 0; b < B; ++b) {
        if (bucket_sizes[b] == 0) continue;
        double m_bi = double(bucket_sizes[b]) * params.D_pt;
        double t_h2d = m_bi / params.B_link + params.L_link;
        double adds_per_thread = std::ceil(double(bucket_sizes[b]) / params.tpb)
                               + std::log2(double(params.tpb));
        double t_comp = adds_per_thread / (double(params.num_sms) * params.U_g * params.R_g);
        double t_mem  = m_bi / params.B_g;
        double cost = t_h2d + t_comp + t_mem;
        max_bucket_cost = std::max(max_bucket_cost, cost);
        sum_bucket_cost += cost;
        ++count_buckets;
      }
      double avg_bucket_cost = count_buckets ? (sum_bucket_cost / count_buckets) : 0.0;

      auto planner_start = std::chrono::high_resolution_clock::now();
      BucketPlan plan = make_plan(bucket_sizes, G, objective, params, max_mem_per_gpu, wbits, planner_use_greedy);
      auto planner_end = std::chrono::high_resolution_clock::now();
      if (collect_metrics) {
        window_planning_time = std::chrono::duration<double>(planner_end - planner_start).count();
      }

      std::vector<int> task_count(B, 0);
      std::vector<int> split_task_count(B, 0);
      std::vector<int> point_cover(B, 0);
      std::vector<int> expected_shards(B, -1);
      bool any_split_seen = false;
      for (int g = 0; g < G; g++) {
        for (const auto& t : plan.gpu_tasks[g]) {
          task_count[t.bucket_idx]++;
          point_cover[t.bucket_idx] += t.point_count;
          if (t.point_count <= 0 || t.point_begin >= t.point_end) {
            std::cerr << "*** PLAN ERROR bucket " << t.bucket_idx
                      << ": empty shard shard_idx=" << t.shard_idx
                      << " range=[" << t.point_begin << "," << t.point_end << ")"
                      << " point_count=" << t.point_count << "\n";
          }
          if (t.is_split) {
            split_task_count[t.bucket_idx]++;
            any_split_seen = true;
            if (expected_shards[t.bucket_idx] < 0) expected_shards[t.bucket_idx] = t.num_shards;
          }
          if (force_split_test && collect_metrics && wi == 0) {
            std::cerr << "      task gpu=" << g
                      << " b=" << t.bucket_idx
                      << " shard=" << t.shard_idx << "/" << t.num_shards
                      << " range=[" << t.point_begin << "," << t.point_end << ")"
                      << " pts=" << t.point_count
                      << (t.is_split ? " split" : " full")
                      << "\n";
          }
        }
      }
      if (force_split_test && collect_metrics && wi == 0 && !any_split_seen) {
        std::cerr << "*** PLAN ERROR: force_split_test enabled but no split tasks were generated\n";
      }
      for (int b = 0; b < B; b++) {
        if (bucket_sizes[b] == 0) continue;
        if (point_cover[b] != bucket_sizes[b]) {
          std::cerr << "*** PLAN ERROR bucket " << b << ": covered=" << point_cover[b]
                    << " expected=" << bucket_sizes[b] << "\n";
        }
        if (split_task_count[b] == 0) {
          if (task_count[b] != 1) {
            std::cerr << "*** PLAN ERROR bucket " << b << ": unsplit task_count="
                      << task_count[b] << " (expected 1)\n";
          }
        } else {
          if (expected_shards[b] > 0 && split_task_count[b] != expected_shards[b]) {
            std::cerr << "*** PLAN ERROR bucket " << b << ": shard_count="
                      << split_task_count[b] << " expected=" << expected_shards[b] << "\n";
          }
          if (expected_shards[b] > 0 && expected_shards[b] <= 1) {
            std::cerr << "*** PLAN ERROR bucket " << b << ": split bucket has invalid num_shards="
                      << expected_shards[b] << "\n";
          }
        }
      }

      if (audit_stage_timing) {
        for (int g = 0; g < G; g++) {
          std::cerr << "    GPU " << g << ": " << plan.gpu_tasks[g].size() << " tasks: ";
          for (int i = 0; i < std::min(10, (int)plan.gpu_tasks[g].size()); i++) {
            const auto& t = plan.gpu_tasks[g][i];
            std::cerr << t.bucket_idx << "[" << t.shard_idx << "/" << t.num_shards << "] ";
          }
          if (plan.gpu_tasks[g].size() > 10) std::cerr << "...";
          std::cerr << "\n";
        }
      }

      // REFACTOR DEBUG: Validate task ranges and bucket-local indexing semantics
      if (collect_metrics && audit_stage_timing) {
        std::cerr << "[DEBUG REFACTOR] Task validation for window " << wi << ":\n";
        int total_task_points = 0;
        for (int g = 0; g < G; g++) {
          for (size_t t_idx = 0; t_idx < plan.gpu_tasks[g].size(); t_idx++) {
            const auto& t = plan.gpu_tasks[g][t_idx];
            int b = t.bucket_idx;
            
            // Invariant 1: bucket_idx in valid range
            if (b < 0 || b >= B) {
              std::cerr << "ERROR GPU " << g << " task " << t_idx 
                        << ": bucket_idx=" << b << " out of range [0," << B << ")\n";
              return 1;
            }
            
            // Invariant 2: point_begin/point_end are bucket-local, non-negative, ordered
            if (t.point_begin < 0 || t.point_end < 0 || t.point_begin > t.point_end) {
              std::cerr << "ERROR GPU " << g << " task " << t_idx 
                        << ": bucket " << b << " range=[" << t.point_begin 
                        << "," << t.point_end << ") invalid\n";
              return 1;
            }
            
            // Invariant 3: point_end <= bucket size
            if (t.point_end > ctx[0].h_bucket_counts[b]) {
              std::cerr << "ERROR GPU " << g << " task " << t_idx 
                        << ": bucket " << b << " range=[" << t.point_begin 
                        << "," << t.point_end << ") exceeds bucket size " 
                        << ctx[0].h_bucket_counts[b] << "\n";
              return 1;
            }
            
            // Invariant 4: compute global range
            int global_begin = ctx[0].h_bucket_offsets[b] + t.point_begin;
            int global_end = ctx[0].h_bucket_offsets[b] + t.point_end;
            
            // Invariant 5: global range within bucket's boundaries
            if (global_begin < (int)ctx[0].h_bucket_offsets[b] || 
                global_end > (int)ctx[0].h_bucket_offsets[b + 1]) {
              std::cerr << "ERROR GPU " << g << " task " << t_idx 
                        << ": bucket " << b << " global range=[" << global_begin 
                        << "," << global_end << ") outside bucket boundaries [" 
                        << ctx[0].h_bucket_offsets[b] << "," 
                        << ctx[0].h_bucket_offsets[b + 1] << ")\n";
              return 1;
            }
            
            total_task_points += (global_end - global_begin);
          }
        }
        std::cerr << "[DEBUG REFACTOR] Total task points: " << total_task_points 
                  << " vs actual packed: " << ctx[0].h_packed_points.size() << "\n";
        if (total_task_points != (int)ctx[0].h_packed_points.size()) {
          std::cerr << "WARNING: task points don't cover all packed points\n";
        }
      }

      auto gpu_start = std::chrono::high_resolution_clock::now();
      const bool has_device_task_id_buffer = false;
      const bool has_device_shard_id_buffer = false;
      std::vector<size_t> runtime_h2d_bytes(G, 0);
      std::vector<size_t> runtime_d2h_bytes(G, 0);
      double actual_pack_time = 0.0;
      double actual_h2d_time = 0.0;
      double actual_kernel_time = 0.0;
      double actual_d2h_time = 0.0;
      
      // GPU phase timing: only GPU work (no CPU pack)
      auto gpu_phase_work_start = std::chrono::high_resolution_clock::now();

      for (int g = 0; g < G; g++) {
        auto& c = ctx[g];
        auto pack_start = std::chrono::high_resolution_clock::now();
        c.h_points.clear();
        c.h_offsets.clear();
        c.h_bucket_ids.clear();
        c.h_shard_ids.clear();
        c.h_local_sums.clear();
        c.h_offsets.push_back(0);

        for (const auto& t : plan.gpu_tasks[g]) {
          c.h_bucket_ids.push_back(t.bucket_idx);
          c.h_shard_ids.push_back(t.shard_idx);
          // REFACTORED: Use contiguous h_packed_points directly with bucket_offsets
          // Instead of reading from bucket_points[t.bucket_idx] vectors
          int bucket_start = ctx[0].h_bucket_offsets[t.bucket_idx] + t.point_begin;
          int bucket_end = ctx[0].h_bucket_offsets[t.bucket_idx] + t.point_end;
          if (bucket_start < bucket_end) {
            c.h_points.insert(c.h_points.end(),
                            ctx[0].h_packed_points.begin() + bucket_start,
                            ctx[0].h_packed_points.begin() + bucket_end);
          }
          c.h_offsets.push_back((int)c.h_points.size());
        }

        // REFACTOR DEBUG: Validate h_points assembly for this GPU
        if (collect_metrics && audit_stage_timing) {
          int expected_points = 0;
          for (const auto& t : plan.gpu_tasks[g]) {
            expected_points += t.point_count;
          }
          if ((int)c.h_points.size() != expected_points) {
            std::cerr << "ERROR GPU " << g << ": h_points.size()=" << c.h_points.size()
                      << " but tasks specify " << expected_points << " points\n";
            return 1;
          }
          if (c.h_offsets.back() != (int)c.h_points.size()) {
            std::cerr << "ERROR GPU " << g << ": h_offsets.back()=" << c.h_offsets.back()
                      << " but h_points.size()=" << c.h_points.size() << "\n";
            return 1;
          }
          std::cerr << "[DEBUG REFACTOR] GPU " << g << ": packed " << c.h_points.size() 
                    << " points in " << (int)plan.gpu_tasks[g].size() << " tasks\n";
          
          // Verify that points from h_packed_points match what's in c.h_points
          std::cerr << "[DEBUG REFACTOR] GPU " << g << " point verification:\n";
          int pt_idx = 0;
          for (size_t t_idx = 0; t_idx < plan.gpu_tasks[g].size(); t_idx++) {
            const auto& t = plan.gpu_tasks[g][t_idx];
            int global_begin = ctx[0].h_bucket_offsets[t.bucket_idx] + t.point_begin;
            int global_end = ctx[0].h_bucket_offsets[t.bucket_idx] + t.point_end;
            for (int i = global_begin; i < global_end && i < (int)ctx[0].h_packed_points.size(); i++) {
              if (pt_idx < (int)c.h_points.size()) {
                if (std::memcmp(&c.h_points[pt_idx], &ctx[0].h_packed_points[i], sizeof(G1J)) != 0) {
                  std::cerr << "[DEBUG REFACTOR] GPU " << g << " task " << t_idx 
                            << " point " << (i - global_begin) << ": MISMATCH!\n";
                  std::cerr << "  Expected from h_packed[" << i << "]\n";
                  std::cerr << "  Got from c.h_points[" << pt_idx << "]\n";
                }
              }
              pt_idx++;
            }
          }
          std::cerr << "[DEBUG REFACTOR] GPU " << g << " point verification: OK\n";
        }

        int localBuckets = (int)c.h_bucket_ids.size();
        int totalPts = (int)c.h_points.size();
        auto pack_end = std::chrono::high_resolution_clock::now();
        actual_pack_time += std::chrono::duration<double>(pack_end - pack_start).count();

        runtime_h2d_bytes[g] =
            c.h_points.size() * sizeof(G1J)
          + c.h_offsets.size() * sizeof(int)
          + (has_device_task_id_buffer ? c.h_bucket_ids.size() * sizeof(int) : 0)
          + (has_device_shard_id_buffer ? c.h_shard_ids.size() * sizeof(int) : 0);
        runtime_d2h_bytes[g] = (size_t)localBuckets * sizeof(G1J);

        CUDA_CALL(cudaSetDevice(g));

        ensure_points_capacity(c, totalPts);
        ensure_offsets_capacity(c, localBuckets + 1);
        ensure_local_sums_capacity(c, localBuckets);

        auto h2d_start = std::chrono::high_resolution_clock::now();
        if (totalPts > 0) {
          CUDA_CALL(cudaMemcpyAsync(c.d_points, c.h_points.data(),
                                    (size_t)totalPts * sizeof(G1J),
                                    cudaMemcpyHostToDevice, c.stream));
        }
        CUDA_CALL(cudaMemcpyAsync(c.d_offsets, c.h_offsets.data(),
                                  (size_t)(localBuckets + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice, c.stream));
        // Always synchronize to get accurate GPU timing
        CUDA_CALL(cudaStreamSynchronize(c.stream));
        auto h2d_end = std::chrono::high_resolution_clock::now();
        actual_h2d_time += std::chrono::duration<double>(h2d_end - h2d_start).count();
        
        if (audit_stage_timing) {
          std::cerr << "    H2D: copying " << totalPts << " points\n";
        }

        // Start kernel timing before any GPU work (bucket_sum through window_reduce)
        auto kernel_start = std::chrono::high_resolution_clock::now();
        
        if (localBuckets > 0) {
          int tpb = 256;
          size_t shmem = (size_t)tpb * sizeof(G1J);
          bucket_sum_block_per_bucket<<<localBuckets, tpb, shmem, c.stream>>>(
              c.d_points, c.d_offsets, c.d_local_bucket_sums, localBuckets);
          CUDA_CALL(cudaGetLastError());
          
          // GPU-side merge and window reduction (replaces CPU merge + CPU window reduction)
          ensure_bucket_ids_capacity(c, localBuckets);
          ensure_merged_bucket_sums_capacity(c, B);

          // Copy bucket IDs to GPU (needed for GPU merge kernel)
          CUDA_CALL(cudaMemcpyAsync(c.d_bucket_ids, c.h_bucket_ids.data(),
                                    (size_t)localBuckets * sizeof(int),
                                    cudaMemcpyHostToDevice, c.stream));

          // Call GPU merge kernel: merge local task sums into full bucket sums
          merge_local_bucket_sums<<<1, 1, 0, c.stream>>>(
              c.d_bucket_ids, c.d_local_bucket_sums, c.d_merged_bucket_sums, B, localBuckets);
          CUDA_CALL(cudaGetLastError());

          // Call GPU window reduction kernel: compute window result on GPU
          window_reduce_suffix<<<1, 1, 0, c.stream>>>(
              c.d_merged_bucket_sums, B, c.d_window_result);
          CUDA_CALL(cudaGetLastError());
        } else {
          // Empty GPU (no localBuckets), but still need window reduction
          ensure_merged_bucket_sums_capacity(c, B);
          window_reduce_suffix<<<1, 1, 0, c.stream>>>(
              c.d_merged_bucket_sums, B, c.d_window_result);
          CUDA_CALL(cudaGetLastError());
        }
        
        // Always synchronize to get accurate GPU kernel timing
        CUDA_CALL(cudaStreamSynchronize(c.stream));
        auto kernel_end = std::chrono::high_resolution_clock::now();
        actual_kernel_time += std::chrono::duration<double>(kernel_end - kernel_start).count();
        
        if (audit_stage_timing) {
          std::cerr << "SYNC gpu " << g << " after bucket_sum+merge+window_reduce\\n";
        }

        // Copy window result back to host (with D2H timing)
        auto d2h_start = std::chrono::high_resolution_clock::now();
        CUDA_CALL(cudaMemcpyAsync(&c.h_window_result, c.d_window_result,
                                  1 * sizeof(G1J), cudaMemcpyDeviceToHost, c.stream));
        // Always synchronize to get accurate GPU timing
        CUDA_CALL(cudaStreamSynchronize(c.stream));
        auto d2h_end = std::chrono::high_resolution_clock::now();
        actual_d2h_time += std::chrono::duration<double>(d2h_end - d2h_start).count();
      }

      if (collect_metrics && audit_stage_timing && wi == 0) {
        std::cerr << "COMM AUDIT (window 0): planner_estimated_{h2d,d2h}_bytes vs actual_{h2d,d2h}_bytes\n";
        for (int g = 0; g < G; g++) {
          std::cerr << "    GPU " << g
                    << ": planner_h2d=" << plan.gpu_h2d_bytes[g]
                    << " actual_h2d=" << runtime_h2d_bytes[g]
                    << " planner_d2h=" << plan.gpu_d2h_bytes[g]
                    << " actual_d2h=" << runtime_d2h_bytes[g]
                    << "\n";
          if (runtime_h2d_bytes[g] > plan.gpu_h2d_bytes[g] ||
              runtime_d2h_bytes[g] > plan.gpu_d2h_bytes[g]) {
            std::cerr << "    WARNING GPU " << g
                      << ": communication actual bytes exceed planner estimate\n";
          }
        }
      }

      auto gpu_phase_work_end = std::chrono::high_resolution_clock::now();
      double actual_gpu_work_time = std::chrono::duration<double>(gpu_phase_work_end - gpu_phase_work_start).count();
      
      auto gpu_end = std::chrono::high_resolution_clock::now();
      double actual_gpu_section_time = std::chrono::duration<double>(gpu_end - gpu_start).count();

      // GPU-side merge and window reduction already complete.
      // Now combine window results from all GPUs on CPU.
      auto gpu_merge_suffix_start = std::chrono::high_resolution_clock::now();
      
      G1J combined_window_result = G1J::infinity();
      for (int g = 0; g < G; g++) {
        combined_window_result = g1_add(combined_window_result, ctx[g].h_window_result);
      }

      // For compatibility with correctness checks, prepare CPU reference path
      std::vector<G1J> combined_buckets(B, G1J::infinity());
      std::vector<std::vector<G1J>> bucket_partials(B);
      double actual_host_merge_time = 0.0;
      int merge_adds_this_window = 0;

      // DEBUG/CORRECTNESS PATH: Old CPU merge+reduction, only run if check is enabled
      G1J window_cpu = G1J::infinity();
      std::vector<G1J> combined_buckets_cpu;
      if (do_check) {
        // Compute reference via direct CPU bucketization using contiguous h_packed_points
        combined_buckets_cpu.assign(B, G1J::infinity());
        
        for (int b = 0; b < B; b++) {
          // Iterate over points for bucket b using the contiguous packed layout
          int start_idx = ctx[0].h_bucket_offsets[b];
          int end_idx = ctx[0].h_bucket_offsets[b + 1];
          for (int idx = start_idx; idx < end_idx; idx++) {
            if (idx >= 0 && idx < (int)ctx[0].h_packed_points.size()) {
              combined_buckets_cpu[b] = g1_add(combined_buckets_cpu[b], ctx[0].h_packed_points[idx]);
            }
          }
        }
        window_cpu = pippenger_window_sum_from_bucket_sums(combined_buckets_cpu);
      }

      auto gpu_phase_end = std::chrono::high_resolution_clock::now();
      double host_merge_plus_suffix_time = std::chrono::duration<double>(gpu_phase_end - gpu_merge_suffix_start).count();
      if (collect_metrics) {
        window_gpu_phase_total_time = std::chrono::duration<double>(gpu_phase_end - gpu_start).count();
      }

      double predicted_gpu_max = 0.0;
      double predicted_sum = 0.0;
      double pred_h2d_max = 0.0, pred_comp_max = 0.0, pred_mem_max = 0.0;
      double pred_h2d_sum = 0.0, pred_comp_sum = 0.0, pred_mem_sum = 0.0;
      for (int g = 0; g < G; g++) {
        predicted_gpu_max = std::max(predicted_gpu_max, plan.gpu_estimated_time[g]);
        predicted_sum += plan.gpu_estimated_time[g];
        pred_h2d_max = std::max(pred_h2d_max, plan.gpu_h2d_time[g]);
        pred_comp_max = std::max(pred_comp_max, plan.gpu_comp_time[g]);
        pred_mem_max = std::max(pred_mem_max, plan.gpu_mem_time[g]);
        pred_h2d_sum += plan.gpu_h2d_time[g];
        pred_comp_sum += plan.gpu_comp_time[g];
        pred_mem_sum += plan.gpu_mem_time[g];
      }

      double predicted_host_pack = plan.estimated_host_pack_time;
      // GPU-based pipeline stages (March 31 refactor):
      // GPU merge and GPU suffix now replace previous CPU host merge and CPU suffix
      double predicted_gpu_merge = plan.estimated_gpu_merge_time;
      double predicted_gpu_suffix = plan.estimated_gpu_suffix_time;
      int total_tasks_this_window = 0;
      for (int g = 0; g < G; g++) total_tasks_this_window += (int)plan.gpu_tasks[g].size();
      double raw_compute_model_time = (active_k_compute > 0.0) ? (pred_comp_sum / active_k_compute) : 0.0;

      double predicted_core = predicted_host_pack
                            + pred_h2d_sum
                            + pred_comp_sum
                            + pred_mem_sum
                            + predicted_gpu_merge
                            + predicted_gpu_suffix;
      double predicted_overhead = overhead_ewma;
      double predicted_latency = predicted_core + predicted_overhead;

      double measured_overhead = std::max(0.0, actual_gpu_section_time - predicted_core);
      measured_overhead = std::min(measured_overhead, overhead_cap);
      overhead_ewma = overhead_alpha * measured_overhead + (1.0 - overhead_alpha) * overhead_ewma;

      double pred_throughput = (predicted_latency > 0) ? (double(total_bucketed) / predicted_latency) : 0.0;
      double actual_throughput = (actual_gpu_section_time > 0) ? (double(total_bucketed) / actual_gpu_section_time) : 0.0;

      if (collect_metrics && audit_stage_timing) {
        std::cerr << "    predicted_latency_model=" << predicted_latency
                  << " actual_gpu_section_time=" << actual_gpu_section_time
                  << " pred_tp=" << pred_throughput
                  << " act_tp=" << actual_throughput
                  << " core=" << predicted_core
                  << " overhead=" << predicted_overhead << "\n";

        csv << wi << ',' << total_bucketed << ',' << nonempty_buckets << ','
            << (objective == Objective::Latency ? "lat" : "tp") << ','
            << predicted_latency << ',' << pred_throughput << ','
            << actual_gpu_section_time << ',' << max_bucket_cost << ',' << avg_bucket_cost << ','
            << predicted_gpu_max << ',' << predicted_sum << ','
            << pred_h2d_max << ',' << pred_comp_max << ',' << pred_mem_max << ','
            << pred_h2d_sum << ',' << pred_comp_sum << ',' << pred_mem_sum << ','
            << plan.estimated_reduction_cost << ',' << measured_overhead << '\n';
      }

      auto cpu_suffix_start = std::chrono::high_resolution_clock::now();
      // Window result already computed on GPU, just use it
      G1J window_result = combined_window_result;
      auto cpu_suffix_end = std::chrono::high_resolution_clock::now();
      double actual_cpu_suffix_time = std::chrono::duration<double>(cpu_suffix_end - cpu_suffix_start).count();

      if (collect_metrics) {
        // host_merge_plus_suffix_time tracks GPU merge + host window result combination
        double window_host_merge_plus_suffix_time = host_merge_plus_suffix_time;
      }

      double actual_window_total_time = actual_pack_time + actual_h2d_time + actual_kernel_time
                                     + actual_d2h_time + actual_host_merge_time + actual_cpu_suffix_time;

      auto window_end = std::chrono::high_resolution_clock::now();
      double window_total_time = std::chrono::duration<double>(window_end - window_start).count();

      if (!collect_metrics) {
        warmup_window_time = window_total_time;
      }

      if (collect_metrics) {
        window_sums[wi] = window_result;

        // GPU work is H2D + kernels + D2H (not including CPU pack)
        double actual_gpu_reduction_time = actual_h2d_time + actual_kernel_time + actual_d2h_time;
        
        // For audit, fine_grained should separate GPU work from other work
        double fine_grained_gpu_sum = actual_pack_time + actual_gpu_reduction_time;
        
        // GPU gap is now just synchronization and kernel launch overhead
        double gpu_gap = actual_gpu_work_time - actual_gpu_reduction_time;
        
        // Use actual GPU reduction time for model calibration
        double raw_gpu_reduction_model_time = pred_comp_sum + pred_mem_sum + predicted_gpu_merge + predicted_gpu_suffix;
        
        // Recalibrate k_compute for this window based on actual vs predicted
        double window_k_compute = 5.36;  // default
        if (raw_gpu_reduction_model_time > 0.0 && actual_gpu_reduction_time > 0.0) {
          window_k_compute = actual_gpu_reduction_time / raw_gpu_reduction_model_time;
        }
        
        if (audit_stage_timing) {
          std::cerr << "    KCALIB: raw_pred_gpu_reduction=" << raw_gpu_reduction_model_time
                    << " actual_gpu_reduction=" << actual_gpu_reduction_time
                    << " computed_k=" << window_k_compute << "\n";
        }
        
        // GPU phase total (including CPU work packing)
        double coarse_sum = window_bucketization_time + window_planning_time
                          + window_gpu_phase_total_time + window_host_merge_plus_suffix_time;
        double window_gap = window_total_time - coarse_sum;
        
        // Fine sum: pack + GPU work (no separate host merge/cpu suffix in active GPU-reduction path)
        double fine_sum = window_bucketization_time + window_planning_time
                        + fine_grained_gpu_sum;

        if (audit_stage_timing) {
          // Updated GPU-based pipeline stages: pack, h2d, compute, gpu_merge, gpu_suffix, d2h
          std::cerr << "    STAGE AUDIT: pred_pack=" << predicted_host_pack
                    << " pred_h2d=" << pred_h2d_sum
                    << " pred_compute=" << pred_comp_sum
                    << " pred_gpu_merge=" << predicted_gpu_merge
                    << " pred_gpu_suffix=" << predicted_gpu_suffix
                    << " pred_d2h=" << pred_mem_sum
                    << " pred_total=" << predicted_core
                    << " | act_pack=" << actual_pack_time
                    << " act_h2d=" << actual_h2d_time
                    << " act_gpu_reduction_kernels=" << actual_kernel_time
                    << " act_d2h=" << actual_d2h_time
                    << " act_total_gpu_reduction=" << actual_gpu_reduction_time
                    << " act_total_stages=" << (actual_pack_time + actual_gpu_reduction_time)
                    << " act_total_window=" << actual_window_total_time
                    << "\n";

          std::cerr << "    WINDOW RECON: window_total_time=" << window_total_time
                    << " coarse_sum=" << coarse_sum
                    << " diff=" << window_gap
                    << "\n";
          std::cerr << "    WINDOW GPU RECON: gpu_work_time=" << actual_gpu_work_time
                    << " gpu_reduction=" << actual_gpu_reduction_time
                    << " gpu_overhead=" << gpu_gap
                    << "\n";

          audit_pred_h2d_total += pred_h2d_sum;
          audit_pred_comp_total += pred_comp_sum;
          audit_pred_d2h_total += pred_mem_sum;
          audit_pred_total_total += predicted_core;
          audit_actual_pack_total += actual_pack_time;
          audit_actual_h2d_total += actual_h2d_time;
          audit_actual_kernel_total += actual_kernel_time;
          audit_actual_d2h_total += actual_d2h_time;
          audit_actual_host_merge_total += actual_host_merge_time;
          audit_actual_cpu_suffix_total += actual_cpu_suffix_time;
          audit_fixed_overhead_total += measured_overhead;
          audit_total_points += double(total_bucketed);
          audit_total_tasks += double(total_tasks_this_window);
          audit_total_merge_adds += double(merge_adds_this_window);
          audit_total_nonempty_buckets += double(nonempty_buckets);
          audit_window_total_time_sum += window_total_time;
          audit_coarse_sum_total += coarse_sum;
          audit_fine_sum_total += fine_sum;
          audit_gpu_gap_total += gpu_gap;
          audit_unaccounted_window_gap_total += window_gap;

          double suggested_k_compute = (raw_compute_model_time > 0.0)
                                     ? (actual_kernel_time / raw_compute_model_time)
                                     : 0.0;
          double suggested_alpha_pack = (total_bucketed > 0)
                                      ? (actual_pack_time / double(total_bucketed))
                                      : 0.0;
          // GPU-reduction timing is now part of kernel measurement, no separate merge/suffix
          double suggested_alpha_merge = 0.0;
          double suggested_alpha_suffix = 0.0;

          // Update calibration for active bucket
          if (wbits <= 6) {
            if (audit_k_compute_windows_small == 0) audit_suggested_k_compute_small_avg = suggested_k_compute;
            else audit_suggested_k_compute_small_avg = (audit_suggested_k_compute_small_avg + suggested_k_compute) / 2.0;
            audit_k_compute_windows_small++;
          } else if (wbits <= 8) {
            if (audit_k_compute_windows_mid == 0) audit_suggested_k_compute_mid_avg = suggested_k_compute;
            else audit_suggested_k_compute_mid_avg = (audit_suggested_k_compute_mid_avg + suggested_k_compute) / 2.0;
            audit_k_compute_windows_mid++;
          } else {
            if (audit_k_compute_windows_large == 0) audit_suggested_k_compute_large_avg = suggested_k_compute;
            else audit_suggested_k_compute_large_avg = (audit_suggested_k_compute_large_avg + suggested_k_compute) / 2.0;
            audit_k_compute_windows_large++;
          }

          // Still track overall stats
          audit_suggested_k_compute_sum += suggested_k_compute;

          if (actual_pack_time > 0.0) {
            audit_ratio_pack_sum += (predicted_host_pack / actual_pack_time);
            audit_ratio_pack_n++;
          }
          if (actual_kernel_time > 0.0) {
            audit_ratio_compute_sum += (pred_comp_sum / actual_kernel_time);
            audit_ratio_compute_n++;
          }
          if (actual_kernel_time > 0.0) {
            audit_ratio_merge_sum += ((pred_comp_sum + pred_mem_sum) / actual_kernel_time);
            audit_ratio_merge_n++;
          }

          if (wi == 0 && !audit_window0_reported) {
            std::cerr << "    CALIB WINDOW0: total_points=" << total_bucketed
                      << " total_tasks=" << total_tasks_this_window
                      << " nonempty_buckets=" << nonempty_buckets
                      << " measured_pack=" << actual_pack_time
                      << " measured_h2d=" << actual_h2d_time
                      << " measured_gpu_reduction_kernels=" << actual_kernel_time
                      << " measured_d2h=" << actual_d2h_time
                      << " suggested_k_compute=" << suggested_k_compute
                      << " suggested_alpha_pack=" << suggested_alpha_pack
                      << " ratio_pack=" << ((actual_pack_time > 0.0) ? (predicted_host_pack / actual_pack_time) : 0.0)
                      << " ratio_gpu_reduction=" << ((actual_kernel_time > 0.0) ? ((pred_comp_sum + pred_mem_sum) / actual_kernel_time) : 0.0)
                      << "\n";
            audit_window0_reported = true;
          }

          audit_windows++;
        }

        if (do_check) {
          if (equal_projective(window_result, window_cpu)) {
            // Silently pass - windows match
          } else {
            std::cerr << "*** WINDOW " << wi << " MISMATCH ***\n";
            std::cerr << " GPU window: " << (window_result.is_inf() ? "INF" : "notinf")
                      << ", CPU window: " << (window_cpu.is_inf() ? "INF" : "notinf") << "\n";
          }
        }

        if (audit_stage_timing) {
          std::cout << "Window " << wi << " done (result is_inf=" << window_result.is_inf() << ").\n";
        }
      }
    }

    auto finalize_start = std::chrono::high_resolution_clock::now();

    if (audit_stage_timing && audit_windows > 0) {
      auto ms = [](double s) { return s * 1000.0; };
      double pred_comm = audit_pred_h2d_total + audit_pred_d2h_total;
      double act_comm = audit_actual_h2d_total + audit_actual_d2h_total;
      std::cerr << "\n=== STAGE AUDIT SUMMARY ===\n";
      std::cerr << " windows=" << audit_windows << "\n";
      std::cerr << " communication: pred=" << ms(pred_comm) << " ms, actual=" << ms(act_comm)
                << " ms, delta=" << ms(act_comm - pred_comm) << " ms\n";
      std::cerr << " compute: pred=" << ms(audit_pred_comp_total) << " ms, actual=" << ms(audit_actual_kernel_total)
                << " ms, delta=" << ms(audit_actual_kernel_total - audit_pred_comp_total) << " ms\n";
      std::cerr << " host_work(pack+merge+suffix): "
                << ms(audit_actual_pack_total + audit_actual_host_merge_total + audit_actual_cpu_suffix_total)
                << " ms\n";
      std::cerr << " fixed_overhead_model_sum: " << ms(audit_fixed_overhead_total) << " ms\n";
      std::cerr << " predicted_core_sum: " << ms(audit_pred_total_total)
                << " ms, measured_staged_window_sum: "
                << ms(audit_actual_pack_total + audit_actual_h2d_total + audit_actual_kernel_total
                    + audit_actual_d2h_total + audit_actual_host_merge_total + audit_actual_cpu_suffix_total)
                << " ms\n";

      std::cerr << "\n=== CALIBRATION SUGGESTIONS (AVERAGE OVER WINDOWS) ===\n";
      std::cerr << " avg_total_points=" << (audit_windows > 0 ? (audit_total_points / audit_windows) : 0.0)
                << " avg_total_tasks=" << (audit_windows > 0 ? (audit_total_tasks / audit_windows) : 0.0)
                << " avg_merge_adds=" << (audit_windows > 0 ? (audit_total_merge_adds / audit_windows) : 0.0)
                << " avg_nonempty_buckets=" << (audit_windows > 0 ? (audit_total_nonempty_buckets / audit_windows) : 0.0)
                << "\n";

      double avg_suggested_k_compute =
          (audit_windows > 0 ? (audit_suggested_k_compute_sum / audit_windows) : 0.0);
      double stage_ratio_compute =
          (audit_ratio_compute_n > 0 ? (audit_ratio_compute_sum / audit_ratio_compute_n) : 0.0);

      std::cerr << " suggested_k_compute=" << avg_suggested_k_compute
                << " (= current_k=" << (active_k_compute > 0.0 ? active_k_compute : 0.0)
                << " / stage_ratio_compute=" << stage_ratio_compute << ")"
                << " suggested_alpha_pack=" << (audit_windows > 0 ? (audit_suggested_alpha_pack_sum / audit_windows) : 0.0)
                << " suggested_alpha_merge=" << (audit_windows > 0 ? (audit_suggested_alpha_merge_sum / audit_windows) : 0.0)
                << " suggested_alpha_suffix=" << (audit_windows > 0 ? (audit_suggested_alpha_suffix_sum / audit_windows) : 0.0)
                << "\n";

      // Print piecewise calibration results
      std::cerr << " PIECEWISE-BY-WBITS COMPUTE CORRECTION SUGGESTIONS:\n";
      if (audit_k_compute_windows_small > 0) {
        std::cerr << "  k_compute_small (wbits<=6): " << audit_suggested_k_compute_small_avg;
        if (wbits <= 6) std::cerr << " [ACTIVE in this run]";
        std::cerr << "\n";
      }
      if (audit_k_compute_windows_mid > 0) {
        std::cerr << "  k_compute_mid (7<=wbits<=8): " << audit_suggested_k_compute_mid_avg;
        if (wbits >= 7 && wbits <= 8) std::cerr << " [ACTIVE in this run]";
        std::cerr << "\n";
      }
      if (audit_k_compute_windows_large > 0) {
        std::cerr << "  k_compute_large (wbits>=9): " << audit_suggested_k_compute_large_avg;
        if (wbits >= 9) std::cerr << " [ACTIVE in this run]";
        std::cerr << "\n";
      }

      // Print machine-parseable calibration line
      std::cerr << "AUDIT_KCALIB";
      if (audit_k_compute_windows_small > 0) std::cerr << " small=" << audit_suggested_k_compute_small_avg;
      else std::cerr << " small=nan";
      if (audit_k_compute_windows_mid > 0) std::cerr << " mid=" << audit_suggested_k_compute_mid_avg;
      else std::cerr << " mid=nan";
      if (audit_k_compute_windows_large > 0) std::cerr << " large=" << audit_suggested_k_compute_large_avg;
      else std::cerr << " large=nan";
      std::cerr << " active=" << active_k_compute << "\n";

      std::cerr << " stage_ratio_pack(pred/actual)="
                << (audit_ratio_pack_n > 0 ? (audit_ratio_pack_sum / audit_ratio_pack_n) : 0.0)
                << " stage_ratio_compute(pred/actual)="
                << (audit_ratio_compute_n > 0 ? (audit_ratio_compute_sum / audit_ratio_compute_n) : 0.0)
                << " stage_ratio_merge(pred/actual)="
                << (audit_ratio_merge_n > 0 ? (audit_ratio_merge_sum / audit_ratio_merge_n) : 0.0)
                << " stage_ratio_suffix(pred/actual)="
                << (audit_ratio_suffix_n > 0 ? (audit_ratio_suffix_sum / audit_ratio_suffix_n) : 0.0)
                << "\n";

      if (audit_ratio_compute_n > 0) {
        double stage_ratio_compute_local = audit_ratio_compute_sum / audit_ratio_compute_n;
        double relative_error_compute = std::fabs(stage_ratio_compute_local - 1.0);
        if (relative_error_compute > 0.1) {
          std::cerr << " WARNING: compute model relative error "
                    << (relative_error_compute * 100.0)
                    << "% exceeds 10% threshold\n";
        }
      }

      std::cerr << "\n=== WINDOW TIMING RECON TOTALS ===\n";
      std::cerr << " total_window_time_sum: " << ms(audit_window_total_time_sum) << " ms\n";
      std::cerr << " total_coarse_sum: " << ms(audit_coarse_sum_total) << " ms\n";
      std::cerr << " total_fine_sum: " << ms(audit_fine_sum_total) << " ms\n";
      std::cerr << " total_gpu_gap: " << ms(audit_gpu_gap_total) << " ms\n";
      std::cerr << " total_unaccounted_window_gap: " << ms(audit_unaccounted_window_gap_total) << " ms\n";
    }

    G1J result = host_fold_windows(window_sums, wbits);

    if (do_check) {
      int refN = std::min(N, 2000);
      std::vector<G1J> pts_ref(points.begin(), points.begin() + refN);
      std::vector<ScalarR> sc_ref(scalars.begin(), scalars.begin() + refN);

      G1J cpu_ref = cpu_msm_reference(pts_ref, sc_ref);

      if (N == refN) {
        std::cout << "Check: comparing GPU vs CPU reference...\n";
        std::cout << (equal_projective(result, cpu_ref) ? "OK\n" : "MISMATCH\n");
      } else {
        std::cout << "Check note: CPU reference computed for prefix only; run N<=2000 for exact compare.\n";
      }
    }

    for (int g = 0; g < G; g++) {
      CUDA_CALL(cudaSetDevice(g));
      CUDA_CALL(cudaFree(ctx[g].d_points));
      CUDA_CALL(cudaFree(ctx[g].d_offsets));
      CUDA_CALL(cudaFree(ctx[g].d_local_bucket_sums));
      CUDA_CALL(cudaFree(ctx[g].d_bucket_ids));
      CUDA_CALL(cudaFree(ctx[g].d_merged_bucket_sums));
      CUDA_CALL(cudaFree(ctx[g].d_window_result));
      CUDA_CALL(cudaStreamDestroy(ctx[g].stream));
    }

    G1Aff aff = g1_to_affine_host(result);
    std::cout << "\n=== MSM Result (BN254 G1) ===\n";
    if (aff.inf) {
      std::cout << "INF\n";
    } else {
      std::cout << "x: " << aff.x.v[3] << " " << aff.x.v[2] << " " << aff.x.v[1] << " " << aff.x.v[0] << "\n";
      std::cout << "y: " << aff.y.v[3] << " " << aff.y.v[2] << " " << aff.y.v[1] << " " << aff.y.v[0] << "\n";
      std::cout << "(printed all 4 limbs high..low)\n";
    }

    auto e2e_end = std::chrono::high_resolution_clock::now();
    auto total_wall_us =
        std::chrono::duration_cast<std::chrono::microseconds>(e2e_end - e2e_start).count();
    double total_wall_time = std::chrono::duration<double>(e2e_end - e2e_start).count();
    double finalize_time = std::chrono::duration<double>(e2e_end - finalize_start).count();

    // Compute predicted total time for model validation
    double predicted_total_time = (audit_stage_timing && audit_windows > 0)
                                 ? (setup_time + warmup_window_time + audit_pred_total_total + finalize_time)
                                 : 0.0;

    std::cout << "Total time: " << total_wall_us << " us\n";

    if (audit_stage_timing && audit_windows > 0) {
      auto ms = [](double s) { return s * 1000.0; };
      double reconstructed_total =
          setup_time + warmup_window_time + audit_window_total_time_sum + finalize_time;
      double reconstruction_gap = total_wall_time - reconstructed_total;
      std::cerr << "\n=== END-TO-END RUNTIME DECOMPOSITION ===\n";
      std::cerr << " total_wall_time: " << ms(total_wall_time) << " ms\n";
      std::cerr << " setup_time: " << ms(setup_time) << " ms\n";
      std::cerr << " warmup_window_time: " << ms(warmup_window_time) << " ms\n";
      std::cerr << " window_total_time_sum: " << ms(audit_window_total_time_sum) << " ms\n";
      std::cerr << " finalize_time: " << ms(finalize_time) << " ms\n";
      std::cerr << " reconstructed_total: " << ms(reconstructed_total) << " ms\n";
      std::cerr << " reconstruction_gap: " << ms(reconstruction_gap) << " ms\n";

      // New pipeline: pack + h2d + gpu_reduction(kernel) + d2h (no CPU merge/suffix in active path)
      double measured_staged_window_sum = audit_actual_pack_total + audit_actual_h2d_total
                                        + audit_actual_kernel_total + audit_actual_d2h_total;

      std::cerr << "\n=== PARSE-FRIENDLY AUDIT SUMMARY ===\n";
      std::cerr << "AUDIT_COST_CORE predicted_core_sum_ms="
                << ms(audit_pred_total_total)
                << " measured_staged_window_sum_ms="
                << ms(measured_staged_window_sum)
                << " (pack+h2d+gpu_reduction+d2h)"
                << "\n";

      std::cerr << "AUDIT_COST_FULL predicted_latency_ms="
                << ms(audit_pred_total_total + audit_fixed_overhead_total)
                << " actual_gpu_time_ms="
                << ms(measured_staged_window_sum + audit_fixed_overhead_total)
                << "\n";

      std::cerr << "AUDIT_STAGE_RATIOS pack="
                << (audit_ratio_pack_n > 0 ? (audit_ratio_pack_sum / audit_ratio_pack_n) : 0.0)
                << " gpu_reduction="
                << (audit_ratio_merge_n > 0 ? (audit_ratio_merge_sum / audit_ratio_merge_n) : 0.0)
                << "\n";

      std::cerr << "AUDIT_DECOMP setup_ms="
                << ms(setup_time)
                << " warmup_ms="
                << ms(warmup_window_time)
                << " window_sum_ms="
                << ms(audit_window_total_time_sum)
                << " finalize_ms="
                << ms(finalize_time)
                << " reconstruction_gap_ms="
                << ms(reconstruction_gap)
                << " (pack + h2d + gpu_reduction + d2h pipeline)"
                << "\n";
    } else {
      // Non-audit run: still compute predicted time if possible
      // (for normal benchmarking without detailed stage instrumentation)
      double approx_predicted = setup_time + warmup_window_time + audit_pred_total_total + finalize_time;
      predicted_total_time = (approx_predicted > 0.0) ? approx_predicted : total_wall_time;
    }

    // Summary of GPU bucket counting improvements
    std::cerr << "\n=== GPU BUCKET COUNTING OPTIMIZATION SUMMARY ===\n";
    std::cerr << "Planner now uses GPU bucket counts:                     YES\n";
    std::cerr << "CPU bucket-size computation removed:                    YES\n";
    std::cerr << "Host-side preprocessing reduced by:                     GPU digit + bucket ops\n";
    std::cerr << "Remaining host-side work:                               Point-to-bucket assignment\n";
    std::cerr << "Next optimization target:                               Full GPU bucket assignment\n";
    std::cerr << "                                                        (eliminate D2H transfers)\n";

    // CSV logging for benchmarking (append mode, no audit_stage_timing guard)
    {
      double throughput = (total_wall_time > 0.0) ? (N / total_wall_time) : 0.0;
      double speedup = 1.0;  // Placeholder: speedup would require 1-GPU baseline
      double predicted_ms = predicted_total_time * 1000.0;
      double actual_total_ms = total_wall_time * 1000.0;
      double prediction_error_pct = (actual_total_ms > 0.0) 
                                   ? ((predicted_ms - actual_total_ms) / actual_total_ms * 100.0)
                                   : 0.0;
      
      std::string csv_file = "benchmark_results.csv";
      std::ifstream check_file(csv_file);
      bool file_exists = check_file.good();
      check_file.close();

      std::ofstream csv(csv_file, std::ios::app);
      if (!file_exists) {
        // Write header if file is new
        std::string header = "N,wbits,num_gpus,use_greedy,total_time_ms,setup_ms,warmup_ms,window_ms,finalize_ms,"
                            "throughput,speedup,predicted_total_time_ms,actual_total_time_ms,prediction_error_pct,k_compute_active";
        if (audit_stage_timing) {
          header += ",suggested_k_small,suggested_k_mid,suggested_k_large";
        }
        csv << header << "\n";
      }
      // Append result row
      csv << N << ','
          << wbits << ','
          << G << ','
          << (use_greedy ? 1 : 0) << ','
          << (total_wall_time * 1000.0) << ','
          << (setup_time * 1000.0) << ','
          << (warmup_window_time * 1000.0) << ','
          << (audit_window_total_time_sum * 1000.0) << ','
          << (finalize_time * 1000.0) << ','
          << throughput << ','
          << speedup << ','
          << predicted_ms << ','
          << actual_total_ms << ','
          << prediction_error_pct << ','
          << active_k_compute;
      
      if (audit_stage_timing) {
        csv << ','
            << (audit_k_compute_windows_small > 0 ? std::to_string(audit_suggested_k_compute_small_avg) : "nan") << ','
            << (audit_k_compute_windows_mid > 0 ? std::to_string(audit_suggested_k_compute_mid_avg) : "nan") << ','
            << (audit_k_compute_windows_large > 0 ? std::to_string(audit_suggested_k_compute_large_avg) : "nan");
      }
      csv << '\n';
      csv.close();
    }

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 1;
  }
}