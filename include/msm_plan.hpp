// include/msm_plan.hpp
#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cstddef>  // for SIZE_MAX
#include <cmath>    // for log2
#include "ec_bn254.cuh"


// parameters describing the system (Table 1 in thesis)
struct SystemParams {
    double R_g;      // compute throughput of one GPU thread (ops/s)
    double B_g;      // memory bandwidth (bytes/s)
    double B_link;   // interconnect bandwidth (bytes/s)
    double L_link;   // interconnect latency (seconds)
    double U_g;      // effective utilization fraction
    double D_pt;     // size of a point in bytes
    double L_sync;   // sync/merge latency (seconds)
    double L_h2d;    // H2D transfer latency (seconds)
    double L_d2h;    // D2H transfer latency (seconds)
    // Active compute model: piecewise correction by wbits bucket.
    double k_compute_small;  // compute correction for wbits <= 6
    double k_compute_mid;    // compute correction for 7 <= wbits <= 8
    double k_compute_large;  // compute correction for wbits >= 9
    // PHASE 1: GPU digit extraction and bucket counting costs
    double k_digit;      // time per scalar for kernel_compute_digits
    double k_count;      // time per scalar for kernel_count_buckets
    // Host-side and GPU-side linear terms
    double alpha_pack;   // host packing time per point
    double k_merge;      // time per task for GPU merge_local_bucket_sums kernel
    double k_suffix;     // time per bucket for GPU window_reduce_suffix kernel
    int    tpb;      // threads per block used by bucket_sum_block_per_bucket
    int    num_sms;  // number of GPU streaming multiprocessors
    size_t M_g;      // effective per-GPU memory budget used by planner
    size_t max_mem_per_gpu; // mirror budget for debug visibility
};

enum class Objective { Throughput, Latency };

constexpr size_t kFixedGpuReserveBytes = size_t(64) * size_t(1024);

inline size_t estimate_task_memory_bytes(int point_count, bool include_shard_id = true) {
  size_t points_bytes = size_t(std::max(0, point_count)) * sizeof(G1J);
  size_t bucket_id_bytes = sizeof(int);
  size_t local_output_bytes = sizeof(G1J);
  size_t offset_bytes = sizeof(int); // amortized per-task offset entry
  size_t shard_id_bytes = include_shard_id ? sizeof(int) : size_t(0);
  return points_bytes + bucket_id_bytes + local_output_bytes + offset_bytes + shard_id_bytes;
}

struct BucketTask {
  int bucket_idx = 0;
  int shard_idx = 0;
  int num_shards = 1;
  int point_begin = 0;
  int point_end = 0;
  int point_count = 0;
  size_t mem = 0;
  double t_assign = 0.0;
  bool is_split = false;
};


struct BucketPlan {
  // For each GPU g, explicit bucket tasks assigned to g.
  // A task may represent either a full bucket or one shard of a split bucket.
  std::vector<std::vector<BucketTask>> gpu_tasks;
  // Estimated time per GPU (T_{GPU,g,i}) used for scheduling decisions
  std::vector<double> gpu_estimated_time;
  // Component breakdown for each GPU estimate
  std::vector<double> gpu_h2d_time;
  std::vector<double> gpu_comp_time;
  std::vector<double> gpu_mem_time;
  // Estimated communication bytes per GPU batch.
  std::vector<size_t> gpu_h2d_bytes;
  std::vector<size_t> gpu_d2h_bytes;
  // Memory used per GPU (bytes) to enforce limits
  std::vector<size_t> gpu_memory;
  // Host-side and GPU stage estimates for the current window.
  double estimated_phase1_digit_time = 0.0;     // PHASE 1: kernel_compute_digits
  double estimated_phase1_count_time = 0.0;     // PHASE 1: kernel_count_buckets
  double estimated_host_pack_time = 0.0;
  double estimated_gpu_merge_time = 0.0;        // GPU merge_local_bucket_sums kernel
  double estimated_gpu_suffix_time = 0.0;       // GPU window_reduce_suffix kernel
  double estimated_staged_total_time = 0.0;     // Total: phase1 + pack + h2d + compute + gpu_merge + gpu_suffix + d2h
  // Estimated cost of the window reduction (T_win_red)
  double estimated_reduction_cost = 0.0;
};

inline size_t estimate_task_h2d_bytes(int point_count,
                                     bool include_task_id = true,
                                     bool include_shard_id = false) {
  size_t points_bytes = size_t(std::max(0, point_count)) * sizeof(G1J);
  size_t offsets_bytes = sizeof(int); // one amortized offset entry per task
  size_t task_id_bytes = include_task_id ? sizeof(int) : size_t(0);
  size_t shard_id_bytes = include_shard_id ? sizeof(int) : size_t(0);
  return points_bytes + offsets_bytes + task_id_bytes + shard_id_bytes;
}

inline size_t estimate_task_d2h_bytes() {
  return sizeof(G1J); // one local partial sum per task
}

inline double get_k_compute_for_wbits(int wbits, const SystemParams& params) {
  if (wbits <= 6) return params.k_compute_small;
  if (wbits <= 8) return params.k_compute_mid;
  return params.k_compute_large;
}

// Plan buckets across GPUs using an analytical cost model and optional constraints
// objective toggles throughput vs latency minimization; max_mem_per_gpu caps VRAM
// N: total number of scalars (needed for Phase 1 digit extraction and bucket counting costs)
inline BucketPlan make_plan(const std::vector<int>& bucket_sizes,
                           int G,
                           Objective objective,
                           const SystemParams& params,
                           size_t max_mem_per_gpu,
                           int N = 0,
                           int wbits = 8,
                           bool use_greedy = true) {
  // Conservative memory model for feasibility checks.
  // Includes per-task payload+metadata and one extra offsets tail entry per GPU.
  // Keep shard-id storage optional depending on whether runtime has a device
  // shard-id buffer in the active path.
  const bool include_device_shard_id = false;
  const bool include_device_task_id = false;
  const size_t per_gpu_reserve = std::min(kFixedGpuReserveBytes, max_mem_per_gpu);
  const size_t per_gpu_offset_tail_bytes = sizeof(int);

  BucketPlan plan;
  plan.gpu_tasks.assign(G, {});
  plan.gpu_estimated_time.assign(G, 0.0);
  plan.gpu_h2d_time.assign(G, 0.0);
  plan.gpu_comp_time.assign(G, 0.0);
  plan.gpu_mem_time.assign(G, 0.0);
  plan.gpu_h2d_bytes.assign(G, 0);
  plan.gpu_d2h_bytes.assign(G, 0);
  plan.gpu_memory.assign(G, 0);

  const int B = (int)bucket_sizes.size();
  if (B == 0 || G <= 0) return plan;

  std::vector<BucketTask> tasks;
  std::vector<char> gpu_has_tasks(G, 0);
  int nonempty_buckets = 0;

  // Compute time for one block of tpb threads summing sz points:
  //   ceil(sz / tpb) sequential adds per thread + log2(tpb) reduction steps.
  // When num_sms blocks run in parallel the per-bucket cost represents 1/num_sms
  // of the GPU's compute capacity, used as the scheduling weight.
  double k_comp = get_k_compute_for_wbits(wbits, params);  // wbits-aware correction factor
  auto block_compute_time = [&](int sz) -> double {
    int tpb = std::max(1, params.tpb);
    int ns  = std::max(1, params.num_sms);
    double adds_per_thread = std::ceil(double(sz) / tpb)
                           + std::log2(double(tpb)); // reduction
    // weight = time if this block ran alone, scaled down by num_sms to
    // reflect it shares the GPU with other concurrent blocks
    return adds_per_thread / (double(ns) * params.U_g * params.R_g);
  };

  auto shard_bucket = [&](int idx, int sz) {
    double m_bi = double(sz) * params.D_pt;
    size_t mem = estimate_task_memory_bytes(sz, include_device_shard_id);
    if (mem <= max_mem_per_gpu) {
      double h2d_bytes = double(estimate_task_h2d_bytes(sz, include_device_task_id, include_device_shard_id));
      double d2h_bytes = double(estimate_task_d2h_bytes());
      double t_h2d = params.L_h2d + h2d_bytes / params.B_link;
      double t_comp = k_comp * block_compute_time(sz);
      double t_mem  = params.L_d2h + d2h_bytes / params.B_link;
      double t_assign = t_h2d + t_comp + t_mem;
      BucketTask t;
      t.bucket_idx = idx;
      t.shard_idx = 0;
      t.num_shards = 1;
      t.point_begin = 0;
      t.point_end = sz;
      t.point_count = sz;
      t.mem = mem;
      t.t_assign = t_assign;
      t.is_split = false;
      tasks.push_back(t);
    } else {
      // Thesis semantics: split only when single-GPU placement is infeasible.
      int shards = (int)std::ceil(m_bi / double(max_mem_per_gpu));
      // Keep shards non-empty under aggressive debug caps.
      shards = std::max(1, std::min(shards, sz));
      const bool split_mode = (shards > 1);
      int base = sz / shards;
      int rem  = sz % shards;
      int begin = 0;
      for (int s = 0; s < shards; ++s) {
        int part = base + (s < rem ? 1 : 0);
        int end = begin + part;
        double h2d_bytes = double(estimate_task_h2d_bytes(part, include_device_task_id, include_device_shard_id));
        double d2h_bytes = double(estimate_task_d2h_bytes());
        double t_h2d = params.L_h2d + h2d_bytes / params.B_link;
        double t_comp = k_comp * block_compute_time(part);
        double t_mem  = params.L_d2h + d2h_bytes / params.B_link;
        double t_assign = t_h2d + t_comp + t_mem + params.L_sync; // merge overhead
        BucketTask t;
        t.bucket_idx = idx;
        t.shard_idx = s;
        t.num_shards = shards;
        t.point_begin = begin;
        t.point_end = end;
        t.point_count = part;
        t.mem = estimate_task_memory_bytes(part, include_device_shard_id);
        t.t_assign = t_assign;
        t.is_split = split_mode;
        tasks.push_back(t);
        begin = end;
      }
    }
  };

  for (int b = 0; b < B; ++b) {
    if (bucket_sizes[b] > 0) {
      shard_bucket(b, bucket_sizes[b]);
      ++nonempty_buckets;
    }
  }

  if (tasks.empty()) return plan;

  // Greedy mode sorts by descending assignment cost.
  if (use_greedy) {
    std::sort(tasks.begin(), tasks.end(), [](const BucketTask &a, const BucketTask &b) {
      return a.t_assign > b.t_assign;
    });
  }

  // helper for choosing GPU under objective
  auto choose_gpu = [&](const BucketTask &task) {
    int candidate = -1;
    double best_metric = 0.0;
    double current_max = 0.0;
    if (objective == Objective::Latency) {
      for (int g = 0; g < G; ++g) current_max = std::max(current_max, plan.gpu_estimated_time[g]);
    }
    for (int g = 0; g < G; ++g) {
      size_t projected = plan.gpu_memory[g] + task.mem;
      if (!gpu_has_tasks[g]) projected += per_gpu_offset_tail_bytes;
      projected += per_gpu_reserve;
      if (projected > max_mem_per_gpu) continue;
      double new_time = plan.gpu_estimated_time[g] + task.t_assign;
      double metric = (objective == Objective::Throughput)
                          ? new_time
                          : std::max(current_max, new_time);
      if (candidate < 0 || metric < best_metric) {
        candidate = g;
        best_metric = metric;
      }
    }
    if (candidate < 0) {
      // no feasible GPU; ignore memory for now
      candidate = 0;
      for (int g = 1; g < G; ++g)
        if (plan.gpu_estimated_time[g] < plan.gpu_estimated_time[candidate])
          candidate = g;
    }
    return candidate;
  };

  // Greedy assignment uses per-task scheduling weights (t_assign).
  // We accumulate them for bin-packing balance, then recompute the true
  // parallel wall-clock time after all tasks have been assigned.
  std::vector<double> gpu_sched_weight(G, 0.0);
  // Track per-GPU: total points, total bytes, max bucket size (for wave model)
  std::vector<int>    gpu_total_pts(G, 0);
  std::vector<int>    gpu_max_sz(G, 0);
  std::vector<int>    gpu_num_tasks(G, 0);
  std::vector<size_t> gpu_h2d_bytes_accum(G, 0);
  std::vector<size_t> gpu_d2h_bytes_accum(G, 0);

  auto assign_task_to_gpu = [&](const BucketTask& task, int g) {
    plan.gpu_tasks[g].push_back(task);
    gpu_sched_weight[g] += task.t_assign;
    if (!gpu_has_tasks[g]) {
      plan.gpu_memory[g] += per_gpu_reserve + per_gpu_offset_tail_bytes;
      gpu_has_tasks[g] = 1;
      gpu_h2d_bytes_accum[g] += per_gpu_offset_tail_bytes;
    }
    plan.gpu_memory[g] += task.mem;
    gpu_h2d_bytes_accum[g] += estimate_task_h2d_bytes(task.point_count,
                                                       include_device_task_id,
                                                       include_device_shard_id);
    gpu_d2h_bytes_accum[g] += estimate_task_d2h_bytes();
    gpu_total_pts[g] += task.point_count;
    gpu_max_sz[g] = std::max(gpu_max_sz[g], task.point_count);
    ++gpu_num_tasks[g];
  };

  if (use_greedy) {
    for (auto &task : tasks) {
      // temporarily store scheduling weight in gpu_estimated_time for choose_gpu
      plan.gpu_estimated_time = gpu_sched_weight;
      int g = choose_gpu(task);
      assign_task_to_gpu(task, g);
    }
  } else {
    // Static mode: round-robin assignment across GPUs in task order.
    int rr = 0;
    for (const auto& task : tasks) {
      int chosen = -1;
      for (int off = 0; off < G; ++off) {
        int g = (rr + off) % G;
        size_t projected = plan.gpu_memory[g] + task.mem;
        if (!gpu_has_tasks[g]) projected += per_gpu_offset_tail_bytes;
        projected += per_gpu_reserve;
        if (projected <= max_mem_per_gpu) {
          chosen = g;
          break;
        }
      }
      if (chosen < 0) {
        // If no GPU passes the cap, keep assignment deterministic by using RR target.
        chosen = rr % G;
      }
      assign_task_to_gpu(task, chosen);
      rr = (rr + 1) % G;
    }
  }

  // Recompute true parallel wall-clock estimate for each GPU:
  //   t_h2d   : transfer task payload + metadata bytes to GPU
  //   t_comp  : ceil(num_tasks / num_sms) waves × slowest-block work
  //   t_mem   : transfer local task sums back to host
  {
    int tpb = std::max(1, params.tpb);
    int ns  = std::max(1, params.num_sms);
    for (int g = 0; g < G; ++g) {
      plan.gpu_h2d_bytes[g] = gpu_h2d_bytes_accum[g];
      plan.gpu_d2h_bytes[g] = gpu_d2h_bytes_accum[g];
      double h2d_bytes = double(plan.gpu_h2d_bytes[g]);
      double d2h_bytes = double(plan.gpu_d2h_bytes[g]);
      double t_h2d  = (h2d_bytes > 0.0) ? (params.L_h2d + h2d_bytes / params.B_link) : 0.0;
      double t_mem  = (d2h_bytes > 0.0) ? (params.L_d2h + d2h_bytes / params.B_link) : 0.0;
      // slowest block in one wave
      double max_adds = std::ceil(double(gpu_max_sz[g]) / tpb)
                      + std::log2(double(tpb));
      // number of waves = ceil(num_tasks / num_sms)
      double waves = std::ceil(double(gpu_num_tasks[g]) / double(ns));
      double t_comp_model = waves * max_adds / (params.U_g * params.R_g);
      double t_comp = k_comp * t_comp_model;
      plan.gpu_h2d_time[g] = t_h2d;
      plan.gpu_comp_time[g] = t_comp;
      plan.gpu_mem_time[g] = t_mem;
      plan.gpu_estimated_time[g] = t_h2d + t_comp + t_mem;
    }
  }

  // reduction cost estimate based on number of GPUs used
  int used = 0;
  for (int g = 0; g < G; ++g) if (!plan.gpu_tasks[g].empty()) ++used;
  if (used <= 1) {
    plan.estimated_reduction_cost = params.L_sync;
  } else {
    // simple tree cost ~ L_sync * (1 + log2(used))
    plan.estimated_reduction_cost = params.L_sync * (1.0 + std::log2(double(used)));
  }

  // GPU-based pipeline cost model (Mar 31 refactor):
  // GPU merge and window reduction operations replace previous CPU merge and suffix stages
  
  int total_points = 0;
  int total_tasks = 0;
  std::vector<int> contributions_per_bucket(B, 0);
  for (int g = 0; g < G; ++g) {
    for (const auto& t : plan.gpu_tasks[g]) {
      total_points += t.point_count;
      ++contributions_per_bucket[t.bucket_idx];
      ++total_tasks;
    }
  }
  
  // New GPU-based stage costs (from SystemParams):
  // PHASE 1: GPU digit extraction and bucket counting on all N scalars
  plan.estimated_phase1_digit_time = params.k_digit * double(N);
  plan.estimated_phase1_count_time = params.k_count * double(N);
  
  // PHASE 2+: Host packing, H2D/D2H transfers, GPU compute, GPU merge, and GPU window reduction
  // k_merge: time per task for GPU merge_local_bucket_sums kernel
  // k_suffix: time per bucket for GPU window_reduce_suffix kernel
  
  plan.estimated_host_pack_time = params.alpha_pack * double(total_points);
  
  // PHASE 2+: Bucket sums, GPU merge, and GPU window reduction
  plan.estimated_gpu_merge_time = params.k_merge * double(total_tasks);
  plan.estimated_gpu_suffix_time = params.k_suffix * double(B);
  
  // Complete predicted time formula (GPU-based pipeline, all phases):
  // t_phase1_digit + t_phase1_count + t_pack + t_h2d + t_compute + t_gpu_merge + t_gpu_suffix + t_d2h
  double gpu_time_sum = std::accumulate(plan.gpu_estimated_time.begin(), plan.gpu_estimated_time.end(), 0.0);
  plan.estimated_staged_total_time =
      plan.estimated_phase1_digit_time
    + plan.estimated_phase1_count_time
    + plan.estimated_host_pack_time
    + gpu_time_sum                       // includes t_h2d + t_compute + t_d2h per GPU
    + plan.estimated_gpu_merge_time
    + plan.estimated_gpu_suffix_time;

  for (int g = 0; g < G; ++g) {
    std::sort(plan.gpu_tasks[g].begin(), plan.gpu_tasks[g].end(),
              [](const BucketTask& a, const BucketTask& b) {
                if (a.bucket_idx != b.bucket_idx) return a.bucket_idx < b.bucket_idx;
                return a.shard_idx < b.shard_idx;
              });
  }
  return plan;
}

