// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Experimental C ABI, intentionally independent of Python and the KV channel.
#include "seal_notifications.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
constexpr size_t capacity = 4096;
using Record = std::array<uint64_t, 9>;
uint64_t now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Job {
  Record r{};
  std::array<uint64_t, 22> audit{};
  cudaEvent_t start = nullptr, end = nullptr;
  bool end_recorded = false;
  ~Job() {
    if (start) cudaEventDestroy(start);
    if (end) cudaEventDestroy(end);
  }
};
class Observer {
  vllm::SealNotifier notifier;
  int fd = -1, audit_fd = -1, device;
  void* mem = MAP_FAILED;
  std::mutex mutex;
  std::condition_variable wake;
  bool stopping = false, failed = false;
  int mode = 0;  // 0: legacy full, 1: record only, 2: full retained
  bool collect_audits = false;
  uint64_t query_count = 0, first_sample = 0, last_sample = 0, max_gap = 0;

  using Jobs = std::map<uint64_t, Job>;
  Jobs jobs;
  std::vector<Jobs::node_type> reusable, retained;
  std::vector<std::array<uint64_t, 22>> audits;
  std::thread thread;
  void write(const Record& r, Job* job = nullptr) {
    auto begin = now();
    if (flock(fd, LOCK_EX)) throw std::runtime_error("timing flock failed");
    auto acquired = now();
    std::memcpy(static_cast<char*>(mem) + (r[0] % capacity) * sizeof(Record),
                r.data(), sizeof(Record));
    if (flock(fd, LOCK_UN)) throw std::runtime_error("timing unlock failed");
    if (r[5] == 0 || r[8]) notifier.compute(r[0], r[5] != 0);
    if (job) {
      job->audit[11] = std::max(job->audit[11], acquired - begin);
      job->audit[12] = std::max(job->audit[12], now() - begin);
    }
  }
  void fail() noexcept {
    failed = true;
    for (auto& [id, job] : jobs) {
      job.r[5] = 0;
      try {
        write(job.r, &job);
      } catch (...) {
      }
    }
    jobs.clear();
  }
  void poll() noexcept {
    pthread_setname_np(pthread_self(), "vllm-gpu-time");
    std::unique_lock<std::mutex> lock(mutex);
    try {
      check(cudaSetDevice(device));
      while (!stopping) {
        if (jobs.empty()) {
          wake.wait(lock, [&] { return stopping || !jobs.empty(); });
          continue;
        }
        for (auto it = jobs.begin(); it != jobs.end();) {
          auto& job = it->second;
          auto& r = job.r;
          auto sample = now();
          if (!first_sample) first_sample = sample;
          if (last_sample) max_gap = std::max(max_gap, sample - last_sample);
          last_sample = sample;
          if (job.audit[18])
            job.audit[19] = std::max(job.audit[19], sample - job.audit[18]);
          job.audit[18] = sample;
          if (!r[3]) {
            auto before = now();
            if (!job.audit[6]) job.audit[6] = before;
            ++query_count;
            ++job.audit[20];
            auto state = cudaEventQuery(job.start);
            auto after = now();
            if (state == cudaSuccess) {
              r[3] = after;
              job.audit[9] = before;
              job.audit[10] = after;
            } else {
              job.audit[7] = before;
              job.audit[8] = after;
              check(state == cudaErrorNotReady ? cudaSuccess : state);
              r[2] = before;
            }
          }
          if (r[3] && job.end_recorded) {
            auto before = now();
            ++query_count;
            ++job.audit[20];
            auto state = cudaEventQuery(job.end);
            if (state == cudaSuccess) {
              r[8] = now();
              r[7] = std::max(r[7], r[2]);
              float elapsed;
              check(cudaEventElapsedTime(&elapsed, job.start, job.end));
              r[6] = static_cast<uint64_t>(elapsed * 1000000.0 + 0.5);
            } else {
              check(state == cudaErrorNotReady ? cudaSuccess : state);
              r[7] = before;
            }
          }
          r[4] = now();
          write(r, &job);
          if (r[8]) {
            job.audit[17] = r[8];
            // Ordinary serving has no audit consumer. Retaining these records
            // would permanently disable timing after 4096 completed forwards.
            // Diagnostic modes keep their existing bounded audit contract.
            if (collect_audits || audit_fd >= 0) {
              if (audits.size() >= capacity)
                throw std::runtime_error("audit overflow");
              audits.push_back(job.audit);
            }
            if (audit_fd >= 0 &&
                ::write(audit_fd, job.audit.data(), 18 * sizeof(uint64_t)) !=
                    18 * sizeof(uint64_t))
              throw std::runtime_error("native timing audit write failed");
            // Event destruction can block on unrelated CUDA activity. Reuse
            // only after completion and publication; destroy at observer
            // shutdown.
            auto completed = it++;
            (mode == 2 ? retained : reusable)
                .push_back(jobs.extract(completed));
          } else
            ++it;
        }
        wake.wait_for(lock, std::chrono::microseconds(500));
      }
      if (!jobs.empty()) fail();
    } catch (...) {
      fail();
    }
  }

 public:
  size_t audit_copy(uint64_t* out, size_t limit) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (limit < audits.size() * 22) return 0;
    std::memcpy(out, audits.data(), audits.size() * sizeof(audits[0]));
    return audits.size();
  }
  int snapshot(uint64_t* out) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    out[0] = query_count;
    out[1] = first_sample;
    out[2] = last_sample;
    out[3] = max_gap;
    out[4] = failed;
    out[5] = jobs.size();
    out[6] = retained.size();
    out[7] = reusable.size();
    return failed ? 0 : 1;
  }
  int drain() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    try {
      check(cudaSetDevice(device));
      // Called only after the engine and device have drained, outside replay.
      for (auto& [id, job] : jobs) {
        if (!job.end_recorded)
          throw std::runtime_error("unfinished step at drain");
        check(cudaEventSynchronize(job.end));
        float elapsed;
        check(cudaEventElapsedTime(&elapsed, job.start, job.end));
        if (mode == 1) {
          if (audits.size() >= capacity)
            throw std::runtime_error("audit overflow");
          job.audit[21] = now();
          audits.push_back(job.audit);
        }
        // Offline observation is deliberately not a live host completion time.
        job.r[6] = static_cast<uint64_t>(elapsed * 1000000.0 + 0.5);
        write(job.r, &job);
      }
      return failed ? 0 : 1;
    } catch (...) {
      fail();
      return 0;
    }
  }
  Observer(const char* path, int dev, const char* audit_path, int selected = 0,
           bool audit_enabled = false)
      : notifier(path),
        device(dev),
        mode(selected),
        collect_audits(audit_enabled) {
    fd = open(path, O_RDWR | O_CLOEXEC);
    struct stat st{};
    if (fd < 0) throw std::runtime_error("timing open failed");
    if (fstat(fd, &st) || st.st_size != capacity * sizeof(Record)) {
      close(fd);
      throw std::runtime_error("invalid timing file size");
    }
    mem = mmap(nullptr, capacity * sizeof(Record), PROT_READ | PROT_WRITE,
               MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED) {
      close(fd);
      throw std::runtime_error("timing mmap failed");
    }
    try {
      if (audit_path && *audit_path) {
        audit_fd =
            open(audit_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (audit_fd < 0)
          throw std::runtime_error("native timing audit open failed");
      }
      check(cudaSetDevice(device));
      audits.reserve(capacity);
      if (mode) {
        reusable.reserve(capacity);
        retained.reserve(capacity);
        for (size_t i = 1; i <= capacity; ++i) {
          auto& job = jobs[i];
          check(cudaEventCreate(&job.start));
          check(cudaEventCreate(&job.end));
        }
        while (!jobs.empty()) reusable.push_back(jobs.extract(jobs.begin()));
      }
      if (mode != 1) thread = std::thread(&Observer::poll, this);
    } catch (...) {
      munmap(mem, capacity * sizeof(Record));
      close(fd);
      if (audit_fd >= 0) close(audit_fd);
      throw;
    }
  }
  ~Observer() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      wake.notify_all();
    }
    if (thread.joinable()) thread.join();
    if (audit_fd >= 0) close(audit_fd);
    munmap(mem, capacity * sizeof(Record));
    close(fd);
  }
  int start(uint64_t step, uintptr_t stream) noexcept {
    auto entered = now();
    std::lock_guard<std::mutex> lock(mutex);
    auto acquired = now();
    if (failed || stopping) return 0;
    try {
      if (!step || jobs.size() >= capacity || jobs.count(step) ||
          (mode && reusable.empty()))
        throw std::runtime_error("duplicate or overflow");
      for (const auto& [id, job] : jobs) {
        if (id % capacity == step % capacity)
          throw std::runtime_error("active timing slot collision");
      }
      check(cudaSetDevice(device));
      if (!reusable.empty()) {
        auto node = std::move(reusable.back());
        reusable.pop_back();
        node.key() = step;
        jobs.insert(std::move(node));
      }
      auto& job = jobs[step];
      job.audit = {};
      job.end_recorded = false;
      job.audit[0] = step;
      job.audit[1] = entered;
      job.audit[2] = acquired;
      job.r = {step, 0, 0, 0, 0, 1, 0, 0, 0};
      job.audit[3] = now();
      if (!job.start) check(cudaEventCreate(&job.start));
      job.audit[4] = now();
      job.r[1] = job.r[2] = now();
      check(cudaEventRecord(job.start, reinterpret_cast<cudaStream_t>(stream)));
      job.audit[5] = now();
      job.r[4] = now();
      write(job.r, &job);
      wake.notify_all();
      return 1;
    } catch (...) {
      fail();
      return 0;
    }
  }
  int finish(uint64_t step, uintptr_t stream) noexcept {
    auto entered = now();
    std::lock_guard<std::mutex> lock(mutex);
    auto acquired = now();
    if (failed || stopping) return 0;
    try {
      auto& job = jobs.at(step);
      if (job.end_recorded) throw std::runtime_error("duplicate finish");
      job.audit[13] = entered;
      job.audit[14] = acquired;
      check(cudaSetDevice(device));
      auto before = now();
      if (!job.end) check(cudaEventCreate(&job.end));
      job.audit[15] = now() - before;
      job.r[7] = now();
      check(cudaEventRecord(job.end, reinterpret_cast<cudaStream_t>(stream)));
      job.audit[16] = now() - job.r[7];
      job.end_recorded = true;
      wake.notify_all();
      return 1;
    } catch (...) {
      fail();
      return 0;
    }
  }
};
}  // namespace
extern "C" {
int timing_abi_version() noexcept { return 2; }
int timing_seal_notification_abi() noexcept { return 1; }
void* timing_create(const char* path, int device,
                    const char* audit_path) noexcept {
  try {
    return new Observer(path, device, audit_path);
  } catch (...) {
    return nullptr;
  }
}
void* timing_create_mode(const char* path, int device, int mode) noexcept {
  if (mode < 0 || mode > 2) return nullptr;
  try {
    return new Observer(path, device, "", mode, true);
  } catch (...) {
    return nullptr;
  }
}
size_t timing_audit_copy(void* p, uint64_t* out, size_t limit) noexcept {
  return static_cast<Observer*>(p)->audit_copy(out, limit);
}
int timing_drain(void* p) noexcept {
  return static_cast<Observer*>(p)->drain();
}
int timing_snapshot(void* p, uint64_t* out) noexcept {
  return static_cast<Observer*>(p)->snapshot(out);
}
int timing_start(void* p, uint64_t step, uintptr_t stream) noexcept {
  return static_cast<Observer*>(p)->start(step, stream);
}
int timing_finish(void* p, uint64_t step, uintptr_t stream) noexcept {
  return static_cast<Observer*>(p)->finish(step, stream);
}
void timing_destroy(void* p) noexcept { delete static_cast<Observer*>(p); }
}
