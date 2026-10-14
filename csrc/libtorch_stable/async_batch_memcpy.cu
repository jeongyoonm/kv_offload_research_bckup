#include "ops.h"
#include "seal_notifications.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(USE_ROCM)
  #include <cuda.h>
  #include <nvtx3/nvToolsExt.h>
#endif

namespace {

constexpr uint64_t kCompletionRingMagic = 0x564c4c4d4b565231ULL;

struct alignas(64) CompletionRingHeader {
  uint64_t magic;
  uint64_t session;
  uint64_t capacity;
  uint64_t write_pos;
  uint64_t read_pos;
  uint32_t overflow;
  uint32_t failure;
  uint8_t padding[16];
};
static_assert(sizeof(CompletionRingHeader) == 64);

struct CompletionRingRecord {
  uint64_t session;
  int64_t job_id;
  uint64_t completion_ns;
  int32_t status;
  uint32_t reserved;
};
static_assert(sizeof(CompletionRingRecord) == 32);

class CompletionRing {
 public:
  CompletionRing(std::string path, int64_t session, int64_t capacity,
                 bool owner)
      : path_(std::move(path)), owner_(owner) {
    STD_TORCH_CHECK(session > 0, "completion ring session must be positive");
    STD_TORCH_CHECK(capacity > 0, "completion ring capacity must be positive");
    if (!owner_) notifier_ = std::make_unique<vllm::SealNotifier>(path_);
    const int flags = owner ? O_RDWR | O_CREAT | O_EXCL : O_RDWR;
    fd_ = open(path_.c_str(), flags, 0600);
    STD_TORCH_CHECK(fd_ >= 0, "failed to open completion ring ", path_, ": ",
                    std::strerror(errno));
    mapped_size_ = sizeof(CompletionRingHeader) +
                   static_cast<size_t>(capacity) * sizeof(CompletionRingRecord);
    if (owner) {
      if (ftruncate(fd_, static_cast<off_t>(mapped_size_)) != 0) {
        const std::string error = std::strerror(errno);
        close(fd_);
        fd_ = -1;
        unlink(path_.c_str());
        STD_TORCH_CHECK(false, "failed to size completion ring ", path_, ": ",
                        error);
      }
    }
    mapping_ =
        mmap(nullptr, mapped_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      const std::string error = std::strerror(errno);
      close(fd_);
      fd_ = -1;
      if (owner_) unlink(path_.c_str());
      STD_TORCH_CHECK(false, "failed to map completion ring ", path_, ": ",
                      error);
    }
    header_ = static_cast<CompletionRingHeader*>(mapping_);
    records_ = reinterpret_cast<CompletionRingRecord*>(header_ + 1);
    if (owner) {
      std::memset(mapping_, 0, mapped_size_);
      header_->session = static_cast<uint64_t>(session);
      header_->capacity = static_cast<uint64_t>(capacity);
      __atomic_store_n(&header_->magic, kCompletionRingMagic, __ATOMIC_RELEASE);
    } else {
      const uint64_t magic = __atomic_load_n(&header_->magic, __ATOMIC_ACQUIRE);
      STD_TORCH_CHECK(magic == kCompletionRingMagic,
                      "invalid completion ring magic for ", path_);
      STD_TORCH_CHECK(header_->session == static_cast<uint64_t>(session),
                      "completion ring session mismatch for ", path_);
      STD_TORCH_CHECK(header_->capacity == static_cast<uint64_t>(capacity),
                      "completion ring capacity mismatch for ", path_);
    }
  }

  ~CompletionRing() {
    if (mapping_ != nullptr) munmap(mapping_, mapped_size_);
    if (fd_ >= 0) close(fd_);
    if (owner_) unlink(path_.c_str());
  }

  void fail() {
    __atomic_store_n(&header_->failure, 1U, __ATOMIC_RELEASE);
    if (notifier_) notifier_->kv(-1, false);
  }

  void publish(int64_t job_id, bool success) {
    const uint64_t write =
        __atomic_load_n(&header_->write_pos, __ATOMIC_RELAXED);
    const uint64_t read = __atomic_load_n(&header_->read_pos, __ATOMIC_ACQUIRE);
    if (write - read >= header_->capacity) {
      __atomic_store_n(&header_->overflow, 1U, __ATOMIC_RELEASE);
      if (notifier_) notifier_->kv(job_id, false);
      throw std::runtime_error("async completion ring overflow for " + path_);
    }
    auto& record = records_[write % header_->capacity];
    record.session = header_->session;
    record.job_id = job_id;
    record.completion_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    record.status = success ? 1 : 0;
    record.reserved = 0;
#if !defined(USE_ROCM)
    const char* trace_flag = std::getenv("VLLM_KV_READINESS_GPU_TRACE");
    if (trace_flag != nullptr && std::string(trace_flag) == "1") {
      const auto label = "kvready|publish|" + std::to_string(job_id) + "|" +
                         std::to_string(record.completion_ns) + "|" +
                         std::to_string(record.status);
      nvtxMarkA(label.c_str());
    }
#endif
    if (!success) {
      __atomic_store_n(&header_->failure, 1U, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&header_->write_pos, write + 1, __ATOMIC_RELEASE);
    if (notifier_) notifier_->kv(job_id, success);
  }

  std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>,
             std::vector<int64_t>, bool, bool>
  poll() {
    std::vector<int64_t> sessions;
    std::vector<int64_t> job_ids;
    std::vector<int64_t> completion_ns;
    std::vector<int64_t> statuses;
    uint64_t read = __atomic_load_n(&header_->read_pos, __ATOMIC_RELAXED);
    const uint64_t write =
        __atomic_load_n(&header_->write_pos, __ATOMIC_ACQUIRE);
    sessions.reserve(write - read);
    job_ids.reserve(write - read);
    completion_ns.reserve(write - read);
    statuses.reserve(write - read);
    while (read < write) {
      const auto& record = records_[read % header_->capacity];
      sessions.push_back(static_cast<int64_t>(record.session));
      job_ids.push_back(record.job_id);
      completion_ns.push_back(static_cast<int64_t>(record.completion_ns));
      statuses.push_back(record.status);
      ++read;
    }
    __atomic_store_n(&header_->read_pos, read, __ATOMIC_RELEASE);
    return {std::move(sessions),
            std::move(job_ids),
            std::move(completion_ns),
            std::move(statuses),
            __atomic_load_n(&header_->overflow, __ATOMIC_ACQUIRE) != 0,
            __atomic_load_n(&header_->failure, __ATOMIC_ACQUIRE) != 0};
  }

 private:
  std::string path_;
  bool owner_;
  std::unique_ptr<vllm::SealNotifier> notifier_;
  int fd_ = -1;
  void* mapping_ = nullptr;
  size_t mapped_size_ = 0;
  CompletionRingHeader* header_ = nullptr;
  CompletionRingRecord* records_ = nullptr;
};

std::mutex completion_ring_registry_mutex;
std::unordered_map<int64_t, std::shared_ptr<CompletionRing>>
    completion_ring_registry;
int64_t next_completion_ring_handle = 1;

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080

using BatchFn = CUresult (*)(CUdeviceptr*, CUdeviceptr*, size_t*, size_t,
                             CUmemcpyAttributes*, size_t*, size_t, size_t*,
                             CUstream);

std::string cuda_runtime_error(cudaError_t error) {
  const char* name = cudaGetErrorName(error);
  const char* message = cudaGetErrorString(error);
  return std::string(name == nullptr ? "unknown CUDA runtime error" : name) +
         ": " + (message == nullptr ? "unknown error" : message);
}

std::string cuda_driver_error(CUresult error) {
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(error, &name);
  cuGetErrorString(error, &message);
  std::ostringstream stream;
  stream << (name == nullptr ? "unknown CUDA driver error" : name);
  if (message != nullptr) stream << ": " << message;
  return stream.str();
}

struct AsyncBatchMemcpyJob {
  int64_t job_id;
  torch::stable::Tensor src_ptrs;
  torch::stable::Tensor dst_ptrs;
  torch::stable::Tensor sizes;
  bool src_access_any;
  cudaEvent_t ready_event = nullptr;
  bool finished = false;
  double transfer_duration_ms = 0.0;

  AsyncBatchMemcpyJob(int64_t job_id, const torch::stable::Tensor& src_ptrs,
                      const torch::stable::Tensor& dst_ptrs,
                      const torch::stable::Tensor& sizes, bool src_access_any)
      : job_id(job_id),
        src_ptrs(src_ptrs),
        dst_ptrs(dst_ptrs),
        sizes(sizes),
        src_access_any(src_access_any) {}
  ~AsyncBatchMemcpyJob() {
    if (ready_event != nullptr) cudaEventDestroy(ready_event);
  }
};

class AsyncBatchMemcpySubmitter {
 public:
  AsyncBatchMemcpySubmitter(int device_index, int64_t chunk_bytes,
                            int64_t max_descriptors,
                            const std::string& completion_ring_path,
                            int64_t completion_session)
      : device_index_(device_index),
        chunk_bytes_(chunk_bytes),
        max_descriptors_(max_descriptors),
        batch_fn_(resolve_batch_fn()),
        completion_ring_(
            completion_ring_path.empty()
                ? nullptr
                : std::make_unique<CompletionRing>(
                      completion_ring_path, completion_session, 4096, false)) {
    worker_ = std::thread(&AsyncBatchMemcpySubmitter::worker_main, this);
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return initialized_ || !error_.empty(); });
    if (!error_.empty()) {
      lock.unlock();
      if (worker_.joinable()) worker_.join();
      STD_TORCH_CHECK(false, error_);
    }
  }

  ~AsyncBatchMemcpySubmitter() { shutdown(false); }

  void configure_pipeline(int64_t depth) {
    STD_TORCH_CHECK(depth >= 1 && depth <= 8,
                    "copy pipeline depth must be between 1 and 8");
    std::lock_guard<std::mutex> lock(mutex_);
    check_error_locked();
    if (pipeline_depth_.load() == depth) return;
    STD_TORCH_CHECK(active_jobs_.empty(),
                    "copy pipeline changes require drained jobs");
    pipeline_depth_.store(depth);
  }

  void submit(int64_t job_id, const torch::stable::Tensor& src_ptrs,
              const torch::stable::Tensor& dst_ptrs,
              const torch::stable::Tensor& sizes, bool src_access_any,
              int64_t compute_stream) {
    validate_descriptors(src_ptrs, dst_ptrs, sizes);
    auto job = std::make_shared<AsyncBatchMemcpyJob>(job_id, src_ptrs, dst_ptrs,
                                                     sizes, src_access_any);
    auto status =
        cudaEventCreateWithFlags(&job->ready_event, cudaEventDisableTiming);
    STD_TORCH_CHECK(status == cudaSuccess, cuda_runtime_error(status));
    status = cudaEventRecord(job->ready_event,
                             reinterpret_cast<cudaStream_t>(compute_stream));
    STD_TORCH_CHECK(status == cudaSuccess, cuda_runtime_error(status));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      check_error_locked();
      STD_TORCH_CHECK(!stopping_,
                      "async batch memcpy submitter is shutting down");
      STD_TORCH_CHECK(active_jobs_.find(job_id) == active_jobs_.end(),
                      "duplicate async batch memcpy job ID ", job_id);
      active_jobs_.emplace(job_id, job);
      work_queue_.push_back(job);
      completion_queue_.push_back(std::move(job));
    }
    condition_.notify_all();
  }

  std::tuple<std::vector<int64_t>, std::vector<double>> poll() {
    std::vector<int64_t> job_ids;
    std::vector<double> transfer_durations_ms;
    std::lock_guard<std::mutex> lock(mutex_);
    check_error_locked();
    while (!completion_queue_.empty() && completion_queue_.front()->finished) {
      auto job = std::move(completion_queue_.front());
      completion_queue_.pop_front();
      active_jobs_.erase(job->job_id);
      job_ids.push_back(job->job_id);
      transfer_durations_ms.push_back(job->transfer_duration_ms);
    }
    return {std::move(job_ids), std::move(transfer_durations_ms)};
  }

  void wait(const std::vector<int64_t>& job_ids) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (int64_t job_id : job_ids) {
      auto it = active_jobs_.find(job_id);
      if (it == active_jobs_.end()) continue;
      auto job = it->second;
      condition_.wait(
          lock, [this, &job] { return job->finished || !error_.empty(); });
      check_error_locked();
    }
  }

  void shutdown(bool raise_error = true) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (raise_error) {
      std::lock_guard<std::mutex> lock(mutex_);
      check_error_locked();
    }
  }

 private:
  static BatchFn resolve_batch_fn() {
    CUresult result = cuInit(0);
    STD_TORCH_CHECK(result == CUDA_SUCCESS,
                    "cuInit failed while creating async batch memcpy "
                    "submitter: ",
                    cuda_driver_error(result));
    int driver_version = 0;
    result = cuDriverGetVersion(&driver_version);
    STD_TORCH_CHECK(result == CUDA_SUCCESS,
                    "cuDriverGetVersion failed while creating async batch "
                    "memcpy submitter: ",
                    cuda_driver_error(result));
    STD_TORCH_CHECK(
        driver_version >= 12080,
        "async_h2d_submission requires CUDA driver 12.8 or newer; found ",
        driver_version / 1000, ".", (driver_version % 1000) / 10);

    CUdriverProcAddressQueryResult symbol_status;
    void* fn_ptr = nullptr;
    result = cuGetProcAddress("cuMemcpyBatchAsync", &fn_ptr, 12080,
                              CU_GET_PROC_ADDRESS_DEFAULT, &symbol_status);
    STD_TORCH_CHECK(result == CUDA_SUCCESS && fn_ptr != nullptr,
                    "async_h2d_submission requires the CUDA driver symbol "
                    "cuMemcpyBatchAsync (CUDA 12.8+); lookup failed: ",
                    cuda_driver_error(result));
    return reinterpret_cast<BatchFn>(fn_ptr);
  }

  static void validate_descriptors(const torch::stable::Tensor& src_ptrs,
                                   const torch::stable::Tensor& dst_ptrs,
                                   const torch::stable::Tensor& sizes) {
    STD_TORCH_CHECK(src_ptrs.device().is_cpu(), "src_ptrs must be on CPU");
    STD_TORCH_CHECK(dst_ptrs.device().is_cpu(), "dst_ptrs must be on CPU");
    STD_TORCH_CHECK(sizes.device().is_cpu(), "sizes must be on CPU");
    STD_TORCH_CHECK(
        src_ptrs.scalar_type() == torch::headeronly::ScalarType::Long,
        "src_ptrs must be int64");
    STD_TORCH_CHECK(
        dst_ptrs.scalar_type() == torch::headeronly::ScalarType::Long,
        "dst_ptrs must be int64");
    STD_TORCH_CHECK(sizes.scalar_type() == torch::headeronly::ScalarType::Long,
                    "sizes must be int64");
    STD_TORCH_CHECK(
        src_ptrs.dim() == 1 && dst_ptrs.dim() == 1 && sizes.dim() == 1,
        "async batch memcpy descriptors must be 1-D");
    STD_TORCH_CHECK(src_ptrs.is_contiguous() && dst_ptrs.is_contiguous() &&
                        sizes.is_contiguous(),
                    "async batch memcpy descriptors must be contiguous");
    STD_TORCH_CHECK(dst_ptrs.size(0) == src_ptrs.size(0),
                    "dst_ptrs length must match src_ptrs");
    STD_TORCH_CHECK(sizes.size(0) == src_ptrs.size(0),
                    "sizes length must match src_ptrs");
  }

  void check_error_locked() const {
    STD_TORCH_CHECK(error_.empty(),
                    "async batch memcpy worker failed: ", error_);
  }

  void set_error(std::string error) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (error_.empty()) error_ = std::move(error);
      stopping_ = true;
    }
    if (completion_ring_) {
      try {
        completion_ring_->fail();
      } catch (...) {
      }
    }
    condition_.notify_all();
  }

  void worker_main() noexcept {
    try {
      cudaError_t result = cudaSetDevice(device_index_);
      if (result != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice(" +
                                 std::to_string(device_index_) +
                                 ") failed: " + cuda_runtime_error(result));
      }
      result = cudaFree(nullptr);
      if (result != cudaSuccess) {
        throw std::runtime_error("CUDA context initialization failed: " +
                                 cuda_runtime_error(result));
      }
      result = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
      if (result != cudaSuccess) {
        throw std::runtime_error("cudaStreamCreateWithFlags failed: " +
                                 cuda_runtime_error(result));
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        initialized_ = true;
      }
      condition_.notify_all();

      while (true) {
        std::shared_ptr<AsyncBatchMemcpyJob> job;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          condition_.wait(lock,
                          [this] { return stopping_ || !work_queue_.empty(); });
          if (work_queue_.empty()) {
            if (stopping_) break;
            continue;
          }
          job = std::move(work_queue_.front());
          work_queue_.pop_front();
        }
        try {
          run_job(*job);
          if (completion_ring_ != nullptr) {
            completion_ring_->publish(job->job_id, true);
          }
        } catch (...) {
          if (completion_ring_ != nullptr) {
            try {
              completion_ring_->publish(job->job_id, false);
            } catch (...) {
            }
          }
          throw;
        }
        {
          std::lock_guard<std::mutex> lock(mutex_);
          job->finished = true;
        }
        condition_.notify_all();
      }
    } catch (const std::exception& error) {
      set_error(error.what());
    } catch (...) {
      set_error("unknown C++ exception");
    }

    if (stream_ != nullptr) {
      cudaError_t result = cudaStreamDestroy(stream_);
      if (result != cudaSuccess) {
        set_error("cudaStreamDestroy failed during shutdown: " +
                  cuda_runtime_error(result));
      }
      stream_ = nullptr;
    }
  }

  int64_t find_chunk_end(AsyncBatchMemcpyJob& job, int64_t first,
                         int64_t count) const {
    auto* size_data = job.sizes.mutable_data_ptr<int64_t>();
    const int64_t descriptor_limit =
        max_descriptors_ == 0 ? count
                              : std::min(count, first + max_descriptors_);
    int64_t chunk_size = 0;
    int64_t end = first;
    while (end < descriptor_limit) {
      const int64_t descriptor_size = size_data[end];
      STD_TORCH_CHECK(descriptor_size >= 0,
                      "async batch memcpy sizes must be nonnegative");
      if (end > first && descriptor_size > chunk_bytes_ - chunk_size) break;
      chunk_size += descriptor_size;
      ++end;
    }
    return end;
  }

  void submit_chunk(AsyncBatchMemcpyJob& job, int64_t first, int64_t count) {
    auto* src_data = job.src_ptrs.mutable_data_ptr<int64_t>();
    auto* dst_data = job.dst_ptrs.mutable_data_ptr<int64_t>();
    auto* size_data = job.sizes.mutable_data_ptr<int64_t>();
    static_assert(sizeof(CUdeviceptr) == sizeof(int64_t));
    static_assert(sizeof(size_t) == sizeof(int64_t));
    CUmemcpyAttributes attributes = {};
    attributes.srcAccessOrder = job.src_access_any
                                    ? CU_MEMCPY_SRC_ACCESS_ORDER_ANY
                                    : CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    size_t attributes_index = 0;
    size_t fail_index = 0;
    CUresult result =
        batch_fn_(reinterpret_cast<CUdeviceptr*>(dst_data + first),
                  reinterpret_cast<CUdeviceptr*>(src_data + first),
                  reinterpret_cast<size_t*>(size_data + first),
                  static_cast<size_t>(count), &attributes, &attributes_index, 1,
                  &fail_index, static_cast<CUstream>(stream_));
    if (result != CUDA_SUCCESS) {
      throw std::runtime_error(
          "cuMemcpyBatchAsync failed for job " + std::to_string(job.job_id) +
          " at descriptor " +
          std::to_string(first + static_cast<int64_t>(fail_index)) + ": " +
          cuda_driver_error(result));
    }
  }

  void run_job(AsyncBatchMemcpyJob& job) {
    struct TraceRange {
      bool enabled;
      explicit TraceRange(int64_t job_id) {
        const char* flag = std::getenv("VLLM_KV_READINESS_GPU_TRACE");
        enabled = flag != nullptr && std::string(flag) == "1";
        if (enabled) {
          const auto label = "kvload|async|" + std::to_string(job_id);
          nvtxRangePushA(label.c_str());
        }
      }
      ~TraceRange() {
        if (enabled) nvtxRangePop();
      }
    } trace_range(job.job_id);
    auto ready_status = cudaStreamWaitEvent(stream_, job.ready_event, 0);
    STD_TORCH_CHECK(ready_status == cudaSuccess,
                    cuda_runtime_error(ready_status));
    const int64_t count = job.src_ptrs.size(0);
    if (count == 0) return;
    const int64_t depth = pipeline_depth_.load();
    if (depth > 1) {
      run_pipelined_chunks(job, count, depth);
      return;
    }

    cudaEvent_t start_event = nullptr;
    cudaEvent_t end_event = nullptr;
    cudaError_t result = cudaEventCreate(&start_event);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaEventCreate(start) failed: " +
                               cuda_runtime_error(result));
    }
    result = cudaEventCreate(&end_event);
    if (result != cudaSuccess) {
      cudaEventDestroy(start_event);
      throw std::runtime_error("cudaEventCreate(end) failed: " +
                               cuda_runtime_error(result));
    }

    try {
      int64_t first = 0;
      while (first < count) {
        const int64_t end = find_chunk_end(job, first, count);
        result = cudaEventRecord(start_event, stream_);
        if (result != cudaSuccess) {
          throw std::runtime_error("cudaEventRecord(start) failed: " +
                                   cuda_runtime_error(result));
        }
        submit_chunk(job, first, end - first);
        result = cudaEventRecord(end_event, stream_);
        if (result != cudaSuccess) {
          throw std::runtime_error("cudaEventRecord(end) failed: " +
                                   cuda_runtime_error(result));
        }
        result = cudaEventSynchronize(end_event);
        if (result != cudaSuccess) {
          throw std::runtime_error("cudaEventSynchronize(chunk) failed: " +
                                   cuda_runtime_error(result));
        }
        float elapsed_ms = 0.0F;
        result = cudaEventElapsedTime(&elapsed_ms, start_event, end_event);
        if (result != cudaSuccess) {
          throw std::runtime_error("cudaEventElapsedTime(chunk) failed: " +
                                   cuda_runtime_error(result));
        }
        job.transfer_duration_ms += elapsed_ms;
        first = end;
      }
    } catch (...) {
      cudaEventDestroy(end_event);
      cudaEventDestroy(start_event);
      throw;
    }
    cudaEventDestroy(end_event);
    cudaEventDestroy(start_event);
  }

  void run_pipelined_chunks(AsyncBatchMemcpyJob& job, int64_t count,
                            int64_t depth) {
    struct Events {
      cudaEvent_t start = nullptr;
      cudaEvent_t end = nullptr;
      bool pending = false;
      ~Events() {
        if (end != nullptr) cudaEventDestroy(end);
        if (start != nullptr) cudaEventDestroy(start);
      }
    };
    std::vector<Events> slots(static_cast<size_t>(depth));
    auto checked = [](cudaError_t status) {
      if (status != cudaSuccess) {
        throw std::runtime_error(cuda_runtime_error(status));
      }
    };
    auto retire = [&](Events& slot) {
      if (!slot.pending) return;
      checked(cudaEventSynchronize(slot.end));
      float elapsed_ms = 0.0F;
      checked(cudaEventElapsedTime(&elapsed_ms, slot.start, slot.end));
      job.transfer_duration_ms += elapsed_ms;
      slot.pending = false;
    };
    for (auto& slot : slots) {
      checked(cudaEventCreate(&slot.start));
      checked(cudaEventCreate(&slot.end));
    }
    int64_t sequence = 0;
    try {
      for (int64_t first = 0; first < count; ++sequence) {
        auto& slot = slots[static_cast<size_t>(sequence % depth)];
        retire(slot);
        const int64_t end = find_chunk_end(job, first, count);
        checked(cudaEventRecord(slot.start, stream_));
        submit_chunk(job, first, end - first);
        checked(cudaEventRecord(slot.end, stream_));
        slot.pending = true;
        first = end;
      }
      // Complete every queued chunk before publishing readiness or freeing
      // descriptors. Chunks and jobs remain FIFO on the original copy stream.
      for (int64_t i = 0; i < depth; ++i) {
        retire(slots[static_cast<size_t>((sequence + i) % depth)]);
      }
    } catch (...) {
      // Already-enqueued copies retain descriptor/buffer ownership until this
      // stream drains, even when a later chunk submission fails.
      cudaStreamSynchronize(stream_);
      throw;
    }
  }

  int device_index_;
  int64_t chunk_bytes_;
  int64_t max_descriptors_;
  std::atomic<int64_t> pipeline_depth_{1};
  BatchFn batch_fn_;
  cudaStream_t stream_ = nullptr;
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable condition_;
  bool initialized_ = false;
  bool stopping_ = false;
  std::string error_;
  std::deque<std::shared_ptr<AsyncBatchMemcpyJob>> work_queue_;
  std::deque<std::shared_ptr<AsyncBatchMemcpyJob>> completion_queue_;
  std::unordered_map<int64_t, std::shared_ptr<AsyncBatchMemcpyJob>>
      active_jobs_;
  std::unique_ptr<CompletionRing> completion_ring_;
};

std::mutex submitter_registry_mutex;
std::unordered_map<int64_t, std::shared_ptr<AsyncBatchMemcpySubmitter>>
    submitter_registry;
int64_t next_submitter_handle = 1;

std::shared_ptr<AsyncBatchMemcpySubmitter> get_submitter(int64_t handle) {
  std::lock_guard<std::mutex> lock(submitter_registry_mutex);
  auto it = submitter_registry.find(handle);
  STD_TORCH_CHECK(it != submitter_registry.end(),
                  "invalid async batch memcpy submitter handle ", handle);
  return it->second;
}

#endif

}  // namespace

int64_t create_async_batch_memcpy_submitter(
    int64_t device_index, int64_t chunk_bytes, int64_t max_descriptors,
    const std::string& completion_ring_path, int64_t completion_session) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  STD_TORCH_CHECK(device_index >= 0, "CUDA device index must be nonnegative");
  STD_TORCH_CHECK(max_descriptors >= 0, "max_descriptors must be nonnegative");
  STD_TORCH_CHECK(chunk_bytes > 0,
                  "async batch memcpy chunk bytes must be positive");
  auto submitter = std::make_shared<AsyncBatchMemcpySubmitter>(
      device_index, chunk_bytes, max_descriptors, completion_ring_path,
      completion_session);
  std::lock_guard<std::mutex> lock(submitter_registry_mutex);
  const int64_t handle = next_submitter_handle++;
  submitter_registry.emplace(handle, std::move(submitter));
  return handle;
#else
  STD_TORCH_CHECK(
      false,
      "async_h2d_submission requires a CUDA build using CUDA toolkit 12.8 "
      "or newer; this build does not provide cuMemcpyBatchAsync");
  return 0;
#endif
}

int64_t create_async_batch_memcpy_completion_ring(const std::string& path,
                                                  int64_t session,
                                                  int64_t capacity) {
  STD_TORCH_CHECK(capacity == 4096,
                  "async completion ring capacity must be 4096");
  auto ring = std::make_shared<CompletionRing>(path, session, capacity, true);
  std::lock_guard<std::mutex> lock(completion_ring_registry_mutex);
  const int64_t handle = next_completion_ring_handle++;
  completion_ring_registry.emplace(handle, std::move(ring));
  return handle;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>,
           std::vector<int64_t>, bool, bool>
poll_async_batch_memcpy_completion_ring(int64_t handle) {
  std::shared_ptr<CompletionRing> ring;
  {
    std::lock_guard<std::mutex> lock(completion_ring_registry_mutex);
    auto it = completion_ring_registry.find(handle);
    STD_TORCH_CHECK(it != completion_ring_registry.end(),
                    "invalid async completion ring handle ", handle);
    ring = it->second;
  }
  return ring->poll();
}

void destroy_async_batch_memcpy_completion_ring(int64_t handle) {
  std::shared_ptr<CompletionRing> ring;
  {
    std::lock_guard<std::mutex> lock(completion_ring_registry_mutex);
    auto it = completion_ring_registry.find(handle);
    STD_TORCH_CHECK(it != completion_ring_registry.end(),
                    "invalid async completion ring handle ", handle);
    ring = std::move(it->second);
    completion_ring_registry.erase(it);
  }
}

void submit_async_batch_memcpy(int64_t handle, int64_t job_id,
                               const torch::stable::Tensor& src_ptrs,
                               const torch::stable::Tensor& dst_ptrs,
                               const torch::stable::Tensor& sizes,
                               bool src_access_any, int64_t compute_stream) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  get_submitter(handle)->submit(job_id, src_ptrs, dst_ptrs, sizes,
                                src_access_any, compute_stream);
#else
  STD_TORCH_CHECK(false, "async batch memcpy submitter is unavailable");
#endif
}

std::tuple<std::vector<int64_t>, std::vector<double>> poll_async_batch_memcpy(
    int64_t handle) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  return get_submitter(handle)->poll();
#else
  STD_TORCH_CHECK(false, "async batch memcpy submitter is unavailable");
  return {};
#endif
}

void wait_async_batch_memcpy(int64_t handle,
                             const std::vector<int64_t>& job_ids) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  get_submitter(handle)->wait(job_ids);
#else
  STD_TORCH_CHECK(false, "async batch memcpy submitter is unavailable");
#endif
}

void configure_async_batch_memcpy_pipeline(int64_t handle, int64_t depth) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  get_submitter(handle)->configure_pipeline(depth);
#else
  STD_TORCH_CHECK(false, "async batch memcpy submitter is unavailable");
#endif
}

void destroy_async_batch_memcpy_submitter(int64_t handle) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  std::shared_ptr<AsyncBatchMemcpySubmitter> submitter;
  {
    std::lock_guard<std::mutex> lock(submitter_registry_mutex);
    auto it = submitter_registry.find(handle);
    STD_TORCH_CHECK(it != submitter_registry.end(),
                    "invalid async batch memcpy submitter handle ", handle);
    submitter = std::move(it->second);
    submitter_registry.erase(it);
  }
  submitter->shutdown();
#else
  STD_TORCH_CHECK(false, "async batch memcpy submitter is unavailable");
#endif
}

int64_t coalesced_seal_notification_abi() { return 1; }
