// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>

namespace vllm {
// ABI shared with vllm/v1/kv_offload/seal_notifications.py.
class SealNotifier {
  static constexpr uint64_t magic = 0x564c4c4d534e3031ULL;
  static constexpr size_t slots = 10, jobs = slots + 16 * 4;
  static constexpr size_t computes = jobs + 256 * 2;
  static constexpr size_t size = (computes + 16 * 3) * sizeof(uint64_t);
  int fd_ = -1, pipe_ = -1;
  uint64_t* words_ = nullptr;
  uint64_t rank_ = 0;
  std::mutex mutex_;

  class Guard {
    int fd_;

   public:
    explicit Guard(int fd) : fd_(fd) {
      while (flock(fd_, LOCK_EX)) {
        if (errno != EINTR) throw std::runtime_error("seal notification lock");
      }
    }
    ~Guard() { flock(fd_, LOCK_UN); }
  };

  void notify() {
    ++words_[5];
    const char byte = 1;
    ssize_t result;
    do {
      result = write(pipe_, &byte, 1);
    } while (result < 0 && errno == EINTR);
    if (result != 1 && errno != EAGAIN)
      throw std::runtime_error("seal notification write");
  }

  void candidates_ready() {
    const uint64_t mask = (1ULL << words_[2]) - 1;
    for (size_t slot = 0; slot < words_[8]; ++slot) {
      auto* entry = words_ + slots + slot * 4;
      if (entry[3] || entry[0] != mask) continue;
      bool ready = true;
      for (size_t job = entry[1]; job < entry[1] + entry[2]; ++job)
        ready &= words_[jobs + job * 2 + 1] == mask;
      if (ready) {
        entry[3] = 1;
        notify();
      }
    }
  }

 public:
  explicit SealNotifier(const std::string& transport) {
    std::ifstream descriptor(transport + ".seal");
    if (!descriptor) {
      if (access((transport + ".seal").c_str(), F_OK) == 0)
        throw std::runtime_error("unreadable seal notification descriptor");
      return;
    }
    std::string path;
    uint64_t session;
    std::getline(descriptor, path);
    if (!(descriptor >> rank_ >> session))
      throw std::runtime_error("invalid seal notification descriptor");
    try {
      fd_ = open(path.c_str(), O_RDWR | O_CLOEXEC);
      struct stat st{};
      if (fd_ < 0 || fstat(fd_, &st) || st.st_size != size)
        throw std::runtime_error("seal notification open");
      void* mapped =
          mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
      if (mapped == MAP_FAILED)
        throw std::runtime_error("seal notification mmap");
      words_ = static_cast<uint64_t*>(mapped);
      if (words_[0] != magic || words_[1] != session || words_[2] > 63 ||
          !words_[2] || rank_ >= words_[2])
        throw std::runtime_error("seal notification identity");
      pipe_ = open((path + ".fifo").c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC);
      if (pipe_ < 0) throw std::runtime_error("seal notification FIFO open");
    } catch (...) {
      if (words_) munmap(words_, size);
      if (fd_ >= 0) close(fd_);
      throw;
    }
  }
  ~SealNotifier() {
    if (words_) munmap(words_, size);
    if (fd_ >= 0) close(fd_);
    if (pipe_ >= 0) close(pipe_);
  }
  bool enabled() const { return words_ != nullptr; }

  void kv(int64_t job, bool success) {
    if (!words_) return;
    std::lock_guard<std::mutex> local(mutex_);
    Guard guard(fd_);
    if (!success) {
      words_[6] = 1;
      if (words_[3]) notify();
      return;
    }
    if (!words_[3]) return;
    for (size_t slot = 0; slot < words_[8]; ++slot) {
      auto* entry = words_ + slots + slot * 4;
      for (size_t index = entry[1]; index < entry[1] + entry[2]; ++index) {
        auto* item = words_ + jobs + index * 2;
        if (item[0] == static_cast<uint64_t>(job)) item[1] |= 1ULL << rank_;
      }
    }
    candidates_ready();
  }

  void compute(uint64_t step, bool success) {
    if (!words_) return;
    std::lock_guard<std::mutex> local(mutex_);
    Guard guard(fd_);
    if (!words_[3]) return;
    for (size_t index = 0; index < words_[9]; ++index) {
      auto* entry = words_ + computes + index * 3;
      if (entry[0] != step) continue;
      if (!success) {
        words_[6] = 1;
        notify();
        return;
      }
      entry[1] |= 1ULL << rank_;
      if (!entry[2] && entry[1] == (1ULL << words_[2]) - 1) {
        entry[2] = 1;
        notify();
      }
      return;
    }
  }
};
}  // namespace vllm
