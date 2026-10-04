// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "seal_notifications.h"

extern "C" {
void* seal_open(const char* path) noexcept {
  try {
    return new vllm::SealNotifier(path);
  } catch (...) {
    return nullptr;
  }
}
void seal_close(void* handle) {
  delete static_cast<vllm::SealNotifier*>(handle);
}
int seal_kv(void* handle, int64_t job, int success) noexcept {
  try {
    static_cast<vllm::SealNotifier*>(handle)->kv(job, success);
    return 1;
  } catch (...) {
    return 0;
  }
}
int seal_compute(void* handle, uint64_t step, int success) noexcept {
  try {
    static_cast<vllm::SealNotifier*>(handle)->compute(step, success);
    return 1;
  } catch (...) {
    return 0;
  }
}
}
