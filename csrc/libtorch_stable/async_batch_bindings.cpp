#include "ops.h"
#include <torch/csrc/stable/library.h>
STABLE_TORCH_LIBRARY_FRAGMENT(_C_cache_ops, ops) {
  ops.def(
      "create_async_batch_memcpy_submitter(int device_index, "
      "int chunk_bytes, int max_descriptors, str completion_ring_path='', "
      "int completion_session=0) -> int");
  ops.def(
      "submit_async_batch_memcpy(int handle, int job_id, Tensor src_ptrs, "
      "Tensor dst_ptrs, Tensor sizes, bool src_access_any, int compute_stream) "
      "-> ()");
  ops.def("poll_async_batch_memcpy(int handle) -> (int[], float[])");
  ops.def("wait_async_batch_memcpy(int handle, int[] job_ids) -> ()");
  ops.def("destroy_async_batch_memcpy_submitter(int handle) -> ()");
  ops.def("configure_async_batch_memcpy_pipeline(int handle, int depth) -> ()");
  ops.def(
      "create_async_batch_memcpy_completion_ring(str path, int session, "
      "int capacity) -> int");
  ops.def(
      "poll_async_batch_memcpy_completion_ring(int handle) -> "
      "(int[], int[], int[], int[], bool, bool)");
  ops.def("destroy_async_batch_memcpy_completion_ring(int handle) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C_cache_ops, CPU, ops) {
  ops.impl("submit_async_batch_memcpy", TORCH_BOX(&submit_async_batch_memcpy));
}
STABLE_TORCH_LIBRARY_IMPL(_C_cache_ops, CompositeExplicitAutograd, ops) {
  ops.impl("create_async_batch_memcpy_submitter",
           TORCH_BOX(&create_async_batch_memcpy_submitter));
  ops.impl("poll_async_batch_memcpy", TORCH_BOX(&poll_async_batch_memcpy));
  ops.impl("wait_async_batch_memcpy", TORCH_BOX(&wait_async_batch_memcpy));
  ops.impl("configure_async_batch_memcpy_pipeline",
           TORCH_BOX(&configure_async_batch_memcpy_pipeline));
  ops.impl("destroy_async_batch_memcpy_submitter",
           TORCH_BOX(&destroy_async_batch_memcpy_submitter));
  ops.impl("create_async_batch_memcpy_completion_ring",
           TORCH_BOX(&create_async_batch_memcpy_completion_ring));
  ops.impl("poll_async_batch_memcpy_completion_ring",
           TORCH_BOX(&poll_async_batch_memcpy_completion_ring));
  ops.impl("destroy_async_batch_memcpy_completion_ring",
           TORCH_BOX(&destroy_async_batch_memcpy_completion_ring));
}
