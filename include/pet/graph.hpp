// Record-and-replay of an evaluation's device work as one CUDA graph launch.
//
// For a small structure the ~160 launches of an evaluation are nearly all of its
// cost. run() does the work eagerly the first time it sees a key, captures it
// the second time, and replays the graph after that: the same kernels, in the
// same order, on the same buffers, so the result is bit-identical.
//
// The key must name everything the recorded work depends on -- shapes, where
// the inputs live, the workspace's allocation generation -- since a replay reuses
// the recorded pointers and arguments. Only the latest graph is kept, which is
// what a stepping loop needs. A failed capture turns graphs off and redoes the
// work eagerly. Off CUDA, run() just does the work.
#pragma once

#include "pet/kokkos.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace pet {

template <class Out>
class GraphCache {
 public:
  using Key = std::vector<std::uintptr_t>;
  GraphCache() = default;
  GraphCache(const GraphCache&) = delete;
  GraphCache& operator=(const GraphCache&) = delete;
  ~GraphCache() { reset(); }

  template <class F>
  Out run(const Key& key, F&& work) {
#if defined(KOKKOS_ENABLE_CUDA)
    if (!broken_ && exec_ && key == key_) return launch(), out_;
    if (broken_ || key != seen_) {  // first sight: run eagerly, capture next time
      seen_ = key;
      return work();
    }
    reset();
    const cudaStream_t s = ExecSpace().cuda_stream();
    cudaGraph_t g = nullptr;
    // Relaxed: Kokkos makes allocator calls (e.g. a zero-byte team-scratch
    // resize) that the global mode would refuse, and none of them enqueue work.
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed) == cudaSuccess) {
      out_ = work();
      if (cudaStreamEndCapture(s, &g) == cudaSuccess && g &&
          cudaGraphInstantiate(&exec_, g, 0) == cudaSuccess) {
        cudaGraphDestroy(g);
        key_ = key;
        launch();
        return out_;
      }
      if (g) cudaGraphDestroy(g);
    }
    (void) cudaGetLastError();  // clear the failed capture's sticky error
    broken_ = true, exec_ = nullptr;
    return work();
#else
    (void) key;
    return work();
#endif
  }

 private:
#if defined(KOKKOS_ENABLE_CUDA)
  void launch() { cudaGraphLaunch(exec_, ExecSpace().cuda_stream()); }
  void reset() {
    if (exec_) cudaGraphExecDestroy(exec_);
    exec_ = nullptr, key_.clear();
  }
  cudaGraphExec_t exec_ = nullptr;
#else
  void reset() {}
#endif
  Key key_, seen_;
  Out out_{};
  bool broken_ = false;
};

}  // namespace pet
