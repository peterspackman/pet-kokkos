// Record-and-replay of an evaluation's device work as one CUDA graph launch.
//
// A PET evaluation is ~160 kernel launches and GEMMs whose shapes and buffers are
// fixed once the shapes are. For a small structure the GPU work takes
// microseconds and the launches are nearly all of the call. run() does the work
// eagerly the first time it sees a key, captures it the second time, and from
// then on replays the captured graph, which is bit-identical: the same kernels,
// in the same order, on the same buffers.
//
// The key must name everything the recorded work depends on -- shapes, where the
// inputs live, and the workspace's allocation generation -- because a replay
// re-reads the recorded pointers and re-applies the recorded kernel arguments.
// Only the most recent graph is kept: a replay after a different key has grown
// the workspace could reach freed memory, and one graph is what a stepping loop
// needs anyway. A capture that fails for any reason turns graphs off for good
// and redoes the work eagerly. CUDA only; elsewhere run() just calls the work.
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
