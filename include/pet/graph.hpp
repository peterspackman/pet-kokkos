// Record-and-replay of an evaluation's device work as one graph launch (CUDA
// graphs, or HIP's, which are the same API).
//
// For a small structure the ~160 launches of an evaluation are nearly all of its
// cost. run() does the work eagerly the first time it sees a key, captures it
// the second time, and replays the graph after that: the same kernels, in the
// same order, on the same buffers, so the result is bit-identical.
//
// The key must name everything the recorded work depends on -- shapes, where
// the inputs live, the workspace's allocation generation -- since a replay reuses
// the recorded pointers and arguments. A few graphs are kept, the least recently
// used dropped first, for a caller that alternates between shapes (an MD engine
// asking for the energy alone on some steps). A failed capture turns graphs off
// and redoes the work eagerly. Without a GPU, run() just does the work.
//
// Capture needs every kernel launched with its functor as kernel arguments
// (pet::RangePolicy / TeamPolicy): a functor Kokkos stages through constant
// memory makes the host wait on an event, which a HIP capture refuses.
#pragma once

#include "pet/kokkos.hpp"

#if defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace pet {

#if defined(KOKKOS_ENABLE_CUDA)
#define PET_HAVE_GRAPHS 1
namespace gpu {
using Graph = cudaGraph_t;
using Exec = cudaGraphExec_t;
inline cudaStream_t stream() { return ExecSpace().cuda_stream(); }
// Relaxed: Kokkos makes allocator calls (e.g. a zero-byte team-scratch resize)
// that the global mode would refuse, and none of them enqueue work.
inline bool begin() { return cudaStreamBeginCapture(stream(), cudaStreamCaptureModeRelaxed) == cudaSuccess; }
inline bool end(Graph* g) { return cudaStreamEndCapture(stream(), g) == cudaSuccess; }
inline bool instantiate(Exec* e, Graph g) { return cudaGraphInstantiate(e, g, 0) == cudaSuccess; }
inline void launch(Exec e) { cudaGraphLaunch(e, stream()); }
inline void destroy(Graph g) { cudaGraphDestroy(g); }
inline void destroy(Exec e) { cudaGraphExecDestroy(e); }
inline void clear_error() { (void) cudaGetLastError(); }
}  // namespace gpu
#elif defined(KOKKOS_ENABLE_HIP)
#define PET_HAVE_GRAPHS 1
namespace gpu {
using Graph = hipGraph_t;
using Exec = hipGraphExec_t;
inline hipStream_t stream() { return ExecSpace().hip_stream(); }
inline bool begin() { return hipStreamBeginCapture(stream(), hipStreamCaptureModeRelaxed) == hipSuccess; }
inline bool end(Graph* g) { return hipStreamEndCapture(stream(), g) == hipSuccess; }
inline bool instantiate(Exec* e, Graph g) { return hipGraphInstantiate(e, g, nullptr, nullptr, 0) == hipSuccess; }
inline void launch(Exec e) { (void) hipGraphLaunch(e, stream()); }
inline void destroy(Graph g) { (void) hipGraphDestroy(g); }
inline void destroy(Exec e) { (void) hipGraphExecDestroy(e); }
inline void clear_error() { (void) hipGetLastError(); }
}  // namespace gpu
#endif

template <class Out>
class GraphCache {
 public:
  using Key = std::vector<std::uintptr_t>;
  static constexpr std::size_t kKeep = 8;  // graphs held; keys remembered: twice that

  GraphCache() = default;
  GraphCache(const GraphCache&) = delete;
  GraphCache& operator=(const GraphCache&) = delete;
  ~GraphCache() { reset(); }

  // capture_now: record on first sight rather than running eagerly first, for a
  // caller that has just run the same shapes (so nothing allocates). If that
  // capture fails, the key is run eagerly and recorded the next time.
  template <class F>
  Out run(const Key& key, F&& work, bool capture_now = false) {
#if defined(PET_HAVE_GRAPHS)
    if (broken_) return work();
    ++tick_;
    for (auto& g : graphs_)
      if (g.key == key) {
        g.used = tick_;
        gpu::launch(g.exec);
        return g.out;
      }
    const bool seen = std::find(seen_.begin(), seen_.end(), key) != seen_.end();
    if (!seen) {
      if (seen_.size() >= 2 * kKeep) seen_.erase(seen_.begin());
      seen_.push_back(key);
      if (!capture_now) return work();  // first sight: run eagerly
    }
    gpu::Graph graph = nullptr;
    gpu::Exec exec = nullptr;
    if (gpu::begin()) {
      Out out = work();
      if (gpu::end(&graph) && graph && gpu::instantiate(&exec, graph)) {
        gpu::destroy(graph);
        if (graphs_.size() >= kKeep) {  // drop the least recently used
          auto lru = std::min_element(graphs_.begin(), graphs_.end(),
                                      [](const Entry& a, const Entry& b) { return a.used < b.used; });
          gpu::destroy(lru->exec);
          graphs_.erase(lru);
        }
        graphs_.push_back({key, exec, out, tick_});
        gpu::launch(exec);
        return out;
      }
      if (graph) gpu::destroy(graph);
    }
    gpu::clear_error();  // the failed capture's sticky error
    if (!seen) return work();  // an early attempt: eagerly now, recorded next time
    broken_ = true;
    return work();
#else
    (void) key;
    return work();
#endif
  }

 private:
#if defined(PET_HAVE_GRAPHS)
  struct Entry {
    Key key;
    gpu::Exec exec;
    Out out;
    std::uint64_t used;
  };
  void reset() {
    for (auto& g : graphs_) gpu::destroy(g.exec);
    graphs_.clear();
  }
  std::vector<Entry> graphs_;
  std::vector<Key> seen_;
  std::uint64_t tick_ = 0;
#else
  void reset() {}
#endif
  bool broken_ = false;
};

}  // namespace pet
