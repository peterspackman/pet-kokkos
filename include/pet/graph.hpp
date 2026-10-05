// Record-and-replay of an evaluation's device work as one CUDA graph launch.
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
// and redoes the work eagerly. Elsewhere run() just does the work: HIP has the
// same graph API, but Kokkos's HIP backend synchronises an event (for team
// scratch) inside the capture, which HIP refuses and Kokkos treats as fatal.
#pragma once

#include "pet/kokkos.hpp"

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

  template <class F>
  Out run(const Key& key, F&& work) {
#if defined(PET_HAVE_GRAPHS)
    if (broken_) return work();
    ++tick_;
    for (auto& g : graphs_)
      if (g.key == key) {
        g.used = tick_;
        gpu::launch(g.exec);
        return g.out;
      }
    if (std::find(seen_.begin(), seen_.end(), key) == seen_.end()) {  // first sight: run eagerly
      if (seen_.size() >= 2 * kKeep) seen_.erase(seen_.begin());
      seen_.push_back(key);
      return work();
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
