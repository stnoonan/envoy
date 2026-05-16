#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "envoy/common/pure.h"
#include "envoy/config/subscription.h"
#include "envoy/event/dispatcher.h"

#include "source/common/common/logger.h"
#include "source/common/common/non_copyable.h"
#include "source/common/config/decoded_resource_impl.h"
#include "source/common/protobuf/protobuf.h"

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace Envoy {
namespace Config {

// Result of decoding a single Protobuf::Any resource off the main thread.
// On success, holds a non-null DecodedResourceImplPtr. On failure, holds the
// non-OK status from the underlying decoder.
using DecodedResourceOrError = absl::StatusOr<DecodedResourceImplPtr>;

// Completion callback invoked on the originating dispatcher thread (main thread
// for the GrpcMuxImpl use case) after a submitted batch has been fully decoded.
using DecodeBatchCallback =
    absl::AnyInvocable<void(std::vector<DecodedResourceOrError> results) &&>;

/**
 * BackgroundResourceDecoder runs a single dedicated worker thread that consumes
 * batches of (Protobuf::Any, version) inputs and produces DecodedResourceImpl
 * objects via the supplied OpaqueResourceDecoder.
 *
 * Rationale: under the legacy Envoy threading model, every xDS DiscoveryResponse
 * is parsed on the main thread. For large pushes (think EDS updates with tens of
 * thousands of endpoints, or RDS updates with very large RouteConfigurations),
 * the proto unpack + validation work blocks the main dispatcher and starves the
 * config init manager, the admin handler, timers, and the watchdog. We address
 * this by offloading just the decode step to a background thread, while keeping
 * the actual apply-to-thread-local-state step on the main thread (where the
 * rest of Envoy's lock-free invariants assume it must live). The handoff back
 * to the main thread is via Event::Dispatcher::post(), which is documented to
 * be safe cross-thread.
 *
 * Concurrency model:
 *   - Exactly one worker thread per decoder instance. Submissions are FIFO and
 *     are dequeued and processed one batch at a time, so per-type-URL ordering
 *     is preserved as long as each subscription type has a single decoder.
 *   - The queue is guarded by a std::mutex and a std::condition_variable.
 *   - The completion callback for batch N is posted to the dispatcher before
 *     batch N+1 is dequeued, which means the dispatcher thread sees completion
 *     callbacks in the same order as the submit() calls (the dispatcher post
 *     queue is itself FIFO per producer-thread; see Event::DispatcherImpl::post
 *     comments).
 *   - The decoder thread takes ownership of submitted Any messages (move) so
 *     callers do not need to keep the original DiscoveryResponse alive.
 *   - The OpaqueResourceDecoder reference must outlive the BackgroundResourceDecoder.
 *     For GrpcMuxImpl this is naturally satisfied: the decoder is owned by the
 *     watch, which is owned by the GrpcMuxImpl, which also owns this object.
 *
 * Lifecycle:
 *   - The worker thread is spawned eagerly in the constructor.
 *   - The destructor signals shutdown, joins the worker, and drops any
 *     pending batches (their callbacks are not invoked). Callers must ensure
 *     that any callback that captures `this` is either (a) safe to drop or
 *     (b) sequenced before destruction.
 *
 * Note: we intentionally use std::thread here rather than the Envoy::Thread
 * ThreadFactory indirection because (i) the worker is internally owned and
 * never needs to be mocked, and (ii) plumbing Api& through GrpcMuxContext
 * would touch a much wider surface area than this targeted change warrants.
 */
class BackgroundResourceDecoder : NonCopyable, public Logger::Loggable<Logger::Id::config> {
public:
  /**
   * @param thread_name name used to identify the worker thread (pthread name,
   *   truncated to 15 chars on Linux).
   */
  explicit BackgroundResourceDecoder(absl::string_view thread_name);

  /**
   * Signals shutdown to the worker thread and joins it. Any batches that are
   * still queued at this point are dropped (their callbacks are not invoked).
   */
  ~BackgroundResourceDecoder();

  /**
   * Submit a batch of Protobuf::Any resources for asynchronous decoding.
   *
   * @param resource_decoder shared decoder to use for each resource. The
   *   shared_ptr is held by the batch for its full lifetime, so the decoder
   *   stays alive even if the originating xDS watch is removed before the
   *   decode completes.
   * @param resources resources to decode, taken by value/move.
   * @param version version string applied to each decoded resource.
   * @param dispatcher dispatcher whose thread will receive `on_complete`.
   *   Typically the main dispatcher of the owning GrpcMuxImpl.
   * @param on_complete callback invoked on `dispatcher`'s thread with the
   *   decoded results, in the same order as the input resources. Errors for
   *   individual resources are reported via absl::Status entries; the batch
   *   as a whole always completes (no exception escapes to the dispatcher).
   *
   * Thread-safe: submit() may be called from any thread, though in practice
   * GrpcMuxImpl only ever calls it from its dispatcher thread.
   */
  void submit(OpaqueResourceDecoderSharedPtr resource_decoder,
              Protobuf::RepeatedPtrField<Protobuf::Any> resources, std::string version,
              Event::Dispatcher& dispatcher, DecodeBatchCallback on_complete);

  /**
   * Test-only: block until the queue is drained. Does not stop the worker.
   */
  void waitForIdleForTest();

  /**
   * Test-only: number of batches still queued (including the one being
   * processed, if any).
   */
  uint64_t queueDepthForTest() const;

private:
  struct Batch {
    OpaqueResourceDecoderSharedPtr resource_decoder;
    Protobuf::RepeatedPtrField<Protobuf::Any> resources;
    std::string version;
    Event::Dispatcher* dispatcher;
    DecodeBatchCallback on_complete;
  };

  void workerLoop();
  void processBatch(Batch batch);

  const std::string thread_name_;

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Batch> queue_; // guarded by mu_
  bool stopping_{false};    // guarded by mu_
  bool processing_{false};  // guarded by mu_

  // worker_ is initialized last so the loop only starts after the rest of the
  // state is constructed.
  std::thread worker_;
};

using BackgroundResourceDecoderPtr = std::unique_ptr<BackgroundResourceDecoder>;

} // namespace Config
} // namespace Envoy
