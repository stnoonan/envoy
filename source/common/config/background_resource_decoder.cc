#include "source/common/config/background_resource_decoder.h"

#if defined(__linux__)
#include <sys/prctl.h>
#elif defined(__APPLE__)
#include <pthread.h>
#endif

#include "source/common/common/assert.h"

namespace Envoy {
namespace Config {

namespace {

void setCurrentThreadName(const std::string& name) {
#if defined(__linux__)
  // pthread_setname_np / prctl is limited to 15 chars + NUL on Linux.
  std::string truncated = name.substr(0, 15);
  ::prctl(PR_SET_NAME, truncated.c_str(), 0, 0, 0);
#elif defined(__APPLE__)
  ::pthread_setname_np(name.c_str());
#else
  (void)name;
#endif
}

} // namespace

BackgroundResourceDecoder::BackgroundResourceDecoder(absl::string_view thread_name)
    : thread_name_(thread_name) {
  worker_ = std::thread([this]() {
    setCurrentThreadName(thread_name_);
    workerLoop();
  });
}

BackgroundResourceDecoder::~BackgroundResourceDecoder() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stopping_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void BackgroundResourceDecoder::submit(OpaqueResourceDecoderSharedPtr resource_decoder,
                                       Protobuf::RepeatedPtrField<Protobuf::Any> resources,
                                       std::string version, Event::Dispatcher& dispatcher,
                                       DecodeBatchCallback on_complete) {
  ASSERT(resource_decoder != nullptr);
  Batch batch{std::move(resource_decoder), std::move(resources), std::move(version), &dispatcher,
              std::move(on_complete)};
  {
    std::lock_guard<std::mutex> lock(mu_);
    // We tolerate but log unexpectedly large queues; if the producer is faster
    // than the worker for sustained periods that means the cluster's xDS push
    // cadence exceeds the decoder's throughput and the operator likely wants
    // to disable the feature or shard subscriptions.
    if (queue_.size() == 64) {
      ENVOY_LOG(warn,
                "background xDS decoder queue at depth {} for thread {}; main thread relief may "
                "be degraded",
                queue_.size(), thread_name_);
    }
    queue_.push_back(std::move(batch));
  }
  cv_.notify_one();
}

void BackgroundResourceDecoder::workerLoop() {
  for (;;) {
    Batch batch;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [this]() { return stopping_ || !queue_.empty(); });
      if (stopping_ && queue_.empty()) {
        return;
      }
      batch = std::move(queue_.front());
      queue_.pop_front();
      processing_ = true;
    }
    processBatch(std::move(batch));
    {
      std::lock_guard<std::mutex> lock(mu_);
      processing_ = false;
    }
    cv_.notify_all();
  }
}

void BackgroundResourceDecoder::processBatch(Batch batch) {
  std::vector<DecodedResourceOrError> results;
  results.reserve(batch.resources.size());

  for (const auto& any : batch.resources) {
    // DecodedResourceImpl::fromResource performs the actual proto unpack +
    // ValidationVisitor traversal, which is the part we are trying to keep
    // off the main thread. It returns absl::StatusOr so we don't have to deal
    // with exceptions crossing the thread boundary (which would terminate the
    // process given our standard build flags).
    absl::StatusOr<DecodedResourceImplPtr> decoded =
        DecodedResourceImpl::fromResource(*batch.resource_decoder, any, batch.version);
    results.emplace_back(std::move(decoded));
  }

  // Post the completion callback back to the dispatcher thread. Dispatcher::post
  // is documented to be safe cross-thread. Note that if the dispatcher has
  // already been destroyed, this will UB; the GrpcMuxImpl ensures that no
  // submit() outlives its dispatcher by destroying the decoder before the
  // dispatcher in its destructor order.
  Event::Dispatcher* dispatcher = batch.dispatcher;
  DecodeBatchCallback cb = std::move(batch.on_complete);
  // Move results into a shared holder so we can move-capture in the lambda.
  auto results_holder = std::make_shared<std::vector<DecodedResourceOrError>>(std::move(results));
  dispatcher->post([results_holder, cb = std::move(cb)]() mutable {
    std::move(cb)(std::move(*results_holder));
  });
}

void BackgroundResourceDecoder::waitForIdleForTest() {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [this]() { return queue_.empty() && !processing_; });
}

uint64_t BackgroundResourceDecoder::queueDepthForTest() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<uint64_t>(queue_.size()) + (processing_ ? 1 : 0);
}

} // namespace Config
} // namespace Envoy
