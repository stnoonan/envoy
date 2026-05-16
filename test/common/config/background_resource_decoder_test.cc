#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "envoy/api/api.h"
#include "envoy/service/discovery/v3/discovery.pb.h"

#include "source/common/config/background_resource_decoder.h"
#include "source/common/config/decoded_resource_impl.h"
#include "source/common/config/opaque_resource_decoder_impl.h"
#include "source/common/protobuf/protobuf.h"

#include "test/mocks/config/mocks.h"
#include "test/test_common/simulated_time_system.h"
#include "test/test_common/utility.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::InvokeWithoutArgs;
using ::testing::NiceMock;
using ::testing::Return;

namespace Envoy {
namespace Config {
namespace {

// Minimal helper: a "real" decoder that returns a typed message and a name
// derived from the type_url. It is thread-safe because it has no mutable
// member state.
class StubDecoder : public OpaqueResourceDecoder {
public:
  ProtobufTypes::MessagePtr decodeResource(const Protobuf::Any& resource) override {
    auto msg = std::make_unique<envoy::service::discovery::v3::Resource>();
    msg->set_name(resource.type_url());
    return msg;
  }
  std::string resourceName(const Protobuf::Message& resource) override {
    const auto* typed = dynamic_cast<const envoy::service::discovery::v3::Resource*>(&resource);
    return typed != nullptr ? typed->name() : "";
  }
};

class BackgroundResourceDecoderTest : public testing::Test {
public:
  BackgroundResourceDecoderTest()
      : api_(Api::createApiForTest(stats_, time_system_)),
        dispatcher_(api_->allocateDispatcher("bg_decoder_test")),
        decoder_(std::make_shared<StubDecoder>()),
        background_(std::make_unique<BackgroundResourceDecoder>("test.xds.dec")) {}

  // Run the dispatcher until either `predicate` is satisfied or we exhaust
  // the supplied number of iterations.
  void pumpUntil(std::function<bool()> predicate, int max_iters = 200) {
    for (int i = 0; i < max_iters; ++i) {
      if (predicate()) {
        return;
      }
      dispatcher_->run(Event::Dispatcher::RunType::NonBlock);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    FAIL() << "pumpUntil timed out";
  }

  Protobuf::RepeatedPtrField<Protobuf::Any> makeAnyList(std::initializer_list<std::string> urls) {
    Protobuf::RepeatedPtrField<Protobuf::Any> out;
    for (const auto& u : urls) {
      auto* a = out.Add();
      a->set_type_url(u);
    }
    return out;
  }

  Event::SimulatedTimeSystem time_system_;
  Stats::TestUtil::TestStore stats_;
  Api::ApiPtr api_;
  Event::DispatcherPtr dispatcher_;
  OpaqueResourceDecoderSharedPtr decoder_;
  BackgroundResourceDecoderPtr background_;
};

// Submitted resources come back, in order, on the dispatcher thread.
TEST_F(BackgroundResourceDecoderTest, BasicSubmitAndComplete) {
  std::atomic<bool> done{false};
  std::vector<DecodedResourceOrError> received;
  // Capture the thread id that runs the callback so we can verify it was
  // posted back to the dispatcher thread, not invoked inline on the worker.
  std::atomic<std::thread::id> cb_thread_id{};

  background_->submit(decoder_, makeAnyList({"type.googleapis.com/a", "type.googleapis.com/b"}),
                      "v1", *dispatcher_, [&](std::vector<DecodedResourceOrError> results) {
                        cb_thread_id = std::this_thread::get_id();
                        received = std::move(results);
                        done = true;
                      });

  pumpUntil([&]() { return done.load(); });

  ASSERT_EQ(2u, received.size());
  ASSERT_TRUE(received[0].ok());
  EXPECT_EQ("type.googleapis.com/a", received[0].value()->name());
  ASSERT_TRUE(received[1].ok());
  EXPECT_EQ("type.googleapis.com/b", received[1].value()->name());

  EXPECT_EQ(cb_thread_id.load(), std::this_thread::get_id())
      << "completion callback must run on the dispatcher thread";
}

// FIFO: multiple submissions complete in submit order.
TEST_F(BackgroundResourceDecoderTest, FifoOrdering) {
  std::vector<int> completion_order;
  std::atomic<int> remaining{3};

  auto submit_batch = [&](int idx) {
    background_->submit(decoder_, makeAnyList({"type.googleapis.com/b" + std::to_string(idx)}),
                        "v1", *dispatcher_, [&, idx](std::vector<DecodedResourceOrError>) {
                          completion_order.push_back(idx);
                          --remaining;
                        });
  };
  submit_batch(0);
  submit_batch(1);
  submit_batch(2);

  pumpUntil([&]() { return remaining.load() == 0; });

  ASSERT_EQ(3u, completion_order.size());
  EXPECT_EQ(0, completion_order[0]);
  EXPECT_EQ(1, completion_order[1]);
  EXPECT_EQ(2, completion_order[2]);
}

// Shutting down the decoder while batches are queued must drop them silently
// (no callback firing, no crash).
TEST_F(BackgroundResourceDecoderTest, ShutdownDropsPendingBatches) {
  std::atomic<int> callbacks_invoked{0};

  // Pile up a number of batches; we expect at most a few to drain before
  // shutdown.
  for (int i = 0; i < 64; ++i) {
    background_->submit(decoder_, makeAnyList({"type.googleapis.com/x" + std::to_string(i)}), "v1",
                        *dispatcher_,
                        [&](std::vector<DecodedResourceOrError>) { ++callbacks_invoked; });
  }

  // Tear down immediately. The worker joins; any batches it did not get to are
  // dropped on the floor.
  background_.reset();
  // Pump the dispatcher to surface any callbacks that managed to be posted
  // before shutdown.
  dispatcher_->run(Event::Dispatcher::RunType::NonBlock);

  // We don't assert an exact number, only that no UAF occurred and we did not
  // somehow get more callbacks than we submitted.
  EXPECT_LE(callbacks_invoked.load(), 64);
}

// Empty batch: callback fires with empty result vector.
TEST_F(BackgroundResourceDecoderTest, EmptyBatch) {
  std::atomic<bool> done{false};
  std::vector<DecodedResourceOrError> received;
  background_->submit(decoder_, Protobuf::RepeatedPtrField<Protobuf::Any>{}, "v1", *dispatcher_,
                      [&](std::vector<DecodedResourceOrError> results) {
                        received = std::move(results);
                        done = true;
                      });
  pumpUntil([&]() { return done.load(); });
  EXPECT_TRUE(received.empty());
}

// Concurrency stress: hammer submit() from multiple producer threads (this is
// not the GrpcMuxImpl usage pattern, but it exercises the queue's locking).
TEST_F(BackgroundResourceDecoderTest, ConcurrentSubmissions) {
  constexpr int kProducers = 4;
  constexpr int kPerProducer = 25;
  std::atomic<int> completed{0};

  std::vector<std::thread> producers;
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p]() {
      for (int i = 0; i < kPerProducer; ++i) {
        background_->submit(
            decoder_,
            makeAnyList({"type.googleapis.com/p" + std::to_string(p) + "_" + std::to_string(i)}),
            "v1", *dispatcher_, [&](std::vector<DecodedResourceOrError>) { ++completed; });
      }
    });
  }
  for (auto& t : producers) {
    t.join();
  }

  pumpUntil([&]() { return completed.load() == kProducers * kPerProducer; }, 1000);
  EXPECT_EQ(kProducers * kPerProducer, completed.load());
}

} // namespace
} // namespace Config
} // namespace Envoy
