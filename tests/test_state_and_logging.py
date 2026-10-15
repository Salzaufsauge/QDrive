import io
from concurrent.futures import ThreadPoolExecutor

from backend.state.train_state import TrainState
from util.logging_broker import LoggingBroker
from util.teestream import StreamType, TeeStream


def drain(subscription):
    queue = subscription.client_queue
    return [queue.get_nowait() for _ in range(queue.qsize())]


def test_episode_cursor_and_copies():
    state = TrainState()
    episode = {"reward": 10, "sequence": -1}
    state.append_episode(episode)
    state.append_episode({"reward": 20})
    assert episode["sequence"] == -1
    assert state.reward_fig(1) == [{"reward": 20, "sequence": 2}]
    snapshot = state.reward_fig()
    snapshot[0]["reward"] = -100
    assert state.reward_fig()[0]["reward"] == 10
    assert state.reward_fig(2) == []


def test_retention_keeps_monotonic_cursor():
    state = TrainState()
    for i in range(505):
        state.append_episode({"reward": i})
    episodes = state.reward_fig()
    assert len(episodes) == 500
    assert episodes[0] == {"reward": 5, "sequence": 6}
    assert episodes[-1]["sequence"] == 505


def test_concurrent_episode_writers_have_unique_ordered_sequences():
    state = TrainState()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: state.append_episode({"reward": i}), range(400)))
    episodes = state.reward_fig()
    assert [episode["sequence"] for episode in episodes] == list(range(1, 401))
    assert {episode["reward"] for episode in episodes} == set(range(400))


def test_subscribers_receive_history_and_unsubscribe_independently():
    broker = LoggingBroker()
    broker.publish("before")
    first, second = broker.subscribe(), broker.subscribe()
    broker.publish("during")
    broker.unsubscribe(first)
    broker.unsubscribe(first)
    broker.publish("after")
    assert drain(first) == ["before", "during"]
    assert drain(second) == ["before", "during", "after"]


def test_history_retains_latest_500_messages():
    broker = LoggingBroker()
    for i in range(505):
        broker.publish(i)
    assert drain(broker.subscribe()) == list(range(5, 505))


def test_slow_subscriber_drops_oldest_without_affecting_fast_subscriber():
    broker = LoggingBroker()
    slow, fast = broker.subscribe(), broker.subscribe()
    for i in range(1005):
        broker.publish(i)
        assert fast.client_queue.get_nowait() == i
    assert drain(slow) == list(range(5, 1005))
    assert slow.dropped_messages == 5
    assert fast.dropped_messages == 0


def test_tee_preserves_original_output_and_filters_blank_messages():
    original = io.StringIO()
    broker = LoggingBroker()
    subscriber = broker.subscribe()
    tee = TeeStream(original, StreamType.STDERR, broker)
    assert tee.write("  error \n") == 9
    assert tee.write("\n \t") == 3
    tee.flush()
    assert original.getvalue() == "  error \n\n \t"
    messages = drain(subscriber)
    assert len(messages) == 1
    assert messages[0].message == "  error"
    assert messages[0].stream_type is StreamType.STDERR
    assert tee.isatty() is False
