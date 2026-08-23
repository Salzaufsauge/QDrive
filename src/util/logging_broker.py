import asyncio
from collections import deque
from dataclasses import dataclass


@dataclass(eq=False)
class LoggingSubscription:
    client_queue: asyncio.Queue


class LoggingBroker:
    def __init__(self):
        self.history = deque(maxlen=500)
        self.subs: set[LoggingSubscription] = set()

    def unsubscribe(self, client_queue: LoggingSubscription):
        self.subs.discard(client_queue)

    def subscribe(self):
        subscription = LoggingSubscription(asyncio.Queue(maxsize=1000))
        for message in self.history:
            subscription.client_queue.put_nowait(message)
        self.subs.add(subscription)
        return subscription

    def publish(self, message):
        self.history.append(message)
        for client in self.subs:
            client.client_queue.put_nowait(message)
