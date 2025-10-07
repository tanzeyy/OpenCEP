# scripts/MyKleeneClosureOperator.py
"""
MyKleeneClosureOperator: Extends OpenCEP's KleeneClosureOperator to include a StateManager for load shedding under bursty workloads.

Implements:
- Window-based expiration (WITHIN 1h)
- Load shedding (random, window, semantic)
"""

from base.PatternStructure import KleeneClosureOperator as BaseKleeneClosureOperator
from datetime import timedelta
from collections import namedtuple
import math, random


class StateManager:
    """Lightweight partial match manager with shedding policies."""
    def __init__(self, window_seconds=3600, capacity=2000,
                 strategy="random", seed=42):
        self.window_seconds = window_seconds
        self.capacity = capacity
        self.strategy = strategy
        self.random = random.Random(seed)
        self.state = []  # list of PartialMatch
        self.total_shed = 0

    def add_partial_match(self, pm, current_time):
        """Add match, prune expired, and apply shedding."""
        self.state.append(pm)
        # Expire stale partial matches
        self.state = [
            p for p in self.state
            if (current_time - p.first_ts).total_seconds() <= self.window_seconds
        ]
        if len(self.state) > self.capacity:
            self._shed(current_time)

    def _shed(self, current_time):
        n = len(self.state)
        shed_count = n - self.capacity
        if shed_count <= 0:
            return

        if self.strategy == "random":
            self.random.shuffle(self.state)
            self.state = self.state[:self.capacity]
        elif self.strategy == "window":
            # Drop oldest by first timestamp
            self.state.sort(key=lambda p: p.first_ts)
            self.state = self.state[-self.capacity:]
        elif self.strategy == "semantic":
            def score(pm):
                length = len(pm.events)
                remaining = (pm.first_ts + timedelta(seconds=self.window_seconds)
                             - current_time).total_seconds()
                remaining = max(0.0, remaining)
                last_end = pm.events[-1].get("end")
                target_bonus = 1.0 if last_end in {7, 8, 9} else 0.0
                return length * 2.0 + math.log(1 + remaining) + target_bonus * 3.0
            self.state.sort(key=score, reverse=True)
            self.state = self.state[:self.capacity]

        self.total_shed += shed_count

    def get_state(self):
        return self.state

    def get_state_size(self):
        return len(self.state)

    def get_total_shed(self):
        return self.total_shed


class MyKleeneClosureOperator(BaseKleeneClosureOperator):
    """
    A drop-in replacement for OpenCEP's KleeneClosureOperator that adds
    StateManager-based load shedding.
    """

    def __init__(self, arg, min_size=1, max_size=None,
                 window_seconds=3600, capacity=2000, strategy="random"):
        super().__init__(arg=arg, min_size=min_size, max_size=max_size)
        self.state_manager = StateManager(window_seconds, capacity, strategy)

    def process_event(self, event, current_time):
        """
        Called when a new event arrives; manages partial matches.
        """
        PartialMatch = namedtuple("PartialMatch", ["events", "first_ts", "last_ts"])
        new_matches = []

        # 1️⃣ Extend existing partial matches
        for pm in self.state_manager.get_state():
            last = pm.events[-1]
            if (last.get("bike") == event.get("bike") and
                last.get("end") == event.get("start")):
                extended = PartialMatch(
                    events=pm.events + [event],
                    first_ts=pm.first_ts,
                    last_ts=current_time
                )
                new_matches.append(extended)

        # 2️⃣ Start new match
        pm_new = PartialMatch(events=[event],
                              first_ts=current_time,
                              last_ts=current_time)
        new_matches.append(pm_new)

        # 3️⃣ Add to state (pruning/shedding applied inside)
        for pm in new_matches:
            self.state_manager.add_partial_match(pm, current_time)

    def get_state_summary(self):
        return {
            "strategy": self.state_manager.strategy,
            "size": self.state_manager.get_state_size(),
            "shed_total": self.state_manager.get_total_shed()
        }
