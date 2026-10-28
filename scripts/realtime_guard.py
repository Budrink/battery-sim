"""Performance acceptance, separate from numerical timestep/physics laws."""
from collections import deque
import math
import re
import time

PROGRESS = re.compile(r'Replay: t=([\d.eE+-]+).*?wall_s=([\d.eE+-]+)')
AVERAGE_MINIMUM = .5
ROLLING_MINIMUM = .25
WINDOW_S = 20.


def completed_performance(simulated_s, wall_s, minimum=AVERAGE_MINIMUM):
    """Whole-trajectory acceptance is separate from the short-window floor."""
    if not math.isfinite(simulated_s) or simulated_s < 0 or not math.isfinite(wall_s) or wall_s <= 0:
        raise ValueError('Invalid completed trajectory timing')
    if not math.isfinite(minimum) or minimum<=0.:raise ValueError('Invalid average performance minimum')
    ratio = simulated_s / wall_s
    return dict(realtime_ratio=ratio, minimum=minimum,
                wall_s=wall_s, simulated_s=simulated_s,
                scope='whole_run', accepted=ratio >= minimum)


class RealtimeFloor:
    """Reject a sustained speed below simulated/wall ratio, never enlarge dt.

    A 20 s native-clock window ignores launch/poll jitter but cannot hide a
    slow destruction phase behind a long, fast warmup. CUDA startup is excluded
    by starting the window at the first native progress report.
    """
    def __init__(self, minimum=ROLLING_MINIMUM, window_s=WINDOW_S):
        if not math.isfinite(minimum) or minimum <= 0 or not math.isfinite(window_s) or window_s <= 0:
            raise ValueError('Invalid realtime floor')
        self.minimum, self.window = minimum, window_s
        self.samples = deque()
        self.last_native_wall=None
        self.last_heartbeat_host=None

    def observe(self, wall_s, simulated_s):
        if not math.isfinite(wall_s) or not math.isfinite(simulated_s):
            raise ValueError('Nonfinite progress')
        if self.samples:
            if wall_s == self.samples[-1][0]:
                return None
            if wall_s < self.samples[-1][0] or simulated_s < self.samples[-1][1]:
                raise ValueError('Nonmonotone progress')
        self.samples.append((wall_s, simulated_s))
        # Retain the point immediately before the left window boundary.
        while len(self.samples) > 2 and self.samples[1][0] <= wall_s-self.window:
            self.samples.popleft()
        if wall_s-self.samples[0][0] < self.window:
            return None
        a, b = self.samples[0], self.samples[1]
        start = wall_s-self.window
        fraction = (start-a[0])/(b[0]-a[0])
        start_sim = a[1]+fraction*(b[1]-a[1])
        ratio = (simulated_s-start_sim)/self.window
        return dict(realtime_ratio=ratio, minimum=self.minimum, window_s=self.window,
                    wall_s=wall_s, simulated_s=simulated_s, scope='rolling', accepted=ratio >= self.minimum)

    def inspect(self, log, now=None):
        matches = list(PROGRESS.finditer(log))
        if not matches:
            return None
        m = matches[-1]
        wall,sim=float(m[2]),float(m[1]);host=time.monotonic()if now is None else now
        if wall==self.last_native_wall:
            if host-self.last_heartbeat_host>=self.window:
                return dict(realtime_ratio=0.,minimum=self.minimum,window_s=self.window,
                    wall_s=wall,simulated_s=sim,accepted=False,stalled=True)
            return None
        self.last_native_wall=wall;self.last_heartbeat_host=host
        return self.observe(wall,sim)
