import os, sys, json, copy
sys.path.insert(0, os.environ.get("ARC3_SRC", "/tmp/bt3/src/ARC3-Inference"))
os.environ["ARC3_WMK"] = "1"
from inference.agent.python_tool_sandbox import run_sandboxed_python

MAZE = ["......",
        ".##.#.",
        ".#..#.",
        ".#.##.",
        "...#G.",
        "####.."]
class Game:
    def __init__(self):
        self.p = (0, 0); self.level = 1; self.step_n = 0; self.history = []
        self.history.append({"action": "", "frame": self.frame(), "result": {}})
    def ascii(self):
        rows = [list(r) for r in MAZE]
        rows[self.p[0]][self.p[1]] = "P"
        return "\n".join("".join(r) for r in rows)
    def frame(self):
        return {"ascii": self.ascii(), "step": self.step_n, "level": self.level, "shape": [6, 6],
                "grid": [[0] * 6 for _ in range(6)]}
    def apply(self, name, row=None, col=None):
        d = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
        self.step_n += 1
        res = {"executed": True, "executed_actions": [name], "gameplay_changed": False}
        if name == "MOUSE":
            label = "MOUSE(row=%d, col=%d)" % (row, col)
        else:
            label = name
            if name in d:
                r, c = self.p[0] + d[name][0], self.p[1] + d[name][1]
                if 0 <= r < 6 and 0 <= c < 6 and MAZE[r][c] != "#":
                    self.p = (r, c); res["gameplay_changed"] = True
        if MAZE[self.p[0]][self.p[1]] == "G":
            res["level_completed"] = True
            self.level += 1; self.p = (0, 0)
        self.history.append({"action": label, "frame": self.frame(), "result": dict(res)})
        return res
    def state(self, last=None):
        return {"current_frame": self.frame(), "history": copy.deepcopy(self.history),
                "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"],
                "last_action_call_result": last or {}}

def make_handler(g, log):
    def handler(actions, stale_after=None):
        if stale_after:
            res = {"stop_reason": "stale_state", "stop_detail": "previous action %s changed nothing" % stale_after,
                   "executed": False}
            return {"action_result": res, "state": g.state(res)}
        res = None
        for a in actions:
            res = g.apply(a["action"].upper(), a.get("row"), a.get("col"))
            log.append(a)
        return {"action_result": res, "state": g.state(res)}
    return handler

MODEL = r'''
D = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
def extract(frame):
    rows = frame.ascii.split("\n")
    for r, line in enumerate(rows):
        c = line.find("P")
        if c >= 0:
            return (r, c)
WALLS = set()
GOAL = None
for r, line in enumerate(current_frame.ascii.split("\n")):
    for c, ch in enumerate(line):
        if ch == "#": WALLS.add((r, c))
        if ch == "G": GOAL = (r, c)
def step(s, a):
    if isinstance(a, dict):
        return s
    dr, dc = D[a]
    r, c = s[0] + SIGN * dr, s[1] + dc
    if not (0 <= r < 6 and 0 <= c < 6) or (r, c) in WALLS:
        return s
    return (r, c)
def step_bfs(s, a):
    n = step(s, a)
    return None if n == s else n
'''

def run(g, code, log):
    out = run_sandboxed_python(code=code, timeout_seconds=30, initial_state=g.state(),
                               action_handler=make_handler(g, log))
    return out

# warm-up history: RIGHT, DOWN (blocked by wall), RIGHT, MOUSE click (no-op), LEFT
g = Game(); log = []
for a in [("RIGHT",), ("DOWN",), ("RIGHT",), ("MOUSE", 2, 3), ("LEFT",)]:
    g.apply(*a)

# T1 correct model: check, plan, run
code = "SIGN = 1\n" + MODEL + r'''
print("check", check_model(extract, step))
res = plan(extract(current_frame), step_bfs, lambda s: s == GOAL, ["UP", "DOWN", "LEFT", "RIGHT"])
print("plan", res["reason"], res["states"], res["path"])
print("run", run_plan(res["path"], extract, step))
'''
out = run(g, code, log)
print("T1 error:", out.get("error")); print(out.get("stdout"))
print("T1 level now", g.level, "actions sent", len(log))

# T2 wrong model (UP/DOWN swapped): check shows mismatches, run_plan stops after 1 action
g = Game(); log = []
for a in [("RIGHT",), ("DOWN",), ("RIGHT",), ("LEFT",)]:
    g.apply(*a)
code = "SIGN = -1\n" + MODEL + r'''
print("check", check_model(extract, step))
res = plan(extract(current_frame), step_bfs, lambda s: s == GOAL, ["UP", "DOWN", "LEFT", "RIGHT"])
print("plan", res["reason"], res["path"])
if res["path"]:
    print("run", run_plan(res["path"], extract, step))
'''
out = run(g, code, log)
print("T2 error:", out.get("error")); print(out.get("stdout"))
print("T2 actions sent", len(log))

# T3 limits: unbounded counter model with no goal -> max_states / time_limit
g = Game(); log = []
code = r'''
print(plan(0, lambda s, a: s + a, lambda s: False, [1, 2, 3], max_states=50000))
print(plan((0, 0), lambda s, a: (s[0] + a, s[1] * 31 + a), lambda s: False, [1, 2, 3, 4, 5], max_states=10**9, time_limit=3))
print(plan(0, lambda s, a: 1 // 0, lambda s: False, [1])["reason"])
print(plan(0, lambda s, a: s + 1 if s < 5 else None, lambda s: False, [1])["reason"])
'''
import time as _t; _t0 = _t.time()
out = run(g, code, log)
print("T3 secs", round(_t.time() - _t0, 2))
print("T3 error:", out.get("error")); print(out.get("stdout"))

# T4 MOUSE label parsing in check_model
g = Game(); log = []
g.apply("MOUSE", 4, 5); g.apply("RIGHT")
code = r'''
seen = []
def ex(f): return f.step
def st(s, a):
    seen.append(a)
    return s + 1
print(check_model(ex, st))
print(seen)
'''
out = run(g, code, log)
print("T4 error:", out.get("error")); print(out.get("stdout"))
# T5 disabled
os.environ["ARC3_WMK"] = "0"
out = run(Game(), """
try:
    plan
    print('present')
except NameError:
    print('absent')
""", [])
print("T5", out.get("error"), out.get("stdout"))
