# Unit test for kaggle/s26_ht2.patch (HT2 = s22_ht + analysts without thinking; on top of s22_sdt).
# Run from <bundle src after harness-changes + bline + s22_sdt + s26_ht2>/ARC3-Inference:
#   PYTHONPATH=$PWD:<tufa-arc-agi-framework/src> python3 /data/arc/kaggle/tests/test_s26_ht2.py
import os, sys, json, time
os.environ['ARC3_SDT'] = '1'; os.environ['ARC3_SDT_LOG'] = '/tmp/ht_events.jsonl'; os.environ['ARC3_HT_K'] = '2'
os.environ['ARC3_HT_THINKING'] = '0'; os.environ['ARC3_HT_MAX_TOKENS'] = '1500'; os.environ['ARC3_HT_TIMEOUT_S'] = '150'
os.environ['EXPOSE_UNDO'] = 'on'
if os.path.exists('/tmp/ht_events.jsonl'): os.remove('/tmp/ht_events.jsonl')
from pathlib import Path
from inference.agent import tool_agent as ta
from inference.agent.runtime_state import Frame, HistoryEntry
A = object.__new__(ta.ToolAgent)
A._session_runtime_dir = Path('/tmp/g1'); A._tokens_at_level_start = 0; A._session_generated_tokens = 0
A._turn_generated_tokens = 0; A._session_total_tokens = 0
A._history_messages = [{'role': 'user', 'content': 'u'}]; A._last_step_summary = None
A._retained_clear_notice = ''; A._resume_after_yield = False; A._current_valid_actions = ['UP', 'DOWN', 'MOUSE']
A._summarized_knowledge = {'world_model': 'the blue block is the player', 'goal_model': 'reach the yellow cell', 'current_plan': ''}
VALID = ['UP', 'DOWN', 'MOUSE']
calls = []
class R:
    def __init__(self, text, n): self.message = {'content': text}; self.usage = {'completion_tokens': n}; self.finish_reason = 'stop'
def fake_chat(messages, *, tools, request_timeout_seconds=None, max_tokens=None, enable_thinking=None):
    calls.append((messages, tools, max_tokens, enable_thinking, request_timeout_seconds))
    if len(calls) == 2: raise RuntimeError('boom')          # analyst 2 of fire 1 fails: must be skipped
    return R('Sure, here is my diagnosis.\nGOAL: touch yellow.\nMISSED: the gray switch.\nTEST: UP UP, expect the wall to open. (call %d)' % len(calls), 3000)
A._chat_completion = fake_chat
G0 = tuple(tuple(0 for _ in range(4)) for _ in range(4))
def grid(r, c, v=9):
    g = [list(row) for row in G0]; g[r][c] = v; return tuple(tuple(x) for x in g)
class FakeInner:
    max_runtime_s_per_game = 7260.0
    def soft_time_remaining_seconds(self): return None
class FakeSolver:
    def __init__(self):
        self.solver = FakeInner(); self.started_at = time.monotonic(); self.level = 1
        self.history_entries = [HistoryEntry(action='', frame=Frame(grid=G0, step=0, level=1))]
    def current_frame(self): return self.history_entries[-1].frame
    def should_stop(self): return False
    def step_env(self, args): return {}
    def act(self, name, g, data=None, no_op=False):
        r = {'action_name': name, 'level': self.level, 'no_op': no_op, 'game_over': False, 'level_completed': False,
             'valid_actions': VALID, 'action_data': data or {}}
        self.history_entries.append(HistoryEntry(action=name, frame=Frame(grid=g, step=0, level=self.level), result=r))
S = FakeSolver(); step = S.step_env
A._sdt_maybe_intervene(step); assert A._sdt_count == 0
S.act('ACTION1', grid(1, 1)); S.act('ACTION1', grid(1, 1), no_op=True); S.act('ACTION6', grid(2, 3), data={'x': 3, 'y': 2})
A._session_generated_tokens = 61000; A._sdt_maybe_intervene(step); assert A._sdt_count == 1
n = A._retained_clear_notice
assert n.startswith('[Harness record, level 1]') and 'trouble with the first level' in n, n[:200]
assert '[Fresh-eyes analysts] 1 independent analyst(s)' in n and 'Analyst 1:\nGOAL: touch yellow.' in n and 'Analyst 2' not in n, n
assert len(calls) == 2 and calls[0][1] is None and calls[0][2] == 1500 and calls[0][3] is False and calls[0][4] == 150.0, calls[0][2:]
sysmsg, user = calls[0][0][0]['content'], calls[0][0][1]['content']
assert 'Answer directly, without preamble' in sysmsg and 'Reason concisely' not in sysmsg
assert 'Sure, here is' not in n   # preamble before GOAL: dropped
assert 'expert player' in sysmsg and 'Lens: the GOAL' in user and 'Lens: the CONTROLS' in calls[1][0][1]['content']
assert 'Valid actions: UP, DOWN, MOUSE' in user and '- UP: 2x, mean 0 cells changed, 1 changed nothing, changes inside rows 1-1 cols 1-1' in user, user
assert '- MOUSE: 1x, mean 2 cells changed' in user and 'Some clicks: (2,3): 2 cells' in user, user
assert '- Goal model: reach the yellow cell' in user and '- World model: the blue block is the player' in user and 'Plan:' not in user
assert 'First board of this level' in user and 'Current board:' in user
assert A._session_generated_tokens == 64000 and A._turn_generated_tokens == 0   # analyst tokens charged to the game, not the turn
# second fire: both analysts answer
A._retained_clear_notice = ''
A._session_generated_tokens = 115000; A._sdt_maybe_intervene(step); assert A._sdt_count == 2 and len(calls) == 4
assert 'Analyst 2:' in A._retained_clear_notice
# K=0 -> plain SDT
os.environ['ARC3_HT_K'] = '0'; A._session_runtime_dir = Path('/tmp/g2'); A._tokens_at_level_start = 200000
A._sdt_maybe_intervene(step); A._retained_clear_notice = ''
A._session_generated_tokens = 270000; A._sdt_maybe_intervene(step); assert A._sdt_count == 1 and len(calls) == 4
assert 'Fresh-eyes' not in A._retained_clear_notice
ev = [json.loads(l) for l in open('/tmp/ht_events.jsonl')]
hts = [e for e in ev if e['event'] == 'ht']
assert [e['ok'] for e in hts] == [1, 2] and hts[0]['tokens'] == 3000 and hts[1]['tokens'] == 6000, hts
assert all(e['thinking'] is False for e in hts), hts
# regression: ARC3_HT_THINKING unset -> the S22 request (thinking default, S22 system prompt)
del os.environ['ARC3_HT_THINKING']; assert ta._ht_thinking() is True and ta._HT_SYSTEM != ta._HT_SYSTEM_NT and 'Reason concisely' in ta._HT_SYSTEM
print(user[:900]); print('---'); print(n[-700:])
print('ALL OK')
