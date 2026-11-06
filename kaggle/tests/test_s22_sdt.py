# Unit test for kaggle/s22_sdt.patch. Run from <bundle src after harness-changes + bline + s22_sdt>/ARC3-Inference:
#   PYTHONPATH=$PWD python3 /data/arc/kaggle/tests/test_s22_sdt.py
import os, sys, json, time
os.environ['ARC3_SDT'] = '1'; os.environ['ARC3_SDT_LOG'] = '/tmp/sdt_events.jsonl'; os.environ['ARC3_PERSISTENT_FUNCTIONS'] = '1'
os.environ['EXPOSE_UNDO'] = 'on'
if os.path.exists('/tmp/sdt_events.jsonl'): os.remove('/tmp/sdt_events.jsonl')
from pathlib import Path
from inference.agent import tool_agent as ta
from inference.agent.runtime_state import Frame, HistoryEntry
A = object.__new__(ta.ToolAgent)
A._session_runtime_dir = Path('/tmp/g1'); A._tokens_at_level_start = 0; A._session_generated_tokens = 0
A._history_messages = [{'role': 'user', 'content': 'u'}]; A._last_step_summary = None; A._kept_functions = {'f': 'def f(): pass'}
A._retained_clear_notice = ''; A._resume_after_yield = False; A._current_valid_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'MOUSE', 'UNDO']
VALID = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'MOUSE', 'UNDO']
class FakeInner:
    max_runtime_s_per_game = 7260.0
    def soft_time_remaining_seconds(self): return None
class FakeSolver:
    def __init__(self):
        self.solver = FakeInner(); self.started_at = time.monotonic(); self.level = 1
        self.history_entries = [HistoryEntry(action='', frame=Frame(grid=((0,),), step=0, level=1))]
    def current_frame(self): return Frame(grid=((0,),), step=len(self.history_entries), level=self.level)
    def should_stop(self): return False
    def step_env(self, args): return {}
    def act(self, name, k=1, no_op=False, game_over=False, level_completed=False, data=None, board=None):
        for i in range(k):
            g = ((len(self.history_entries) if board is None else board,),)
            r = {'action_name': name, 'level': self.level, 'no_op': no_op, 'game_over': game_over and i == k - 1,
                 'level_completed': level_completed and i == k - 1, 'valid_actions': VALID, 'action_data': data or {}}
            self.history_entries.append(HistoryEntry(action=name, frame=Frame(grid=g, step=0, level=self.level), result=r))
S = FakeSolver(); step = S.step_env
A._sdt_maybe_intervene(step); assert A._sdt_count == 0 and A._retained_clear_notice == ''         # first turn: mark only
S.act('ACTION4', 6); S.act('ACTION3', 3, no_op=True); S.act('ACTION6', 4, data={'x': 1, 'y': 2})
A._session_generated_tokens = 59000; A._sdt_maybe_intervene(step); assert A._sdt_count == 0         # below 60K
A._session_generated_tokens = 61000
A._resume_after_yield = True; A._sdt_maybe_intervene(step); assert A._sdt_count == 0                # skip mid-turn
A._resume_after_yield = False; A._sdt_maybe_intervene(step); assert A._sdt_count == 1
n1 = A._retained_clear_notice
assert n1.startswith('[Harness record, level 1] 13 actions (RIGHT 6, MOUSE 4, LEFT 3)'), n1[:120]
assert 'About 61K reasoning tokens spent on this level so far, 4692 per action.' in n1, n1
assert 'never used on this level: UP, DOWN' in n1 and 'UNDO' not in n1.split('never used on this level:')[1].split(';')[0], n1
assert '4 MOUSE click(s) on 1 distinct target(s)' in n1 and 'boards' not in n1 and '3 action(s) that changed nothing' in n1 and '0 game over(s)' in n1
assert 'trouble with the first level' in n1 and '3. Formulate a new hypothesis about the goal of the level.' in n1
assert '5. Prefer a short experiment' in n1   # 61000/13 tokens per action > 1500
assert A._tokens_at_level_start == 0 and A._history_messages == [{'role': 'user', 'content': 'u'}] and A._kept_functions == {'f': 'def f(): pass'}
A._retained_clear_notice = ''   # consumed by the next user message
A._session_generated_tokens = 100000; A._sdt_maybe_intervene(step); assert A._sdt_count == 1        # needs 50K more
A._session_generated_tokens = 111500; A._sdt_maybe_intervene(step); assert A._sdt_count == 2
A._retained_clear_notice = 'Your retained functions remain available.'
A._session_generated_tokens = 200000; A._sdt_maybe_intervene(step); assert A._sdt_count == 2        # max 2
assert A._retained_clear_notice == 'Your retained functions remain available.'
# level 1 completes: scheduler moves the clock, SDT logs cleared_after_sdt and records the level cost
S.act('ACTION4', 1, level_completed=True); S.level = 2
A._tokens_at_level_start = 210000; A._session_generated_tokens = 210000; A._retained_clear_notice = ''
A._sdt_maybe_intervene(step); assert A._sdt_count == 0 and A._sdt_level_costs == [210000]
# level 2: 61 moves, then a death (auto-reset follows) -> death trigger at 25K level tokens
S.act('ACTION1', 30); S.act('ACTION2', 31, game_over=True); S.act('RESET', 1)
A._last_step_summary = {'game_over': True, 'level': 2}
A._session_generated_tokens = 235000; A._sdt_maybe_intervene(step); assert A._sdt_count == 1
n2 = A._retained_clear_notice
assert n2.startswith('[Harness record, level 2] 61 actions (DOWN 31, UP 30)'), n2[:100]
assert '1 game over(s)' in n2 and 'restarted from its initial state' in n2 and 'trouble with this level' in n2
assert 'Levels you completed in this game took about 210K tokens each.' in n2 and 'how to solve the level' in n2
assert '5. Before moving again' in n2   # 25000/61 = 409 tokens per action, >= 60 moves
A._retained_clear_notice = ''
A._session_generated_tokens = 236000; A._sdt_maybe_intervene(step); assert A._sdt_count == 1        # same death, no repeat
# time guard: little time left -> no fire
S.started_at = time.monotonic() - 7000; A._last_step_summary = None
A._session_generated_tokens = 300000; A._sdt_maybe_intervene(step); assert A._sdt_count == 1
# new game resets the per-game state
A._session_runtime_dir = Path('/tmp/g2'); A._tokens_at_level_start = 300000
A._sdt_maybe_intervene(step); assert A._sdt_count == 0 and A._sdt_level_costs == []
# a solver without history disables the arm instead of raising
A._session_generated_tokens = 400000
A._sdt_maybe_intervene(lambda x: x); assert A._sdt_disabled is True
# analyze() wiring: the hook sits right after valid actions are normalized
import inspect
src = inspect.getsource(ta.ToolAgent.analyze)
assert src.index('_normalize_valid_actions(valid_actions)') < src.index('self._sdt_maybe_intervene(step_env)') < src.index('load_runtime_state(state_path)')
ev = [json.loads(l) for l in open('/tmp/sdt_events.jsonl')]
assert [e['event'] for e in ev] == ['fire', 'fire', 'cleared_after_sdt', 'fire'] and ev[3]['reason'] == 'death', ev
print(n1); print('---'); print(n2); print('---')
print('ALL OK')
