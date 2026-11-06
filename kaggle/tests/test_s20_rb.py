import os, sys, json, time, types
os.environ['ARC3_RB']='1'; os.environ['ARC3_RB_TOKENS']='1000'; os.environ['ARC3_RB_LOG']='/tmp/rb_events.jsonl'
os.environ['ARC3_PERSISTENT_FUNCTIONS']='1'
from pathlib import Path
from inference.agent import tool_agent as ta
A = object.__new__(ta.ToolAgent)
A._session_runtime_dir = Path('/tmp/g1'); A._tokens_at_level_start = 0; A._session_generated_tokens = 0
A._history_messages = []; A._last_step_summary = None; A._last_action_call_result = None
A._summarized_knowledge = {'world_model': ''}; A._kept_functions = {}; A._retained_clear_notice = ''
A._pending_wm_rebuild_reason = ''; A._pending_wm_revise_reason = ''; A._turns_without_wm_update = 0
A._resume_after_yield = False; A._auto_diff_prev_frame = 'X'
class FakeInner:
    max_runtime_s_per_game = 7260.0
    def soft_time_remaining_seconds(self): return None
class FakeSolver:
    def __init__(self): self.solver = FakeInner(); self.started_at = time.monotonic(); self.action_count = 0; self.resets = 0; self.writes = 0
    def _execute_auto_reset(self): self.resets += 1; self.action_count += 1
    def write_runtime_state(self): self.writes += 1
    def should_stop(self): return False
    def step_env(self, args): return {}
S = FakeSolver(); step = S.step_env
# turn 1 level 1: snapshot (empty history)
A._rb_maybe_rollback(step); assert A._rb_count == 0 and A._rb_snapshot['history_messages'] == []
# model plays: history grows, tokens grow below threshold
A._history_messages = [{'role':'user','content':'u1'},{'role':'assistant','content':'a1'}]; A._session_generated_tokens = 800
A._rb_maybe_rollback(step); assert S.resets == 0
# level 1 completes inside a turn: tokens_at_level_start moves
A._session_generated_tokens = 900; A._tokens_at_level_start = 900; A._kept_functions = {'f':'def f(): pass'}
A._last_step_summary = {'level_transition': True, 'level': 2}
A._history_messages = A._history_messages + [{'role':'user','content':'u2'},{'role':'assistant','content':'a2'}]
A._rb_maybe_rollback(step); assert A._rb_count == 0 and len(A._rb_snapshot['history_messages']) == 4, A._rb_snapshot
# stall on level 2
A._history_messages = A._history_messages + [{'role':'user','content':'u3'},{'role':'assistant','content':'stuck'}]*5
A._kept_functions = {'f':'def f(): pass', 'g':'def g(): wrong'}; A._summarized_knowledge = {'world_model': 'wrong theory'}
A._last_step_summary = {'level': 2, 'gameplay_changed': True}
A._session_generated_tokens = 2500
A._resume_after_yield = True; A._rb_maybe_rollback(step); assert S.resets == 0   # skip on resume
A._resume_after_yield = False; A._rb_maybe_rollback(step)
assert S.resets == 1 and S.writes == 1, (S.resets, S.writes)
assert len(A._history_messages) == 4 and A._kept_functions == {'f':'def f(): pass'} and A._summarized_knowledge == {'world_model': ''}
assert A._last_step_summary == {'level_transition': True, 'level': 2} and A._auto_diff_prev_frame is None
assert A._retained_clear_notice.startswith('Note: the harness reset this level'), A._retained_clear_notice
assert A._tokens_at_level_start == 2500 and A._rb_count == 1
# next turn: no new snapshot, no rollback until 1000 more tokens
A._rb_maybe_rollback(step); assert S.resets == 1 and len(A._rb_snapshot['history_messages']) == 4
A._session_generated_tokens = 3600; A._rb_maybe_rollback(step); assert S.resets == 2 and A._rb_count == 2
A._session_generated_tokens = 5000; A._rb_maybe_rollback(step); assert S.resets == 2   # max 2
# level 2 cleared later -> cleared_after_rb event + new snapshot
A._tokens_at_level_start = 5200; A._session_generated_tokens = 5200; A._rb_maybe_rollback(step); assert A._rb_count == 0
# time guard: little time left -> no rollback
S.started_at = time.monotonic() - 7000; A._session_generated_tokens = 7000; A._rb_maybe_rollback(step); assert S.resets == 2
# error path disables arm
A2 = object.__new__(ta.ToolAgent); A2.__dict__.update(A.__dict__); A2._rb_mark = None
print(open('/tmp/rb_events.jsonl').read())
print('ALL OK')
