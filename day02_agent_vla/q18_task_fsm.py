"""
【Q18】长序列任务状态机（带超时/回退/重试）

要求：
1. TaskFSM 类，状态：IDLE -> PLAN -> EXECUTE -> VERIFY -> DONE / ERROR
2. __init__(self, max_retries: int = 3, timeout: float = 10.0)
   - timeout 是秒数，记录进入当前状态的时间戳（time.time()）
3. transition(event: str) -> str：
   - event 可以是："start", "success", "failure", "verify_ok", "verify_fail", "timeout"
   - IDLE + "start" -> PLAN
   - PLAN + "success" -> EXECUTE
   - EXECUTE + "success" -> VERIFY
   - EXECUTE + "failure" -> PLAN（重试计数+1，若超过 max_retries -> ERROR）
   - VERIFY + "verify_ok" -> DONE
   - VERIFY + "verify_fail" -> EXECUTE（重试计数+1）
   - 任何状态 + "timeout" -> ERROR
4. 每次 transition 检查：若 time.time() - state_enter_time > timeout，强制转 ERROR
5. get_state() -> 当前状态字符串
6. is_terminal() -> 是否 DONE 或 ERROR

C++ 程序员注意：
- time.time() 返回 float 秒数（类似 C++ chrono::system_clock）
- 状态用字符串常量即可，不需要 enum（笔试节省时间）
- 重试计数器 retry_count 在 EXECUTE failure 和 VERIFY fail 时分别计数
- 进入新状态时必须更新 state_enter_time = time.time()
"""

import time
import pytest


class TaskFSM:
    def __init__(self, max_retries: int = 3, timeout: float = 10.0):
        # TODO:
        # 1. self.state = "IDLE"
        # 2. self.max_retries = max_retries
        # 3. self.timeout = timeout
        # 4. self.retry_count = 0
        # 5. self.state_enter_time = time.time()
        pass

    def transition(self, event: str) -> str:
        # TODO:
        # 1. 先检查超时：若 time.time() - state_enter_time > timeout -> ERROR
        # 2. 根据当前状态和 event 做状态转换
        # 3. 若转换到新状态，更新 state_enter_time 和 retry_count（如需）
        # 4. 返回新状态
        pass

    def get_state(self) -> str:
        # TODO
        pass

    def is_terminal(self) -> bool:
        # TODO: DONE 或 ERROR
        pass


class TestTaskFSM:
    def test_happy_path(self):
        fsm = TaskFSM()
        assert fsm.transition("start") == "PLAN"
        assert fsm.transition("success") == "EXECUTE"
        assert fsm.transition("success") == "VERIFY"
        assert fsm.transition("verify_ok") == "DONE"
        assert fsm.is_terminal() is True

    def test_execute_failure_retry(self):
        fsm = TaskFSM(max_retries=2)
        fsm.transition("start")
        fsm.transition("success")  # PLAN -> EXECUTE

        assert fsm.transition("failure") == "PLAN"  # 第 1 次重试
        fsm.transition("success")
        assert fsm.transition("failure") == "PLAN"  # 第 2 次重试
        fsm.transition("success")
        assert fsm.transition("failure") == "ERROR"  # 第 3 次，超了

    def test_timeout(self):
        fsm = TaskFSM(timeout=0.01)
        fsm.transition("start")  # 进入 PLAN
        time.sleep(0.02)  # 确保超时
        assert fsm.transition("success") == "ERROR"  # 超时强制 ERROR

    def test_terminal(self):
        fsm = TaskFSM()
        assert fsm.is_terminal() is False
        fsm.transition("start")
        assert fsm.is_terminal() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
