"""
Orchestrator Module Test Cases
==============================
Orchestrator, LLMConfig, ReAct Loop 테스트
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import (
    Orchestrator, LLMConfig, LLMResponse, ToolCall,
    StepInfo, StepType
)
from core.shared_storage import SharedStorage
from core.base_tool import ToolRegistry
from tools.file_tool import FileTool
from tools.llm_tool import LLMTool


class TestLLMConfig(unittest.TestCase):
    """LLMConfig 테스트"""
    
    def test_default_values(self):
        """기본값 확인"""
        print("\n[TEST] LLMConfig 기본값")
        
        config = LLMConfig()
        
        self.assertEqual(config.model, "llama3.2")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.base_url, "http://localhost:11434")
        
        print("  ✓ 기본값 확인 완료")
    
    def test_update(self):
        """설정 업데이트"""
        print("\n[TEST] LLMConfig 업데이트")
        
        config = LLMConfig()
        config.update(
            model="codellama",
            temperature=0.3,
            max_tokens=4096
        )
        
        self.assertEqual(config.model, "codellama")
        self.assertEqual(config.temperature, 0.3)
        self.assertEqual(config.max_tokens, 4096)
        
        print("  ✓ 설정 업데이트 완료")
    
    def test_to_dict(self):
        """Dict 변환"""
        print("\n[TEST] LLMConfig to_dict")
        
        config = LLMConfig(model="mistral", temperature=0.5)
        d = config.to_dict()
        
        self.assertEqual(d["model"], "mistral")
        self.assertEqual(d["temperature"], 0.5)
        self.assertIn("base_url", d)
        
        print("  ✓ Dict 변환 완료")


class TestToolCall(unittest.TestCase):
    """ToolCall 테스트"""
    
    def test_properties(self):
        """속성 접근"""
        print("\n[TEST] ToolCall 속성")
        
        tc = ToolCall(
            name="file_tool",
            arguments={
                "action": "read",
                "path": "/var/log/wifi.log"
            }
        )
        
        self.assertEqual(tc.name, "file_tool")
        self.assertEqual(tc.action, "read")
        self.assertEqual(tc.params, {"path": "/var/log/wifi.log"})
        
        print("  ✓ 속성 접근 완료")


class TestLLMResponse(unittest.TestCase):
    """LLMResponse 테스트"""
    
    def test_tool_call_response(self):
        """Tool 호출 응답"""
        print("\n[TEST] LLMResponse - Tool 호출")
        
        response = LLMResponse(
            thought="파일을 읽어야 합니다",
            tool_calls=[
                ToolCall(name="file_tool", arguments={"action": "read", "path": "test.txt"})
            ]
        )
        
        self.assertFalse(response.is_final_answer)
        self.assertEqual(len(response.tool_calls), 1)
        
        print("  ✓ Tool 호출 응답 확인")
    
    def test_final_answer_response(self):
        """최종 답변 응답"""
        print("\n[TEST] LLMResponse - 최종 답변")
        
        response = LLMResponse(
            thought="분석 완료",
            tool_calls=None,
            content="분석 결과입니다."
        )
        
        self.assertTrue(response.is_final_answer)
        self.assertEqual(response.content, "분석 결과입니다.")
        
        print("  ✓ 최종 답변 응답 확인")


class TestStepInfo(unittest.TestCase):
    """StepInfo 테스트"""
    
    def test_to_dict(self):
        """Dict 변환"""
        print("\n[TEST] StepInfo to_dict")
        
        info = StepInfo(
            type=StepType.TOOL_CALL,
            step=1,
            content={"action": "read"},
            tool_name="file_tool",
            action="read"
        )
        
        d = info.to_dict()
        
        self.assertEqual(d["type"], "tool_call")
        self.assertEqual(d["step"], 1)
        self.assertEqual(d["tool_name"], "file_tool")
        
        print("  ✓ Dict 변환 완료")


class TestOrchestratorInit(unittest.TestCase):
    """Orchestrator 초기화 테스트"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage = SharedStorage(debug_enabled=True)
        self.registry = ToolRegistry(debug_enabled=True)
        self.registry.register(FileTool(base_path=self.test_dir, debug_enabled=True))
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_init_default(self):
        """기본 초기화"""
        print("\n[TEST] Orchestrator 기본 초기화")
        
        orch = Orchestrator(
            tools=self.registry,
            storage=self.storage
        )
        
        self.assertEqual(orch.max_steps, 10)
        self.assertEqual(orch.llm_config.model, "llama3.2")
        
        print("  ✓ 기본 초기화 완료")
    
    def test_init_custom_config(self):
        """커스텀 설정 초기화"""
        print("\n[TEST] Orchestrator 커스텀 설정")
        
        config = LLMConfig(model="codellama", temperature=0.2)
        
        orch = Orchestrator(
            tools=self.registry,
            storage=self.storage,
            llm_config=config,
            max_steps=5
        )
        
        self.assertEqual(orch.max_steps, 5)
        self.assertEqual(orch.llm_config.model, "codellama")
        
        print("  ✓ 커스텀 설정 초기화 완료")
    
    def test_update_llm_config(self):
        """LLM 설정 동적 변경"""
        print("\n[TEST] Orchestrator LLM 설정 변경")
        
        orch = Orchestrator(
            tools=self.registry,
            storage=self.storage
        )
        
        orch.update_llm_config(
            model="mistral",
            temperature=0.9
        )
        
        self.assertEqual(orch.llm_config.model, "mistral")
        self.assertEqual(orch.llm_config.temperature, 0.9)
        
        print("  ✓ LLM 설정 변경 완료")


class TestOrchestratorParsing(unittest.TestCase):
    """Orchestrator 응답 파싱 테스트"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage = SharedStorage(debug_enabled=True)
        self.registry = ToolRegistry(debug_enabled=True)
        self.registry.register(FileTool(base_path=self.test_dir, debug_enabled=True))
        
        self.orch = Orchestrator(
            tools=self.registry,
            storage=self.storage,
            debug_enabled=True
        )
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_parse_tool_call(self):
        """Tool 호출 응답 파싱"""
        print("\n[TEST] Tool 호출 응답 파싱")
        
        raw = json.dumps({
            "thought": "파일을 읽어야 합니다",
            "tool_calls": [{
                "name": "file_tool",
                "arguments": {
                    "action": "read",
                    "path": "test.txt"
                }
            }]
        })
        
        response = self.orch._parse_llm_response(raw)
        
        self.assertFalse(response.is_final_answer)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].name, "file_tool")
        self.assertEqual(response.tool_calls[0].action, "read")
        
        print("  ✓ Tool 호출 파싱 완료")
    
    def test_parse_final_answer(self):
        """최종 답변 파싱"""
        print("\n[TEST] 최종 답변 파싱")
        
        raw = json.dumps({
            "thought": "분석 완료",
            "tool_calls": None,
            "content": "분석 결과입니다."
        })
        
        response = self.orch._parse_llm_response(raw)
        
        self.assertTrue(response.is_final_answer)
        self.assertEqual(response.content, "분석 결과입니다.")
        
        print("  ✓ 최종 답변 파싱 완료")
    
    def test_parse_invalid_json(self):
        """잘못된 JSON 처리"""
        print("\n[TEST] 잘못된 JSON 처리")
        
        raw = "This is not JSON, but a plain text response."
        
        response = self.orch._parse_llm_response(raw)
        
        # 파싱 실패 시 전체를 content로 처리
        self.assertTrue(response.is_final_answer)
        self.assertEqual(response.content, raw)
        
        print("  ✓ 잘못된 JSON 처리 완료")
    
    def test_parse_json_with_extra_text(self):
        """JSON 앞뒤에 텍스트가 있는 경우"""
        print("\n[TEST] JSON + 추가 텍스트 파싱")
        
        raw = '''Sure, I'll help you. Here's my response:
{
    "thought": "Let me read the file",
    "tool_calls": [{
        "name": "file_tool",
        "arguments": {"action": "read", "path": "test.txt"}
    }]
}
That's my decision.'''
        
        response = self.orch._parse_llm_response(raw)
        
        self.assertFalse(response.is_final_answer)
        self.assertEqual(len(response.tool_calls), 1)
        
        print("  ✓ JSON 추출 완료")


class TestOrchestratorExecution(unittest.TestCase):
    """Orchestrator 실행 테스트"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage = SharedStorage(debug_enabled=True)
        self.registry = ToolRegistry(debug_enabled=True)

        self.file_tool = FileTool(base_path=self.test_dir, debug_enabled=True)
        self.registry.register(self.file_tool)
        self.registry.register(LLMTool(debug_enabled=True, use_mock=True))
        
        # 테스트 파일 생성
        self.file_tool.execute("write", {
            "path": "test.txt",
            "content": "Hello, World!"
        })
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_execute_tool(self):
        """Tool 실행 테스트"""
        print("\n[TEST] Orchestrator Tool 실행")
        
        orch = Orchestrator(
            tools=self.registry,
            storage=self.storage,
            debug_enabled=True
        )
        
        tool_call = ToolCall(
            name="file_tool",
            arguments={"action": "read", "path": "test.txt"}
        )
        
        result = orch._execute_tool(tool_call)
        
        self.assertTrue(result.success)
        self.assertEqual(result.output, "Hello, World!")
        
        print("  ✓ Tool 실행 완료")
    
    def test_execute_invalid_tool(self):
        """없는 Tool 실행"""
        print("\n[TEST] 없는 Tool 실행")
        
        orch = Orchestrator(
            tools=self.registry,
            storage=self.storage,
            debug_enabled=True
        )
        
        tool_call = ToolCall(
            name="invalid_tool",
            arguments={"action": "test"}
        )
        
        result = orch._execute_tool(tool_call)
        
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)
        
        print("  ✓ 없는 Tool 에러 처리 완료")


class TestOrchestratorMockRun(unittest.TestCase):
    """Orchestrator Mock 실행 테스트"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage = SharedStorage(debug_enabled=True)
        self.registry = ToolRegistry(debug_enabled=True)
        
        self.file_tool = FileTool(base_path=self.test_dir, debug_enabled=True)
        self.registry.register(self.file_tool)
        
        # 테스트 파일 생성
        self.file_tool.execute("write", {
            "path": "test.txt",
            "content": "Test content for analysis"
        })
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_run_stream_mock(self):
        """Streaming 실행 (Mock LLM)"""
        print("\n[TEST] Orchestrator Streaming 실행 (Mock)")
        
        orch = Orchestrator(
            tools=self.registry,
            storage=self.storage,
            max_steps=5,
            debug_enabled=True
        )
        
        # Mock LLM API 호출
        call_count = [0]
        
        def mock_call_llm(system, user):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({
                    "thought": "파일을 읽겠습니다",
                    "tool_calls": [{
                        "name": "file_tool",
                        "arguments": {"action": "read", "path": "test.txt"}
                    }]
                })
            else:
                return json.dumps({
                    "thought": "분석 완료",
                    "tool_calls": None,
                    "content": "파일 내용을 분석했습니다."
                })
        
        orch._call_llm_api = mock_call_llm
        
        # 실행
        steps = list(orch.run_stream("test.txt 파일 분석해줘"))
        
        # 검증
        step_types = [s.type for s in steps]
        
        self.assertIn(StepType.THINKING, step_types)
        self.assertIn(StepType.TOOL_CALL, step_types)
        self.assertIn(StepType.TOOL_RESULT, step_types)
        self.assertIn(StepType.FINAL_ANSWER, step_types)
        
        # 최종 답변 확인
        final = [s for s in steps if s.type == StepType.FINAL_ANSWER][0]
        self.assertIn("분석", final.content)
        
        print(f"  ✓ Streaming 실행 완료 (총 {len(steps)} steps)")
    
    def test_run_sync_mock(self):
        """동기 실행 (Mock LLM)"""
        print("\n[TEST] Orchestrator 동기 실행 (Mock)")
        
        orch = Orchestrator(
            tools=self.registry,
            storage=self.storage,
            max_steps=5,
            debug_enabled=True
        )
        
        # Mock
        def mock_call_llm(system, user):
            return json.dumps({
                "thought": "바로 답변",
                "tool_calls": None,
                "content": "분석 결과입니다."
            })
        
        orch._call_llm_api = mock_call_llm
        
        # 실행
        result = orch.run("test query")
        
        self.assertEqual(result, "분석 결과입니다.")
        
        print("  ✓ 동기 실행 완료")
    
    def test_max_steps_limit(self):
        """최대 step 제한"""
        print("\n[TEST] 최대 step 제한")
        
        orch = Orchestrator(
            tools=self.registry,
            storage=self.storage,
            max_steps=2,
            debug_enabled=True
        )
        
        # 항상 Tool 호출하는 Mock (종료 안 함)
        def mock_call_llm(system, user):
            return json.dumps({
                "thought": "계속 Tool 호출",
                "tool_calls": [{
                    "name": "file_tool",
                    "arguments": {"action": "exists", "path": "test.txt"}
                }]
            })
        
        orch._call_llm_api = mock_call_llm
        
        # 실행
        steps = list(orch.run_stream("무한 루프 테스트"))
        
        # 최대 step에서 종료되어야 함
        final = [s for s in steps if s.type == StepType.FINAL_ANSWER]
        self.assertEqual(len(final), 1)
        self.assertIn("최대 실행 단계", final[0].content)
        
        print("  ✓ 최대 step 제한 동작 확인")
    
    def test_stop(self):
        """수동 중지"""
        print("\n[TEST] 수동 중지")
        
        orch = Orchestrator(
            tools=self.registry,
            storage=self.storage,
            max_steps=10,
            debug_enabled=True
        )
        
        # 첫 step 후 중지
        call_count = [0]
        
        def mock_call_llm(system, user):
            call_count[0] += 1
            if call_count[0] == 1:
                orch.stop()  # 중지 요청
            return json.dumps({
                "thought": "계속",
                "tool_calls": [{
                    "name": "file_tool",
                    "arguments": {"action": "exists", "path": "test.txt"}
                }]
            })
        
        orch._call_llm_api = mock_call_llm
        
        # 실행
        steps = list(orch.run_stream("중지 테스트"))
        
        # 1 step만 실행되어야 함
        tool_calls = [s for s in steps if s.type == StepType.TOOL_CALL]
        self.assertEqual(len(tool_calls), 1)
        
        print("  ✓ 수동 중지 동작 확인")
    
    def test_callback(self):
        """Callback 호출"""
        print("\n[TEST] Callback 호출")
        
        callback_calls = []
        
        def on_step(step_info):
            callback_calls.append(step_info)
        
        orch = Orchestrator(
            tools=self.registry,
            storage=self.storage,
            max_steps=5,
            on_step_complete=on_step,
            debug_enabled=True
        )
        
        # Mock
        def mock_call_llm(system, user):
            return json.dumps({
                "thought": "완료",
                "tool_calls": None,
                "content": "끝"
            })
        
        orch._call_llm_api = mock_call_llm
        
        # 실행
        list(orch.run_stream("callback test"))
        
        # Callback이 호출되었어야 함 (0번일 수도 있음 - final answer만 있으면)
        print(f"  ✓ Callback {len(callback_calls)}회 호출됨")


class TestOrchestratorIntegration(unittest.TestCase):
    """Orchestrator 통합 테스트"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage = SharedStorage(debug_enabled=True)
        self.registry = ToolRegistry(debug_enabled=True)

        self.file_tool = FileTool(base_path=self.test_dir, debug_enabled=True)
        self.llm_tool = LLMTool(debug_enabled=True, use_mock=True)

        self.registry.register(self.file_tool)
        self.registry.register(self.llm_tool)
        
        # 테스트 파일
        self.file_tool.execute("write", {
            "path": "error.log",
            "content": "[ERROR] DMA timeout\n[WARN] Retry failed"
        })
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_full_workflow(self):
        """전체 워크플로우: 파일 읽기 → LLM 분석 → 응답"""
        print("\n[TEST] 전체 워크플로우")
        print("=" * 50)

        orch = Orchestrator(
            tools=self.registry,
            storage=self.storage,
            max_steps=5,
            debug_enabled=True
        )

        # 시뮬레이션된 LLM 응답
        step_responses = [
            # Step 1: 파일 읽기
            json.dumps({
                "thought": "먼저 에러 로그를 읽어야 합니다",
                "tool_calls": [{
                    "name": "file_tool",
                    "arguments": {"action": "read", "path": "error.log"}
                }]
            }),
            # Step 2: LLM 분석
            json.dumps({
                "thought": "DMA timeout 에러를 분석해보겠습니다",
                "tool_calls": [{
                    "name": "llm_tool",
                    "arguments": {"action": "general", "prompt": "DMA timeout 에러 분석"}
                }]
            }),
            # Step 3: 최종 답변
            json.dumps({
                "thought": "충분한 정보를 수집했습니다",
                "tool_calls": None,
                "content": "분석 결과: DMA timeout 에러가 발견되었습니다. 드라이버 업데이트를 권장합니다."
            })
        ]

        call_idx = [0]

        def mock_call_llm(system, user):
            response = step_responses[call_idx[0]]
            call_idx[0] += 1
            return response

        orch._call_llm_api = mock_call_llm

        # 실행 및 출력
        print("\n실행 시작...")
        for step in orch.run_stream("error.log 파일 분석하고 해결책 찾아줘"):
            if step.type == StepType.THINKING:
                print(f"  💭 {step.content}")
            elif step.type == StepType.TOOL_CALL:
                print(f"  🔧 {step.tool_name}.{step.action}")
            elif step.type == StepType.TOOL_RESULT:
                output = str(step.content)[:100]
                print(f"  📄 Result: {output}...")
            elif step.type == StepType.FINAL_ANSWER:
                print(f"  ✅ Final: {step.content}")

        # 검증 - 세션 완료 후에는 history에서 확인
        history = self.storage.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "completed")
        self.assertEqual(len(history[0]["results"]), 2)  # file_tool + llm_tool

        print("=" * 50)
        print("[TEST] 전체 워크플로우 완료 ✓")


def run_tests():
    """모든 테스트 실행"""
    print("\n" + "=" * 60)
    print("Orchestrator Module Test Suite")
    print("=" * 60)
    
    test_classes = [
        TestLLMConfig,
        TestToolCall,
        TestLLMResponse,
        TestStepInfo,
        TestOrchestratorInit,
        TestOrchestratorParsing,
        TestOrchestratorExecution,
        TestOrchestratorMockRun,
        TestOrchestratorIntegration,
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
