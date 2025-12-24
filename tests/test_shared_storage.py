"""
SharedStorage Module Test Cases
================================
SharedStorage의 모든 기능을 검증하는 테스트 케이스
"""

import os
import sys
import tempfile
import unittest

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.shared_storage import SharedStorage, Context, SessionHistory
from core.base_tool import ToolResult


def create_tool_result(step, executor, executor_type, action, input, output, status="success", error_message=None):
    """헬퍼 함수: ToolResult 객체 생성"""
    success = (status == "success")
    result = ToolResult(
        step = step,
        success=success,
        output=output,
        error=error_message,
        executor=executor,
        executor_type=executor_type,
        action=action,
        input=input
    )
    return result


class TestSharedStorageBasic(unittest.TestCase):
    """기본 기능 테스트"""

    def setUp(self):
        """각 테스트 전 실행"""
        self.storage = SharedStorage(debug_enabled=True)

    def tearDown(self):
        """각 테스트 후 실행"""
        self.storage.reset()
    
    def test_01_initialization(self):
        """초기화 테스트"""
        print("\n[TEST] 초기화 테스트")
        
        self.assertIsNone(self.storage.get_context())
        self.assertEqual(self.storage.get_results(), [])
        self.assertEqual(self.storage.get_history(), [])
        
        print("  ✓ 초기 상태 검증 완료")
    
    def test_02_start_session(self):
        """세션 시작 테스트"""
        print("\n[TEST] 세션 시작 테스트")
        
        user_query = "wifi.log 파일 읽어서 분석해줘"
        plan = ["file_tool로 파일 읽기", "llm_tool로 분석"]
        
        session_id = self.storage.start_session(user_query, plan)
        
        self.assertIsNotNone(session_id)
        self.assertEqual(len(session_id), 8)
        
        context = self.storage.get_context()
        self.assertEqual(context["user_query"], user_query)
        self.assertEqual(context["current_plan"], plan)
        self.assertEqual(context["current_step"], 0)
        
        print(f"  ✓ 세션 생성됨: {session_id}")
        print(f"  ✓ Context 검증 완료")
    
    def test_03_update_plan(self):
        """계획 업데이트 테스트"""
        print("\n[TEST] 계획 업데이트 테스트")
        
        self.storage.start_session("테스트 쿼리")
        
        new_plan = ["Step 1", "Step 2", "Step 3"]
        self.storage.update_plan(new_plan)
        
        context = self.storage.get_context()
        self.assertEqual(context["current_plan"], new_plan)
        
        print("  ✓ 계획 업데이트 완료")
    
    def test_04_advance_step(self):
        """Step 이동 테스트"""
        print("\n[TEST] Step 이동 테스트")
        
        self.storage.start_session("테스트 쿼리", ["Step 1", "Step 2"])
        
        self.assertEqual(self.storage.get_context()["current_step"], 0)
        
        self.storage.advance_step()
        self.assertEqual(self.storage.get_context()["current_step"], 1)
        
        self.storage.advance_step()
        self.assertEqual(self.storage.get_context()["current_step"], 2)
        
        print("  ✓ Step 이동 완료")


class TestSharedStorageResults(unittest.TestCase):
    """Results 관리 테스트"""
    
    def setUp(self):
        self.storage = SharedStorage(debug_enabled=True)
        self.storage.start_session("테스트 쿼리", ["Step 1", "Step 2", "Step 3"])
    
    def tearDown(self):
        self.storage.reset()
    
    def test_01_add_result(self):
        """결과 추가 테스트"""
        print("\n[TEST] 결과 추가 테스트")

        tool_result = create_tool_result(
            step=1,
            executor="file_tool",
            executor_type="tool",
            action="read",
            input={"path": "/var/log/wifi.log"},
            output="로그 내용입니다...",
            status="success"
        )

        result = self.storage.add_result(tool_result)

        self.assertEqual(result.step, 1)
        self.assertEqual(result.executor, "file_tool")
        self.assertEqual(result.status, "success")

        results = self.storage.get_results()
        self.assertEqual(len(results), 1)
        
        print("  ✓ 결과 추가 완료")
    
    def test_02_multiple_results(self):
        """여러 결과 추가 테스트"""
        print("\n[TEST] 여러 결과 추가 테스트")
        
        # Step 1: file_tool
        self.storage.add_result(create_tool_result(
            step=1,
            executor="file_tool",
            executor_type="tool",
            action="read",
            input={"path": "test.txt"},
            output="파일 내용"
        ))
        
        # Step 2: llm_tool
        self.storage.add_result(create_tool_result(
            step=2,
            executor="llm_tool",
            executor_type="tool",
            action="analyze",
            input={"content": "파일 내용"},
            output="분석 결과입니다"
        ))
        
        # Step 3: llm_tool
        self.storage.add_result(create_tool_result(
            step=3,
            executor="llm_tool",
            executor_type="tool",
            action="general",
            input={"prompt": "wifi driver 문제 해결 방법"},
            output="드라이버 재설치를 권장합니다"
        ))
        
        results = self.storage.get_results()
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["step"], 1)
        self.assertEqual(results[1]["step"], 2)
        self.assertEqual(results[2]["step"], 3)
        
        print("  ✓ 3개 결과 추가 완료")
    
    def test_03_get_last_result(self):
        """마지막 결과 조회 테스트"""
        print("\n[TEST] 마지막 결과 조회 테스트")
        
        self.storage.add_result(create_tool_result(
            step=1,
            executor="tool_a", executor_type="tool",
            action="action_a", input={}, output="output_a"
        ))
        self.storage.add_result(create_tool_result(
            step=2,
            executor="tool_b", executor_type="tool",
            action="action_b", input={}, output="output_b"
        ))
        
        last = self.storage.get_last_result()
        self.assertEqual(last["executor"], "tool_b")
        self.assertEqual(last["output"], "output_b")
        
        last_output = self.storage.get_last_output()
        self.assertEqual(last_output, "output_b")
        
        print("  ✓ 마지막 결과 조회 완료")
    
    def test_04_get_result_by_step(self):
        """특정 Step 결과 조회 테스트"""
        print("\n[TEST] 특정 Step 결과 조회 테스트")
        
        self.storage.add_result(create_tool_result(
            step=1,
            executor="tool_1", executor_type="tool",
            action="action", input={}, output="output_1"
        ))
        self.storage.add_result(create_tool_result(
            step=2,
            executor="tool_2", executor_type="tool",
            action="action", input={}, output="output_2"
        ))
        self.storage.add_result(create_tool_result(
            step=3,
            executor="tool_3", executor_type="tool",
            action="action", input={}, output="output_3"
        ))
        
        result_2 = self.storage.get_result_by_step(2)
        self.assertEqual(result_2["executor"], "tool_2")
        
        output_2 = self.storage.get_output_by_step(2)
        self.assertEqual(output_2, "output_2")
        
        # 존재하지 않는 step
        result_99 = self.storage.get_result_by_step(99)
        self.assertIsNone(result_99)
        
        print("  ✓ 특정 Step 결과 조회 완료")
    
    def test_05_error_result(self):
        """에러 결과 추가 테스트"""
        print("\n[TEST] 에러 결과 추가 테스트")
        
        result = self.storage.add_result(create_tool_result(
            step=1,
            executor="file_tool",
            executor_type="tool",
            action="read",
            input={"path": "/not/exist"},
            output=None,
            status="error",
            error_message="File not found"
        ))
        
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error, "File not found")
        
        print("  ✓ 에러 결과 추가 완료")


class TestSharedStorageHistory(unittest.TestCase):
    """History 관리 테스트"""
    
    def setUp(self):
        self.storage = SharedStorage(debug_enabled=True)
    
    def tearDown(self):
        self.storage.reset()
    
    def test_01_complete_session(self):
        """세션 완료 테스트"""
        print("\n[TEST] 세션 완료 테스트")
        
        session_id = self.storage.start_session("쿼리 1", ["Step 1"])
        self.storage.add_result(create_tool_result(
            step=1,
            executor="tool", executor_type="tool",
            action="action", input={}, output="result"
        ))
        
        session = self.storage.complete_session("최종 응답입니다", "completed")
        
        self.assertEqual(session.session_id, session_id)
        self.assertEqual(session.status, "completed")
        self.assertEqual(len(session.results), 1)
        
        # History에 저장됨
        history = self.storage.get_history()
        self.assertEqual(len(history), 1)
        
        print("  ✓ 세션 완료 및 히스토리 저장 완료")
    
    def test_02_multiple_sessions(self):
        """여러 세션 히스토리 테스트"""
        print("\n[TEST] 여러 세션 히스토리 테스트")
        
        # 세션 1
        self.storage.start_session("쿼리 1")
        self.storage.add_result(create_tool_result(
            step=1,
            executor="tool", executor_type="tool",
            action="action", input={}, output="result 1"
        ))
        self.storage.complete_session("응답 1")
        
        # 세션 2
        self.storage.start_session("쿼리 2")
        self.storage.add_result(create_tool_result(
            step=1,
            executor="tool", executor_type="tool",
            action="action", input={}, output="result 2"
        ))
        self.storage.complete_session("응답 2")
        
        history = self.storage.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["user_query"], "쿼리 1")
        self.assertEqual(history[1]["user_query"], "쿼리 2")
        
        print("  ✓ 여러 세션 히스토리 저장 완료")
    
    def test_03_get_history_by_session_id(self):
        """세션 ID로 히스토리 조회 테스트"""
        print("\n[TEST] 세션 ID로 히스토리 조회 테스트")
        
        session_id = self.storage.start_session("특정 쿼리")
        self.storage.complete_session("특정 응답")
        
        found = self.storage.get_history_by_session_id(session_id)
        self.assertIsNotNone(found)
        self.assertEqual(found["user_query"], "특정 쿼리")
        
        not_found = self.storage.get_history_by_session_id("not-exist")
        self.assertIsNone(not_found)
        
        print("  ✓ 세션 ID로 히스토리 조회 완료")


class TestSharedStorageSummary(unittest.TestCase):
    """Summary 생성 테스트"""
    
    def setUp(self):
        self.storage = SharedStorage(debug_enabled=True)
    
    def tearDown(self):
        self.storage.reset()
    
    def test_01_get_summary_empty(self):
        """빈 상태 요약 테스트"""
        print("\n[TEST] 빈 상태 요약 테스트")

        summary = self.storage.get_summary()
        self.assertEqual(summary, "No active session")

        print("  ✓ 빈 상태 요약 완료")
    
    def test_02_get_summary_with_results(self):
        """결과 포함 요약 테스트"""
        print("\n[TEST] 결과 포함 요약 테스트")
        
        self.storage.start_session(
            "wifi.log 분석해줘",
            ["파일 읽기", "분석하기", "결과 정리"]
        )
        
        self.storage.add_result(create_tool_result(
            step=1,
            executor="file_tool", executor_type="tool",
            action="read", input={"path": "wifi.log"},
            output="[ERROR] Connection failed..."
        ))
        self.storage.advance_step()
        
        self.storage.add_result(create_tool_result(
            step=2,
            executor="llm_tool", executor_type="tool",
            action="analyze", input={"content": "..."},
            output="Connection 에러 발견, 원인은..."
        ))
        self.storage.advance_step()
        
        summary = self.storage.get_summary()
        
        # 검증
        self.assertIn("wifi.log 분석해줘", summary)
        self.assertIn("file_tool", summary)
        self.assertIn("llm_tool", summary)
        self.assertIn("✓", summary)  # 완료된 step 표시
        
        print("  ✓ 결과 포함 요약 생성 완료")
        print("\n--- Generated Summary ---")
        print(summary)
        print("--- End Summary ---\n")
    
    def test_03_get_summary_truncate(self):
        """긴 output truncate 테스트"""
        print("\n[TEST] 긴 output truncate 테스트")

        self.storage.start_session("테스트")

        # 매우 긴 output
        long_output = "A" * 1000
        self.storage.add_result(create_tool_result(
            step=1,
            executor="tool", executor_type="tool",
            action="action", input={}, output=long_output
        ))

        summary = self.storage.get_summary()

        # 전체 output이 아닌 truncate된 버전 포함
        self.assertIn("...", summary)
        self.assertIn("chars total", summary)  # "1000 chars total" 표시 (영어 기반)

        print("  ✓ 긴 output truncate 완료")


class TestSharedStorageFilePersistence(unittest.TestCase):
    """파일 저장/로드 테스트"""
    
    def setUp(self):
        self.storage = SharedStorage(debug_enabled=True)
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ).name
    
    def tearDown(self):
        self.storage.reset()
        if os.path.exists(self.temp_file):
            os.unlink(self.temp_file)
    
    def test_01_save_and_load(self):
        """저장 및 로드 테스트"""
        print("\n[TEST] 파일 저장 및 로드 테스트")
        
        # 데이터 생성
        self.storage.start_session("저장 테스트 쿼리", ["Step 1", "Step 2"])
        self.storage.add_result(create_tool_result(
            step=1,
            executor="test_tool", executor_type="tool",
            action="test_action", input={"key": "value"},
            output={"result": "success"}
        ))
        
        original_context = self.storage.get_context()
        original_results = self.storage.get_results()
        
        # 저장
        self.storage.save_to_file(self.temp_file)
        self.assertTrue(os.path.exists(self.temp_file))
        
        # 새 인스턴스로 로드
        new_storage = SharedStorage(debug_enabled=True)
        success = new_storage.load_from_file(self.temp_file)
        
        self.assertTrue(success)
        
        # 검증
        loaded_context = new_storage.get_context()
        loaded_results = new_storage.get_results()
        
        self.assertEqual(original_context["user_query"], loaded_context["user_query"])
        self.assertEqual(original_context["current_plan"], loaded_context["current_plan"])
        self.assertEqual(len(original_results), len(loaded_results))
        
        print("  ✓ 파일 저장 및 로드 완료")
    
    def test_02_save_with_history(self):
        """히스토리 포함 저장 테스트"""
        print("\n[TEST] 히스토리 포함 저장 테스트")
        
        # 완료된 세션 생성
        self.storage.start_session("과거 쿼리")
        self.storage.add_result(create_tool_result(
            step=1,
            executor="tool", executor_type="tool",
            action="action", input={}, output="result"
        ))
        self.storage.complete_session("과거 응답")
        
        # 현재 세션 생성
        self.storage.start_session("현재 쿼리")
        
        # 저장
        self.storage.save_to_file(self.temp_file)
        
        # 로드
        new_storage = SharedStorage(debug_enabled=True)
        new_storage.load_from_file(self.temp_file)
        
        # 히스토리 검증
        history = new_storage.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["user_query"], "과거 쿼리")
        
        print("  ✓ 히스토리 포함 저장/로드 완료")
    
    def test_03_load_nonexistent_file(self):
        """존재하지 않는 파일 로드 테스트"""
        print("\n[TEST] 존재하지 않는 파일 로드 테스트")
        
        success = self.storage.load_from_file("/not/exist/file.json")
        self.assertFalse(success)
        
        print("  ✓ 존재하지 않는 파일 처리 완료")


class TestSharedStorageEdgeCases(unittest.TestCase):
    """Edge Case 테스트"""
    
    def setUp(self):
        self.storage = SharedStorage(debug_enabled=True)
    
    def tearDown(self):
        self.storage.reset()
    
    def test_01_no_session_errors(self):
        """세션 없이 작업 시 에러 테스트"""
        print("\n[TEST] 세션 없이 작업 시 에러 테스트")
        
        # 세션 없이 결과 추가 시도
        with self.assertRaises(RuntimeError):
            self.storage.add_result(create_tool_result(
                step=1,
                executor="tool", executor_type="tool",
                action="action", input={}, output="result"
            ))
        
        # 세션 없이 계획 업데이트 시도
        with self.assertRaises(RuntimeError):
            self.storage.update_plan(["Step 1"])
        
        # 세션 없이 완료 시도
        with self.assertRaises(RuntimeError):
            self.storage.complete_session("response")
        
        print("  ✓ 에러 처리 완료")
    
    def test_02_reset(self):
        """전체 초기화 테스트"""
        print("\n[TEST] 전체 초기화 테스트")
        
        # 데이터 생성
        self.storage.start_session("쿼리")
        self.storage.add_result(create_tool_result(
            step=1,
            executor="tool", executor_type="tool",
            action="action", input={}, output="result"
        ))
        self.storage.complete_session("응답")
        self.storage.start_session("새 쿼리")
        
        # 초기화
        self.storage.reset()
        
        # 검증
        self.assertIsNone(self.storage.get_context())
        self.assertEqual(self.storage.get_results(), [])
        self.assertEqual(self.storage.get_history(), [])
        
        print("  ✓ 전체 초기화 완료")
    
    def test_03_clear_history(self):
        """히스토리만 초기화 테스트"""
        print("\n[TEST] 히스토리만 초기화 테스트")
        
        # 히스토리 생성
        self.storage.start_session("쿼리 1")
        self.storage.complete_session("응답 1")
        self.storage.start_session("쿼리 2")
        self.storage.complete_session("응답 2")
        
        self.assertEqual(len(self.storage.get_history()), 2)
        
        # 히스토리만 초기화
        self.storage.clear_history()
        
        self.assertEqual(len(self.storage.get_history()), 0)
        
        print("  ✓ 히스토리 초기화 완료")
    
    def test_04_complex_output_types(self):
        """다양한 output 타입 테스트"""
        print("\n[TEST] 다양한 output 타입 테스트")
        
        self.storage.start_session("테스트")
        
        # Dict output
        self.storage.add_result(create_tool_result(
            step=1,
            executor="tool", executor_type="tool",
            action="action", input={},
            output={"nested": {"data": [1, 2, 3]}}
        ))
        
        # List output
        self.storage.add_result(create_tool_result(
            step=2,
            executor="tool", executor_type="tool",
            action="action", input={},
            output=["item1", "item2", {"key": "value"}]
        ))
        
        # None output
        self.storage.add_result(create_tool_result(
            step=3,
            executor="tool", executor_type="tool",
            action="action", input={},
            output=None
        ))
        
        # String output
        self.storage.add_result(create_tool_result(
            step=4,
            executor="tool", executor_type="tool",
            action="action", input={},
            output="simple string"
        ))
        
        results = self.storage.get_results()
        self.assertEqual(len(results), 4)
        
        # Summary 생성 가능 확인
        summary = self.storage.get_summary()
        self.assertIsNotNone(summary)
        
        print("  ✓ 다양한 output 타입 처리 완료")


class TestDataFlowScenario(unittest.TestCase):
    """실제 시나리오 기반 데이터 흐름 테스트"""
    
    def setUp(self):
        self.storage = SharedStorage(debug_enabled=True)
    
    def tearDown(self):
        self.storage.reset()
    
    def test_full_scenario(self):
        """전체 시나리오 테스트: 파일 읽기 → 분석 → 웹 검색"""
        print("\n[TEST] 전체 시나리오 테스트")
        print("=" * 50)
        
        # 1. 세션 시작
        user_query = "wifi.log 파일을 읽고 에러를 분석해줘"
        plan = [
            "file_tool로 wifi.log 읽기",
            "llm_tool로 에러 분석"
        ]
        
        session_id = self.storage.start_session(user_query, plan)
        print(f"\n1. 세션 시작: {session_id}")
        
        # 2. Step 1: file_tool 실행
        file_content = """
[2025-01-15 10:30:01] ERROR: DMA timeout in rx_handler
[2025-01-15 10:30:02] WARN: Retry count exceeded
[2025-01-15 10:30:03] ERROR: Connection dropped
"""
        self.storage.add_result(create_tool_result(
            step=1,
            executor="file_tool",
            executor_type="tool",
            action="read",
            input={"path": "/var/log/wifi.log"},
            output=file_content
        ))
        self.storage.advance_step()
        print("\n2. Step 1 완료: file_tool.read")
        
        # 3. Step 2: llm_tool 실행 (이전 output 사용)
        previous_output = self.storage.get_last_output()
        self.assertIn("DMA timeout", previous_output)
        
        analysis_result = {
            "errors_found": 2,
            "main_issue": "DMA timeout",
            "related_issues": ["Connection dropped"],
            "suggested_keywords": ["DMA timeout wifi driver fix"]
        }
        self.storage.add_result(create_tool_result(
            step=2,
            executor="llm_tool",
            executor_type="tool",
            action="analyze",
            input={"content": previous_output},
            output=analysis_result
        ))
        self.storage.advance_step()
        print("3. Step 2 완료: llm_tool.analyze")
        
        # Summary 전에 세션 완료
        
        # 5. Summary 확인
        summary = self.storage.get_summary()
        print("\n5. 현재 Summary:")
        print("-" * 40)
        print(summary)
        print("-" * 40)
        
        # 6. 모든 결과에서 데이터 흐름 확인
        results = self.storage.get_results()

        # Step 1 → Step 2: file output이 llm input으로
        step1_output = self.storage.get_output_by_step(1)
        step2_input = results[1]["input"]["content"]
        self.assertEqual(step1_output, step2_input)
        print("\n6. 데이터 흐름 검증: Step 1 output → Step 2 input ✓")
        
        # 7. 세션 완료
        final_response = """
분석 결과:
- wifi.log에서 2개의 에러 발견
- 주요 이슈: DMA timeout
- 드라이버 재설치를 권장합니다
"""
        session = self.storage.complete_session(final_response)
        print(f"\n7. 세션 완료: {session.session_id}")
        
        # 8. 히스토리 확인
        history = self.storage.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["session_id"], session_id)
        print("8. 히스토리 저장 확인 ✓")
        
        # 9. 디버그 상태 출력
        print("\n9. 최종 상태:")
        self.storage.debug_print_state()
        
        print("=" * 50)
        print("[TEST] 전체 시나리오 완료 ✓")


def run_tests():
    """모든 테스트 실행"""
    print("\n" + "=" * 60)
    print("SharedStorage Module Test Suite")
    print("=" * 60)
    
    # 테스트 순서 정의
    test_classes = [
        TestSharedStorageBasic,
        TestSharedStorageResults,
        TestSharedStorageHistory,
        TestSharedStorageSummary,
        TestSharedStorageFilePersistence,
        TestSharedStorageEdgeCases,
        TestDataFlowScenario,
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 결과 요약
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
