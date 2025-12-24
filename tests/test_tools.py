"""
Tools Module Test Cases
=======================
BaseTool, FileTool, LLMTool 테스트
"""

import os
import sys
import json
import tempfile
import shutil
import unittest

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_tool import BaseTool, ActionSchema, ActionParam, ToolResult, ToolRegistry
from tools.file_tool import FileTool
from tools.llm_tool import LLMTool


class TestToolResult(unittest.TestCase):
    """ToolResult 테스트"""
    
    def test_success_result(self):
        """성공 결과 생성"""
        print("\n[TEST] ToolResult 성공 결과")
        
        result = ToolResult.success_result(
            output="test output",
            metadata={"key": "value"}
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.output, "test output")
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["key"], "value")
        
        print("  ✓ 성공 결과 생성 완료")
    
    def test_error_result(self):
        """에러 결과 생성"""
        print("\n[TEST] ToolResult 에러 결과")
        
        result = ToolResult.error_result("Something went wrong")
        
        self.assertFalse(result.success)
        self.assertIsNone(result.output)
        self.assertEqual(result.error, "Something went wrong")
        
        print("  ✓ 에러 결과 생성 완료")


class TestActionSchema(unittest.TestCase):
    """ActionSchema 테스트"""
    
    def test_to_dict(self):
        """스키마 dict 변환"""
        print("\n[TEST] ActionSchema to_dict")
        
        schema = ActionSchema(
            name="test_action",
            description="테스트 액션",
            params=[
                ActionParam("required_param", "str", True, "필수 파라미터"),
                ActionParam("optional_param", "int", False, "선택 파라미터", 10)
            ],
            output_type="dict",
            output_description="테스트 출력"
        )
        
        result = schema.to_dict()
        
        self.assertEqual(result["description"], "테스트 액션")
        self.assertIn("required_param", result["parameters"]["properties"])
        self.assertIn("optional_param", result["parameters"]["properties"])
        self.assertEqual(result["parameters"]["required"], ["required_param"])
        self.assertEqual(result["output"]["type"], "object")
        
        print("  ✓ 스키마 변환 완료")


class TestFileTool(unittest.TestCase):
    """FileTool 테스트"""
    
    def setUp(self):
        """테스트 디렉토리 생성"""
        self.test_dir = tempfile.mkdtemp()
        self.tool = FileTool(base_path=self.test_dir, debug_enabled=True)
    
    def tearDown(self):
        """테스트 디렉토리 삭제"""
        shutil.rmtree(self.test_dir)
    
    def test_01_get_schema(self):
        """스키마 조회"""
        print("\n[TEST] FileTool 스키마 조회")
        
        schema = self.tool.get_schema()
        
        self.assertEqual(schema["name"], "file_tool")
        self.assertIn("read", schema["actions"])
        self.assertIn("write", schema["actions"])
        self.assertIn("delete", schema["actions"])
        
        print("  ✓ 스키마 조회 완료")
        print(f"    Actions: {list(schema['actions'].keys())}")
    
    def test_02_write_and_read(self):
        """파일 쓰기/읽기"""
        print("\n[TEST] FileTool 쓰기/읽기")
        
        # 쓰기
        write_result = self.tool.execute("write", {
            "write_path": "test.txt",
            "write_content": "Hello, World!"
        })
        
        self.assertTrue(write_result.success)
        print(f"  ✓ 파일 쓰기 완료: {write_result.output}")
        
        # 읽기
        read_result = self.tool.execute("read", {"read_path": "test.txt"})
        
        self.assertTrue(read_result.success)
        self.assertEqual(read_result.output, "Hello, World!")
        print(f"  ✓ 파일 읽기 완료: {read_result.output}")
    
    def test_03_write_overwrite(self):
        """파일 덮어쓰기"""
        print("\n[TEST] FileTool 덮어쓰기")
        
        # 첫 번째 쓰기
        self.tool.execute("write", {
            "write_path": "overwrite.txt",
            "write_content": "Original"
        })
        
        # 덮어쓰기 시도 (실패해야 함)
        result = self.tool.execute("write", {
            "write_path": "overwrite.txt",
            "write_content": "New content"
        })
        self.assertFalse(result.success)
        self.assertIn("already exists", result.error)
        print("  ✓ 덮어쓰기 방지 확인")
        
        # overwrite=True로 덮어쓰기
        result = self.tool.execute("write", {
            "write_path": "overwrite.txt",
            "write_content": "New content",
            "write_overwrite": True
        })
        self.assertTrue(result.success)
        
        # 내용 확인
        read_result = self.tool.execute("read", {"read_path": "overwrite.txt"})
        self.assertEqual(read_result.output, "New content")
        print("  ✓ 덮어쓰기 완료")
    
    def test_04_append(self):
        """파일에 내용 추가"""
        print("\n[TEST] FileTool 내용 추가")
        
        # 파일 생성
        self.tool.execute("write", {
            "write_path": "append.txt",
            "write_content": "Line 1\n"
        })
        
        # 내용 추가
        result = self.tool.execute("append", {
            "append_path": "append.txt",
            "append_content": "Line 2\n"
        })
        self.assertTrue(result.success)
        
        # 확인
        read_result = self.tool.execute("read", {"read_path": "append.txt"})
        self.assertEqual(read_result.output, "Line 1\nLine 2\n")
        print("  ✓ 내용 추가 완료")
    
    def test_05_delete(self):
        """파일 삭제"""
        print("\n[TEST] FileTool 삭제")
        
        # 파일 생성
        self.tool.execute("write", {
            "write_path": "delete_me.txt",
            "write_content": "To be deleted"
        })
        
        # 삭제
        result = self.tool.execute("delete", {"delete_path": "delete_me.txt"})
        self.assertTrue(result.success)
        self.assertIn("deleted successfully", result.output)
        
        # 존재 확인
        exists_result = self.tool.execute("exists", {"exists_path": "delete_me.txt"})
        self.assertIn("does not exist", exists_result.output)
        print("  ✓ 파일 삭제 완료")
    
    def test_06_list_dir(self):
        """디렉토리 목록"""
        print("\n[TEST] FileTool 디렉토리 목록")
        
        # 파일들 생성
        self.tool.execute("write", {"write_path": "file1.txt", "write_content": "1"})
        self.tool.execute("write", {"write_path": "file2.txt", "write_content": "2"})
        self.tool.execute("write", {"write_path": "file3.log", "write_content": "3"})
        
        # 전체 목록
        result = self.tool.execute("list_dir", {"list_dir_path": "."})
        self.assertTrue(result.success)
        self.assertEqual(len(result.output), 3)
        print(f"  ✓ 전체 목록: {result.output}")
        
        # 패턴 필터
        result = self.tool.execute("list_dir", {
            "list_dir_path": ".",
            "list_dir_pattern": "*.txt"
        })
        self.assertEqual(len(result.output), 2)
        print(f"  ✓ *.txt 필터: {result.output}")
    
    def test_07_invalid_action(self):
        """잘못된 액션"""
        print("\n[TEST] FileTool 잘못된 액션")
        
        result = self.tool.execute("invalid_action", {})
        
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)
        print(f"  ✓ 에러 처리: {result.error}")
    
    def test_08_missing_required_param(self):
        """필수 파라미터 누락"""
        print("\n[TEST] FileTool 필수 파라미터 누락")
        
        result = self.tool.execute("read", {})
        
        self.assertFalse(result.success)
        self.assertIn("Required parameter", result.error)
        print(f"  ✓ 에러 처리: {result.error}")
    
    def test_09_file_not_found(self):
        """파일 없음"""
        print("\n[TEST] FileTool 파일 없음")
        
        result = self.tool.execute("read", {"read_path": "not_exist.txt"})
        
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)
        print(f"  ✓ 에러 처리: {result.error}")


class TestLLMTool(unittest.TestCase):
    """LLMTool 테스트 (Mock 모드)"""
    
    def setUp(self):
        self.tool = LLMTool(debug_enabled=True, use_mock=True)
    
    def test_01_get_schema(self):
        """스키마 조회"""
        print("\n[TEST] LLMTool 스키마 조회")
        
        schema = self.tool.get_schema()
        
        self.assertEqual(schema["name"], "llm_tool")
        self.assertIn("ask", schema["actions"])
        self.assertIn("analyze", schema["actions"])
        self.assertIn("summarize", schema["actions"])
        self.assertIn("extract", schema["actions"])
        
        print("  ✓ 스키마 조회 완료")
        print(f"    Actions: {list(schema['actions'].keys())}")
    
    def test_02_ask(self):
        """질문하기 (Mock)"""
        print("\n[TEST] LLMTool 질문하기 (Mock)")
        
        result = self.tool.execute("ask", {
            "ask_prompt": "Python의 장점은 무엇인가요?"
        })
        
        self.assertTrue(result.success)
        self.assertIsInstance(result.output, str)
        self.assertTrue(len(result.output) > 0)
        
        print(f"  ✓ 질문 응답 완료")
        print(f"    Response length: {len(result.output)}")
    
    def test_03_analyze(self):
        """분석하기 (Mock)"""
        print("\n[TEST] LLMTool 분석하기 (Mock)")
        
        result = self.tool.execute("analyze", {
            "analyze_content": "[ERROR] DMA timeout at 0x1234\n[WARN] Retry failed",
            "analyze_instruction": "에러 로그를 분석해주세요"
        })
        
        self.assertTrue(result.success)
        self.assertIsInstance(result.output, str)
        
        print(f"  ✓ 분석 완료")
        print(f"    Response preview: {result.output[:100]}...")
    
    def test_04_summarize(self):
        """요약하기 (Mock)"""
        print("\n[TEST] LLMTool 요약하기 (Mock)")
        
        long_content = "이것은 매우 긴 내용입니다. " * 50
        
        result = self.tool.execute("summarize", {
            "summarize_content": long_content,
            "summarize_max_length": 100
        })
        
        self.assertTrue(result.success)
        self.assertIsInstance(result.output, str)
        
        print(f"  ✓ 요약 완료")
        print(f"    Original: {len(long_content)} -> Summary: {len(result.output)}")
    
    def test_05_extract(self):
        """정보 추출하기 (Mock)"""
        print("\n[TEST] LLMTool 정보 추출하기 (Mock)")

        result = self.tool.execute("extract", {
            "extract_content": "[ERROR] Connection failed\n[ERROR] Timeout\n[WARN] Retry",
            "extract_what": "errors"
        })

        self.assertTrue(result.success)
        self.assertIsInstance(result.output, str)
        self.assertTrue(len(result.output) > 0)

        print(f"  ✓ 추출 완료")
        print(f"    Extracted: {result.output}")


class TestToolRegistry(unittest.TestCase):
    """ToolRegistry 테스트"""
    
    def setUp(self):
        self.registry = ToolRegistry(debug_enabled=True)
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_01_register_and_get(self):
        """Tool 등록 및 조회"""
        print("\n[TEST] ToolRegistry 등록 및 조회")

        file_tool = FileTool(base_path=self.test_dir, debug_enabled=True)
        llm_tool = LLMTool(debug_enabled=True, use_mock=True)

        self.registry.register(file_tool)
        self.registry.register(llm_tool)

        # 조회
        retrieved = self.registry.get("file_tool")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "file_tool")

        # 목록
        tools = self.registry.list_tools()
        self.assertEqual(len(tools), 2)
        self.assertIn("file_tool", tools)
        self.assertIn("llm_tool", tools)

        print(f"  ✓ 등록된 Tools: {tools}")
    
    def test_02_get_all_schemas(self):
        """모든 스키마 조회"""
        print("\n[TEST] ToolRegistry 모든 스키마 조회")

        self.registry.register(FileTool(base_path=self.test_dir))
        self.registry.register(LLMTool(use_mock=True))

        schemas = self.registry.get_all_schemas()

        self.assertEqual(len(schemas), 2)

        tool_names = [s["name"] for s in schemas]
        self.assertIn("file_tool", tool_names)
        self.assertIn("llm_tool", tool_names)

        print(f"  ✓ 스키마 조회 완료: {tool_names}")
    
    def test_03_execute_via_registry(self):
        """Registry를 통한 실행"""
        print("\n[TEST] ToolRegistry를 통한 실행")
        
        file_tool = FileTool(base_path=self.test_dir, debug_enabled=True)
        self.registry.register(file_tool)
        
        # 파일 쓰기
        result = self.registry.execute("file_tool", "write", {
            "write_path": "registry_test.txt",
            "write_content": "Created via registry"
        })
        self.assertTrue(result.success)
        self.assertIn("created successfully", result.output)
        
        # 파일 읽기
        result = self.registry.execute("file_tool", "read", {
            "read_path": "registry_test.txt"
        })
        self.assertTrue(result.success)
        self.assertEqual(result.output, "Created via registry")
        
        print("  ✓ Registry를 통한 실행 완료")
    
    def test_04_execute_unknown_tool(self):
        """없는 Tool 실행"""
        print("\n[TEST] ToolRegistry 없는 Tool 실행")
        
        result = self.registry.execute("unknown_tool", "action", {})
        
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)
        
        print(f"  ✓ 에러 처리: {result.error}")


class TestToolIntegration(unittest.TestCase):
    """Tool 통합 테스트 - SharedStorage와 함께"""
    
    def setUp(self):
        from core.shared_storage import SharedStorage

        self.test_dir = tempfile.mkdtemp()
        self.storage = SharedStorage(debug_enabled=True)
        self.file_tool = FileTool(base_path=self.test_dir, debug_enabled=True)
        self.llm_tool = LLMTool(debug_enabled=True, use_mock=True)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_full_workflow(self):
        """전체 워크플로우: 파일 읽기 -> 분석"""
        print("\n[TEST] 전체 워크플로우 테스트")
        print("=" * 50)

        # 1. 세션 시작
        self.storage.start_session(
            user_query="error.log 파일을 분석해줘",
            plan=[
                "file_tool로 error.log 읽기",
                "llm_tool로 에러 분석"
            ]
        )
        print("1. 세션 시작 완료")

        # 2. 테스트 파일 생성
        error_log = """[2025-01-15 10:30:01] ERROR: DMA timeout in rx_handler
[2025-01-15 10:30:02] WARN: Retry count exceeded (max: 3)
[2025-01-15 10:30:03] ERROR: Connection dropped - ECONNRESET
[2025-01-15 10:30:04] INFO: Attempting reconnection...
[2025-01-15 10:30:05] ERROR: Reconnection failed
"""
        self.file_tool.execute("write", {
            "write_path": "error.log",
            "write_content": error_log
        })

        # 3. Step 1: 파일 읽기
        result = self.file_tool.execute("read", {"read_path": "error.log"})
        self.assertTrue(result.success)
        result.step=1

        self.storage.add_result(result)
        self.storage.advance_step()
        print("2. Step 1 완료: file_tool.read")

        # 4. Step 2: LLM 분석
        previous_output = self.storage.get_last_output()
        result = self.llm_tool.execute("general", {
            "general_prompt": f"이 에러 로그에서 주요 문제를 식별하고 원인을 분석해주세요:\n{previous_output}"
        })
        self.assertTrue(result.success)
        result.step=2

        self.storage.add_result(result)
        self.storage.advance_step()
        print("3. Step 2 완료: llm_tool.general")

        # 5. Summary 확인
        summary = self.storage.get_summary()
        print("\n4. 현재 Summary:")
        print("-" * 40)
        print(summary)
        print("-" * 40)

        # 6. 검증
        results = self.storage.get_results()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["executor"], "file_tool")
        self.assertEqual(results[1]["executor"], "llm_tool")

        # 7. 세션 완료
        self.storage.complete_session(
            final_response="분석 완료: DMA timeout 에러가 주요 원인입니다.",
            status="completed"
        )
        print("\n5. 세션 완료")

        # 8. 히스토리 확인
        history = self.storage.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "completed")

        print("=" * 50)
        print("[TEST] 전체 워크플로우 완료 ✓")


def run_tests():
    """모든 테스트 실행"""
    print("\n" + "=" * 60)
    print("Tools Module Test Suite")
    print("=" * 60)
    
    test_classes = [
        TestToolResult,
        TestActionSchema,
        TestFileTool,
        TestLLMTool,
        TestToolRegistry,
        TestToolIntegration,
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
