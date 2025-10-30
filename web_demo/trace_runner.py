import json
import time
import os
import sys
from typing import Dict, Any, List, Generator
import importlib

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 添加全局变量来跟踪运行状态
running_status = {
    "is_running": False,
    "should_stop": False
}

# 导入必要的模块，处理可能的导入错误
try:
    from problem_processor import _parse_llm_output, _build_instrumentation_prompt, _build_analysis_planning_prompt, _build_code_implementation_prompt, _parse_implementation_output
    from reporting import format_check_correctness_result
    from src.generation import generator
    from src.traceRunner import execute_code_and_capture_prints_last
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    IMPORTS_AVAILABLE = False

class TraceCoderRunner:
    """实际运行TraceCoder流程的类"""
    
    def __init__(self):
        # 创建一个模拟的args对象，包含必要的参数
        class Args:
            def __init__(self):
                self.model = "gemini-1.5-flash"
                self.max_debug_attempts = 3
                self.timeout = 10
                self.no_instrumentation = False
                self.max_no_improvement_streak = 2
        
        self.args = Args()
    
    def run_problem_stream(self, problem_description: str, test_cases: str, model_name: str = "gemini-1.5-flash") -> Generator[Dict[str, Any], None, None]:
        """流式运行单个问题的TraceCoder流程"""
        # 设置运行状态
        running_status["is_running"] = True
        running_status["should_stop"] = False
        
        try:
            # 更新模型名称
            self.args.model = model_name
            
            # 构造问题数据
            problem_data = {
                "task_id": "custom_problem",
                "complete_prompt": problem_description,
                "test": test_cases
            }
            
            # 检查是否应该停止
            if running_status["should_stop"]:
                yield {
                    "step": 0,
                    "title": "已停止",
                    "description": "用户已停止TraceCoder流程",
                    "content": "用户已停止TraceCoder流程",
                    "status": "completed"
                }
                return
            
            # 步骤1: 问题描述输入
            yield {
                "step": 1,
                "title": "问题描述输入",
                "description": "接收并理解编程问题的描述",
                "content": problem_description,
                "status": "completed"
            }
            
            # 步骤2: 生成初始代码
            yield {
                "step": 2,
                "title": "生成初始代码",
                "description": "使用LLM生成解决问题的初始代码",
                "content": "正在生成初始代码...",
                "status": "in-progress"
            }
            
            # 检查是否应该停止
            if running_status["should_stop"]:
                yield {
                    "step": 0,
                    "title": "已停止",
                    "description": "用户已停止TraceCoder流程",
                    "content": "用户已停止TraceCoder流程",
                    "status": "completed"
                }
                return
            
            # 生成初始代码
            if IMPORTS_AVAILABLE:
                initial_code, p_tokens, c_tokens = generator(problem_description, "code3_generate", self.args.model)
            else:
                initial_code = "# 无法生成代码：缺少必要的依赖\n# 请安装transformers等依赖包"
            
            yield {
                "step": 2,
                "title": "生成初始代码",
                "description": "使用LLM生成解决问题的初始代码",
                "content": initial_code,
                "status": "completed"
            }
            
            # 步骤3: 执行代码
            yield {
                "step": 3,
                "title": "执行代码",
                "description": "在沙箱环境中执行生成的代码",
                "content": "正在执行代码...",
                "status": "in-progress"
            }
            
            # 检查是否应该停止
            if running_status["should_stop"]:
                yield {
                    "step": 0,
                    "title": "已停止",
                    "description": "用户已停止TraceCoder流程",
                    "content": "用户已停止TraceCoder流程",
                    "status": "completed"
                }
                return
            
            # 执行代码并获取结果
            if IMPORTS_AVAILABLE:
                exec_result = execute_code_and_capture_prints_last(
                    initial_code + "\n\n" + test_cases, timeout_seconds=self.args.timeout)
                
                # 使用简化版测试函数
                initial_eval = self._simple_check_correctness(problem_data, initial_code, self.args.timeout)
            else:
                exec_result = {"display_output": "无法执行代码：缺少必要的依赖"}
                initial_eval = {
                    "passed": False,
                    "passed_count": 0,
                    "total_count": 1,
                    "result": "无法执行代码：缺少必要的依赖"
                }
            
            yield {
                "step": 3,
                "title": "执行代码",
                "description": "在沙箱环境中执行生成的代码",
                "content": f"执行结果:\n{exec_result}\n\n测试结果:\n{initial_eval}",
                "status": "completed"
            }
            
            # 步骤4: 检查执行是否成功
            if initial_eval.get('passed', False):
                yield {
                    "step": 4,
                    "title": "检查执行结果",
                    "description": "检查代码执行是否成功通过所有测试",
                    "content": "功能测试通过，问题已解决！",
                    "status": "completed"
                }
                
                yield {
                    "step": 5,
                    "title": "解决方案完成",
                    "description": "TraceCoder流程完成",
                    "content": "问题已成功解决！代码功能正确，所有测试用例通过。",
                    "status": "completed"
                }
            else:
                yield {
                    "step": 4,
                    "title": "检查执行结果",
                    "description": "检查代码执行是否成功通过所有测试",
                    "content": f"功能测试失败，需要进入调试循环进行修复\n\n测试结果:\n{initial_eval}",
                    "status": "completed"
                }
                
                # 进入调试循环
                if IMPORTS_AVAILABLE:
                    yield from self._run_debugging_stream(initial_code, initial_eval, problem_data, self._simple_check_correctness)
                else:
                    yield {
                        "step": 5,
                        "title": "调试循环",
                        "description": "进入调试循环进行修复",
                        "content": "无法进入调试循环：缺少必要的依赖",
                        "status": "error"
                    }
        finally:
            # 重置运行状态
            running_status["is_running"] = False
            running_status["should_stop"] = False
    
    def _simple_check_correctness(self, problem_data, code, timeout):
        """
        简化版的测试函数，用于测试生成的代码
        """
        try:
            # 直接使用传入的测试用例
            # 支持两种情况：problem_data是测试用例字符串，或者是一个包含测试用例的字典
            if isinstance(problem_data, dict):
                test_cases = problem_data.get('test', '')
            else:
                test_cases = problem_data  # 假设problem_data就是测试用例字符串
            
            if not test_cases:
                return {
                    "passed": False,
                    "passed_count": 0,
                    "total_count": 1,
                    "result": "没有提供测试用例"
                }
            
            # 创建一个临时文件来执行代码和测试用例
            import tempfile
            import re
            import subprocess
            import sys
            
            # 检查测试用例是否已经是unittest格式
            if "unittest.TestCase" in test_cases or "class Test" in test_cases:
                # 如果已经是unittest格式，尝试修复常见的缩进问题
                try:
                    full_code = code + "\n\n" + test_cases
                except Exception as e:
                    # 如果修复失败，仍然使用原始测试用例
                    full_code = code + "\n\n" + test_cases
                is_unittest_format = True
            else:
                # 如果是简单的assert语句，需要包装成unittest格式
                # 提取函数名（假设函数名在代码中）
                func_name = "has_close_elements"  # 默认函数名
                func_match = re.search(r"def\s+(\w+)\s*\(", code)
                if func_match:
                    func_name = func_match.group(1)
                
                # 将简单的assert语句包装成unittest测试类
                test_lines = [line.strip() for line in test_cases.split('\n') if line.strip() and not line.strip().startswith('#') and line.strip().startswith('assert')]
                
                # 构造unittest测试类
                unittest_wrapper = f"""
import unittest

class Test{func_name.capitalize()}(unittest.TestCase):
"""
                
                for i, test_line in enumerate(test_lines, 1):
                    # 将assert语句转换为unittest格式
                    if "==" in test_line:
                        # 处理类似 "assert has_close_elements(...) == True" 的语句
                        unittest_wrapper += f"    def test_case_{i}(self):\n"
                        # 替换assert为self.assertEqual，并处理== True/False的情况
                        if " == True" in test_line:
                            cleaned_line = test_line.replace("assert ", "").replace(" == True", "")
                            unittest_wrapper += f"        self.assertTrue({cleaned_line})\n"
                        elif " == False" in test_line:
                            cleaned_line = test_line.replace("assert ", "").replace(" == False", "")
                            unittest_wrapper += f"        self.assertFalse({cleaned_line})\n"
                        else:
                            # 处理其他形式的==比较
                            parts = test_line.replace("assert ", "").split(" == ")
                            unittest_wrapper += f"        self.assertEqual({parts[0]}, {parts[1]})\n"
                    else:
                        # 处理其他形式的assert语句
                        unittest_wrapper += f"    def test_case_{i}(self):\n"
                        cleaned_line = test_line.replace("assert ", "")
                        unittest_wrapper += f"        self.assertTrue({cleaned_line})\n"
                
                unittest_wrapper += """
if __name__ == '__main__':
    unittest.main()
"""
                
                full_code = code + "\n\n" + unittest_wrapper
                is_unittest_format = False
            
            print("full_code:\n" + full_code)

            # 检查生成的代码是否有语法错误
            import ast
            try:
                ast.parse(full_code)
            except SyntaxError as e:
                return {
                    "passed": False,
                    "passed_count": 0,
                    "total_count": 1,
                    "result": f"测试用例格式错误: {str(e)}"
                }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name
            
            # 执行代码和测试用例
            if IMPORTS_AVAILABLE:
                exec_result = execute_code_and_capture_prints_last(
                    full_code, timeout_seconds=timeout)
                display_output = str(exec_result.get('display_output', ''))
            else:
                # 如果无法使用execute_code_and_capture_prints_last，尝试直接运行Python
                try:
                    result = subprocess.run([sys.executable, temp_file], 
                                          capture_output=True, text=True, timeout=timeout)
                    display_output = result.stdout + result.stderr
                except subprocess.TimeoutExpired:
                    display_output = "执行超时"
                except Exception as e:
                    display_output = f"执行错误: {str(e)}"
                exec_result = {"display_output": display_output}
            
            # 清理临时文件
            os.unlink(temp_file)
            
            # 分析执行结果来判断测试是否通过
            display_output = str(exec_result.get('display_output', ''))
            
            # 初始化变量
            has_error = False
            has_failure = False
            has_traceback = False
            has_timeout = False
            passed_count = 0
            total_count = 1
            failed_count = 0
            
            # 智能分析测试用例数量 - 更准确地匹配测试用例
            if is_unittest_format:
                # 对于unittest格式，从执行结果中提取测试数量
                # 查找类似 "Ran 4 tests" 的模式
                ran_match = re.search(r"Ran\s+(\d+)\s+tests", display_output)
                if ran_match:
                    total_count = int(ran_match.group(1))
                else:
                    total_count = 1
                
                # 查找失败的测试数量
                failures_match = re.search(r"failures=(\d+)", display_output)
                errors_match = re.search(r"errors=(\d+)", display_output)
                
                failed_count = 0
                if failures_match:
                    failed_count += int(failures_match.group(1))
                if errors_match:
                    failed_count += int(errors_match.group(1))
                
                # 检查是否有AssertionError
                if "AssertionError" in display_output:
                    failed_count = max(failed_count, 1)
                
                # 检查输出中是否包含"FAILED"或类似的失败信息
                if "FAILED" in display_output.upper():
                    failed_count = max(failed_count, 1)
                
                # 计算通过的测试数量
                passed_count = total_count - failed_count
            else:
                # 针对用户提供的格式进行优化匹配
                # 匹配形如 "assert has_close_elements(...)" 的语句
                assert_matches = re.findall(r"assert\s+has_close_elements\s*\(", test_cases)
                total_count = len(assert_matches)
                
                # 如果没有找到特定的assert语句，尝试更通用的匹配
                if total_count == 0:
                    # 匹配所有assert语句
                    assert_matches = re.findall(r"assert\s+", test_cases)
                    total_count = len(assert_matches)
                
                # 如果仍然没有找到assert语句，尝试按行分割并计算非空行数
                if total_count == 0:
                    # 按行分割并过滤掉空行和注释行
                    lines = [line.strip() for line in test_cases.split('\n') if line.strip() and not line.strip().startswith('#')]
                    # 计算包含assert的行数
                    total_count = sum(1 for line in lines if line.startswith('assert'))
                
                # 如果还是没有找到测试用例，设置默认值
                if total_count == 0:
                    total_count = 1
                
                # 检查是否有测试失败或错误的信息
                has_error = "error" in display_output.lower()
                has_failure = "failed" in display_output.lower()
                has_traceback = "Traceback" in display_output
                has_timeout = "timeout" in display_output.lower()
                
                # 计算失败的测试数量
                failed_count = 0
                
                # 查找失败的测试数量
                failures_match = re.search(r"failures=(\d+)", display_output)
                errors_match = re.search(r"errors=(\d+)", display_output)
                
                if failures_match:
                    failed_count += int(failures_match.group(1))
                if errors_match:
                    failed_count += int(errors_match.group(1))
                
                # 检查是否有AssertionError
                if "AssertionError" in display_output:
                    failed_count = max(failed_count, 1)
                
                # 检查输出中是否包含"FAILED"或类似的失败信息
                if "FAILED" in display_output.upper():
                    failed_count = max(failed_count, 1)
                
                # 计算通过的测试数量
                passed_count = total_count - failed_count
            
            # 判断是否所有测试都通过
            passed = (passed_count == total_count) and (total_count > 0) and not has_error and not has_failure and not has_traceback and not has_timeout
            
            # 特殊处理：如果显示"OK"且没有错误信息，则认为所有测试通过
            if "OK" in display_output and not has_error and not has_failure and not has_traceback:
                passed = True
                passed_count = total_count
            
            # 如果没有任何测试运行（例如"Ran 0 tests"），则认为测试失败
            if "Ran 0 tests" in display_output:
                passed = False
                passed_count = 0
                total_count = 1
            
            return {
                "passed": passed,
                "passed_count": passed_count,
                "total_count": total_count,
                "result": display_output
            }
        except Exception as e:
            return {
                "passed": False,
                "passed_count": 0,
                "total_count": 1,
                "result": f"执行测试时发生错误: {str(e)}"
            }
    
    
    def _run_debugging_stream(self, initial_code, initial_eval, problem_data, check_correctness_func) -> Generator[Dict[str, Any], None, None]:
        """流式执行调试循环"""
        # 检查是否应该停止
        if running_status["should_stop"]:
            yield {
                "step": 0,
                "title": "已停止",
                "description": "用户已停止TraceCoder流程",
                "content": "用户已停止TraceCoder流程",
                "status": "completed"
            }
            return
            
        code_to_debug, last_eval, best_code, best_eval = initial_code, initial_eval, initial_code, initial_eval
        log, history, p_tokens, c_tokens, streak = [], [], 0, 0, 0
        instrumentation_suggestions = ""  # Initialize suggestions
        task_id = problem_data["task_id"]
        
        # 添加HLLM和RM状态展示
        hllm_records = []  # 存储HLLM记录
        rm_state = {
            "best_code": best_code,
            "best_score": best_eval.get('passed_count', 0),
            "streak": streak
        }
        
        for attempt in range(1, self.args.max_debug_attempts + 1):
            print(
                f"--- Debug attempt: {attempt}/{self.args.max_debug_attempts} | Best: {best_eval.get('passed_count', 0)}/{best_eval.get('total_count', 1)} ---")
            
            base_step = 5 + (attempt - 1) * 13  # 基础步骤编号，增加到13步以容纳所有步骤
            
            # 检查是否应该停止
            if running_status["should_stop"]:
                yield {
                    "step": 0,
                    "title": "已停止",
                    "description": "用户已停止TraceCoder流程",
                    "content": "用户已停止TraceCoder流程",
                    "status": "completed"
                }
                return
            
            # 步骤1: 代码插桩
            yield {
                "step": base_step,
                "title": f"代码插桩（第{attempt}轮）",
                "description": "插桩智能体为代码添加诊断探针以收集运行时信息",
                "content": f"正在进行第{attempt}轮调试的代码插桩...",
                "status": "in-progress"
            }
            
            # 1. Instrumentation
            captured_prints = "Instrumentation skipped or not applicable."
            instrumented_code = code_to_debug  # Default to original if instrumentation fails
            if not self.args.no_instrumentation:
                failure_info = format_check_correctness_result(last_eval)
                instr_prompt = _build_instrumentation_prompt(code_to_debug, instrumentation_suggestions, failure_info)
                
                # 检查是否应该停止
                if running_status["should_stop"]:
                    yield {
                        "step": 0,
                        "title": "已停止",
                        "description": "用户已停止TraceCoder流程",
                        "content": "用户已停止TraceCoder流程",
                        "status": "completed"
                    }
                    return
                
                instrumented_code_gen, p, c = generator(instr_prompt, "code3_generate", self.args.model)
                p_tokens, c_tokens = p_tokens + p, c_tokens + c
                
                if instrumented_code_gen.strip():
                    instrumented_code = instrumented_code_gen
                    
                    # 检查是否应该停止
                    if running_status["should_stop"]:
                        yield {
                            "step": 0,
                            "title": "已停止",
                            "description": "用户已停止TraceCoder流程",
                            "content": "用户已停止TraceCoder流程",
                            "status": "completed"
                        }
                        return
                    
                    exec_result = execute_code_and_capture_prints_last(
                        instrumented_code + "\n\n" + problem_data.get('test', ''), timeout_seconds=self.args.timeout)
                    captured_prints = exec_result.get('display_output', 'No instrumentation output captured.')
            
            # 更新代码插桩步骤 - 完成状态
            yield {
                "step": base_step,
                "title": f"代码插桩（第{attempt}轮）",
                "description": "插桩智能体为代码添加诊断探针以收集运行时信息",
                "content": f"第{attempt}轮调试的代码插桩完成。\n\n插桩后的代码:\n{instrumented_code}",
                "status": "completed"
            }
            
            # 步骤2: 收集执行轨迹
            yield {
                "step": base_step + 1,
                "title": f"收集执行轨迹（第{attempt}轮）",
                "description": "收集代码执行过程中的详细轨迹信息",
                "content": f"第{attempt}轮调试的执行轨迹信息收集完成。\n\n捕获的输出:\n{captured_prints}",
                "status": "completed"
            }
            
            # 步骤3: 分析轨迹并生成修复计划
            yield {
                "step": base_step + 2,
                "title": f"分析轨迹并生成修复计划（第{attempt}轮）",
                "description": "分析智能体分析执行轨迹并生成详细的修复计划",
                "content": f"正在进行第{attempt}轮调试结果分析并生成修复计划...",
                "status": "in-progress"
            }
            
            # 2. Analysis and Planning (This is now the unified first step)
            analysis_prompt = _build_analysis_planning_prompt(problem_data, instrumented_code, captured_prints, history)
            
            # 检查是否应该停止
            if running_status["should_stop"]:
                yield {
                    "step": 0,
                    "title": "已停止",
                    "description": "用户已停止TraceCoder流程",
                    "content": "用户已停止TraceCoder流程",
                    "status": "completed"
                }
                return
            
            llm_response, p, c = generator(analysis_prompt, "code4_generate", self.args.model)
            p_tokens, c_tokens = p_tokens + p, c_tokens + c
            
            repair_plan, instrumentation_suggestions = _parse_llm_output(llm_response)
            
            # 更新分析轨迹并生成修复计划步骤 - 完成状态
            # 提取REPAIR_PLAN_START和REPAIR_PLAN_END之间的内容
            clean_repair_plan = ""
            if repair_plan:
                # 查找REPAIR_PLAN_START和REPAIR_PLAN_END之间的内容
                import re
                plan_match = re.search(r"REPAIR_PLAN_START\s*``(?:python)?\s*(.*?)\s*```\s*REPAIR_PLAN_END", repair_plan, re.DOTALL)
                if plan_match:
                    clean_repair_plan = plan_match.group(1).strip()
                else:
                    # 如果没有找到标记，使用原始repair_plan
                    clean_repair_plan = repair_plan.strip()
            
            plan_content = clean_repair_plan if clean_repair_plan else f"第{attempt}轮调试结果分析完成，生成修复计划。"
            yield {
                "step": base_step + 2,
                "title": f"分析轨迹并生成修复计划（第{attempt}轮）",
                "description": "分析智能体分析执行轨迹并生成详细的修复计划",
                "content": plan_content,
                "status": "completed"
            }
            
            # 步骤4: 提出新插桩方案
            # 提取Instrumentation_Suggestions部分的内容
            instrumentation_content = ""
            if instrumentation_suggestions:
                # 查找Instrumentation_Suggestions_START和Instrumentation_Suggestions_END之间的内容
                import re
                inst_match = re.search(r"Instrumentation_Suggestions_START\s*```(?:python)?\s*(.*?)\s*```\s*Instrumentation_Suggestions_END", instrumentation_suggestions, re.DOTALL)
                if inst_match:
                    instrumentation_content = inst_match.group(1).strip()
                else:
                    # 如果没有找到标记，使用原始instrumentation_suggestions
                    instrumentation_content = instrumentation_suggestions.strip()
            
            content_text = instrumentation_content if instrumentation_content else f"基于第{attempt}轮分析结果提出新的插桩方案完成。"
            yield {
                "step": base_step + 3,
                "title": f"提出新插桩方案（第{attempt}轮）",
                "description": "基于分析结果提出新的插桩建议",
                "content": content_text,
                "status": "completed"
            }
            
            # 步骤5: 修复代码
            yield {
                "step": base_step + 4,
                "title": f"修复代码（第{attempt}轮）",
                "description": "修复智能体根据修复计划修改代码",
                "content": f"正在进行第{attempt}轮代码修复...",
                "status": "in-progress"
            }
            
            # 3. Code Implementation (This is the unified second step, skipped if no plan)
            candidate_code = ""
            if repair_plan and repair_plan.strip():
                # The two-step repair is now the default if a plan is returned.
                # The `no_two_step_repair` flag is implicitly handled by whether a plan is generated.
                impl_prompt = _build_code_implementation_prompt(problem_data, code_to_debug, captured_prints, repair_plan)
                
                # 检查是否应该停止
                if running_status["should_stop"]:
                    yield {
                        "step": 0,
                        "title": "已停止",
                        "description": "用户已停止TraceCoder流程",
                        "content": "用户已停止TraceCoder流程",
                        "status": "completed"
                    }
                    return
            
                llm_response_impl, p_impl, c_impl = generator(impl_prompt, "code4_generate", self.args.model)
                p_tokens, c_tokens = p_tokens + p_impl, c_tokens + c_impl
                candidate_code = _parse_implementation_output(llm_response_impl)
            
            # 更新修复代码步骤 - 完成状态
            # 提取CODE部分的内容，去除多余的标记
            clean_candidate_code = candidate_code
            if candidate_code:
                # 查找CODE:和END_CODE之间的内容，去除标记和多余的格式
                import re
                # 匹配包含或不包含语言标识的代码块
                # 首先尝试匹配完整的CODE:到END_CODE格式
                code_match = re.search(r"CODE:\s*```(?:python)?\s*(.*?)\s*```\s*END_CODE", candidate_code, re.DOTALL)
                if code_match:
                    clean_candidate_code = code_match.group(1).strip()
                else:
                    # 如果没有找到CODE:标记，尝试匹配```python到END_CODE格式
                    code_match = re.search(r"```(?:python)?\s*(.*?)\s*END_CODE", candidate_code, re.DOTALL)
                    if code_match:
                        clean_candidate_code = code_match.group(1).strip()
                    else:
                        # 如果没有找到END_CODE标记，尝试匹配完整的```python到```格式
                        code_match = re.search(r"``(?:python)?\s*(.*?)\s*```", candidate_code, re.DOTALL)
                        if code_match:
                            clean_candidate_code = code_match.group(1).strip()
                        else:
                            # 如果没有找到标记，尝试去除可能的多余标记
                            clean_candidate_code = re.sub(r"```(?:python)?\s*", "", candidate_code).strip()
                            clean_candidate_code = re.sub(r"\s*```\s*END_CODE", "", clean_candidate_code).strip()
            
            content_text = clean_candidate_code if clean_candidate_code else "根据分析结果修复代码完成。"
            yield {
                "step": base_step + 4,
                "title": f"修复代码（第{attempt}轮）",
                "description": "修复智能体根据修复计划修改代码",
                "content": content_text,
                "status": "completed"
            }
            
            # 步骤6: 测试修复后的代码
            yield {
                "step": base_step + 5,
                "title": f"测试修复后的代码（第{attempt}轮）",
                "description": "执行修复后的代码并验证结果",
                "content": f"正在进行第{attempt}轮修复后代码测试...",
                "status": "in-progress"
            }
            
            # 3. Evaluation and decision-making
            if not candidate_code.strip():
                print("LLM did not provide valid code, terminating debugging.");
                break
            
            # 检查是否应该停止
            if running_status["should_stop"]:
                yield {
                    "step": 0,
                    "title": "已停止",
                    "description": "用户已停止TraceCoder流程",
                    "content": "用户已停止TraceCoder流程",
                    "status": "completed"
                }
                return
            
            eval_candidate = check_correctness_func(problem_data, candidate_code, self.args.timeout) # Use the passed-in function
            print(f"Candidate repair result: {format_check_correctness_result(eval_candidate)}")
            
            # 更新测试修复后的代码步骤 - 完成状态
            test_content = f"第{attempt}轮修复后代码测试结果：\n"
            test_content += f"通过测试用例：{eval_candidate.get('passed_count', 0)}/{eval_candidate.get('total_count', 1)}\n"
            
            # 清理测试结果，去除格式错误信息
            result_text = eval_candidate.get('result', 'No result')
            if "测试用例格式错误" in result_text:
                # 提取关键的错误信息
                lines = result_text.split('\n')
                error_lines = [line for line in lines if "测试用例格式错误" in line or "invalid syntax" in line]
                if error_lines:
                    result_text = '\n'.join(error_lines)
            
            test_content += f"测试结果：{result_text}"
            
            yield {
                "step": base_step + 5,
                "title": f"测试修复后的代码（第{attempt}轮）",
                "description": "执行修复后的代码并验证结果",
                "content": test_content,
                "status": "completed"
            }
            
            # 步骤7: 检查测试结果改善情况
            passed = eval_candidate.get('passed', False)
            yield {
                "step": base_step + 6,
                "title": f"检查测试结果改善情况（第{attempt}轮）",
                "description": "检查修复后的代码是否通过了更多测试",
                "content": f"第{attempt}轮修复{'成功' if passed else '失败'}\n\n测试结果:\n{eval_candidate}",
                "status": "completed"
            }
            
            attempt_log = {"attempt": attempt, "plan": repair_plan, "eval_result": eval_candidate}
            log.append(attempt_log)
            
            # 步骤8: HLLM机制更新
            # 将当前失败记录添加到HLLM记录中（无论是否通过）
            hllm_record = {
                "attempt": attempt,
                "passed_count": eval_candidate.get('passed_count', 0),
                "total_count": eval_candidate.get('total_count', 1),
                "repair_plan": repair_plan if repair_plan else "无修复计划",
                "code": candidate_code,
                "result": eval_candidate.get('result', 'No result')
            }
            hllm_records.append(hllm_record)
            
            # 格式化HLLM记录用于展示
            hllm_content = "HLLM（历史学习机制）更新：\n"
            if passed:
                hllm_content += f"第{attempt}轮修复成功！记录成功的修复经验。\n\n"
            else:
                hllm_content += f"已记录第{attempt}轮失败的修复尝试。\n\n"
            
            hllm_content += "历史记录：\n"
            for record in hllm_records:
                status = "成功" if record.get('passed_count', 0) == record.get('total_count', 1) and record.get('total_count', 1) > 0 else "失败"
                hllm_content += f"  第{record['attempt']}轮: {status} - 通过{record['passed_count']}/{record['total_count']}测试用例\n"
                if record['repair_plan'] and record['repair_plan'] != "无修复计划":
                    hllm_content += f"    修复计划: {record['repair_plan'][:100]}...\n"
                hllm_content += f"    结果: {record['result'][:100]}...\n\n"
            
            yield {
                "step": base_step + 7,
                "title": f"HLLM机制更新（第{attempt}轮）",
                "description": "历史学习机制记录当前修复尝试，供后续分析使用",
                "content": hllm_content,
                "status": "completed"
            }
            
            if passed:
                best_code, best_eval = candidate_code, eval_candidate
                # 步骤9: 保存修复经验到HLLM
                yield {
                    "step": base_step + 8,
                    "title": f"保存修复经验到HLLM（第{attempt}轮）",
                    "description": "将成功的修复经验保存到历史学习机制中",
                    "content": f"已将第{attempt}轮成功的修复经验保存到HLLM知识库中。\n\n修复成功的代码:\n{candidate_code}",
                    "status": "completed"
                }
                break
            
            # 步骤10: RM机制更新
            if eval_candidate.get('passed_count', 0) > best_eval.get('passed_count', 0):
                best_code, best_eval, code_to_debug, last_eval, streak = candidate_code, eval_candidate, candidate_code, eval_candidate, 0
                # 更新RM状态
                rm_state = {
                    "best_code": best_code,
                    "best_score": best_eval.get('passed_count', 0),
                    "streak": streak
                }
                
                # 展示RM机制更新（改善情况）
                rm_content = "RM（回滚机制）更新：\n"
                rm_content += f"发现更好的解决方案！\n"
                rm_content += f"更新最佳代码版本：通过{best_eval.get('passed_count', 0)}/{best_eval.get('total_count', 1)}测试用例\n"
                rm_content += f"重置无改善计数器：{streak}\n\n"
                rm_content += f"当前最佳代码:\n{best_code}"
                
                yield {
                    "step": base_step + 9,
                    "title": f"RM机制更新（第{attempt}轮）",
                    "description": "回滚机制检测到改善，更新最佳代码版本",
                    "content": rm_content,
                    "status": "completed"
                }
            else:
                streak += 1
                # 更新RM状态
                rm_state = {
                    "best_code": best_code,
                    "best_score": best_eval.get('passed_count', 0),
                    "streak": streak
                }
                
                # 展示RM机制更新（无改善情况）
                rm_content = "RM（回滚机制）更新：\n"
                rm_content += f"未发现改善，当前最佳代码仍通过{best_eval.get('passed_count', 0)}/{best_eval.get('total_count', 1)}测试用例\n"
                rm_content += f"无改善计数器：{streak}/{self.args.max_no_improvement_streak}\n\n"
                rm_content += f"当前最佳代码:\n{best_code}"
                
                yield {
                    "step": base_step + 9,
                    "title": f"RM机制更新（第{attempt}轮）",
                    "description": "回滚机制检测到无改善，更新计数器",
                    "content": rm_content,
                    "status": "completed"
                }

            history.append(attempt_log)
            
            # 步骤11: 检查是否触发RM回滚机制
            if streak >= self.args.max_no_improvement_streak:
                print(f"{streak} consecutive attempts with no improvement, terminating debugging.");
                # 触发RM回滚机制
                yield {
                    "step": base_step + 10,
                    "title": f"触发RM回滚机制（第{attempt}轮）",
                    "description": "修复失败，触发回滚机制并更新失败模式",
                    "content": f"第{attempt}轮修复失败，连续{streak}次无改善，触发RM回滚机制。\n\n当前代码:\n{code_to_debug}\n\n最佳代码:\n{best_code}",
                    "status": "completed"
                }
                break
            
            # 步骤12: 准备下一轮调试
            yield {
                "step": base_step + 11,
                "title": f"准备下一轮调试（第{attempt}轮）",
                "description": "准备进入下一轮调试循环",
                "content": f"第{attempt}轮调试完成，准备进入第{attempt+1}轮调试。\n\n当前状态:\n- 最佳代码通过: {best_eval.get('passed_count', 0)}/{best_eval.get('total_count', 1)}测试用例\n- 无改善计数器: {streak}/{self.args.max_no_improvement_streak}",
                "status": "completed"
            }
