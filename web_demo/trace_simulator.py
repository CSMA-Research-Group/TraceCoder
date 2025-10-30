import json
import time
import os
import sys
from typing import Dict, Any, List

class TraceCoderSimulator:
    """模拟TraceCoder的14步流程"""
    
    def __init__(self):
        # 加载示例配置
        config_path = os.path.join(os.path.dirname(__file__), 'examples_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.examples = json.load(f)['examples']
    
    def get_example_by_id(self, example_id: str) -> Dict[str, Any]:
        """根据ID获取示例"""
        for example in self.examples:
            if example['id'] == example_id:
                return example
        return None
    
    def simulate_tracecoder_process(self, example_id: str, model_name: str = "gemini-1.5-flash") -> List[Dict[str, Any]]:
        """模拟完整的TraceCoder流程"""
        example = self.get_example_by_id(example_id)
        if not example:
            raise ValueError(f"Example with id {example_id} not found")
        
        # 特别处理example4，展示多轮修复过程
        if example_id == "example4":
            return self._simulate_complex_debugging_process(example, model_name)
        
        steps = []
        
        # 步骤1: 问题描述输入
        steps.append({
            "step": 1,
            "title": "问题描述输入",
            "description": "接收并理解编程问题的描述",
            "content": example['description'],
            "status": "completed"
        })
        
        # 步骤2: 生成初始代码
        # 模拟LLM生成代码（对于example3，生成一个有缺陷的版本）
        if example_id == "example1":
            initial_code = """def two_sum(nums, target):
    # 暴力解法
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []"""
        elif example_id == "example2":
            # 生成一个有轻微错误的回文数判断实现
            initial_code = """def is_palindrome(x):
    # 负数不是回文数
    if x < 0:
        return False
    # 将数字转换为字符串并检查是否为回文
    s = str(x)
    # 错误：没有处理单个数字的情况
    if len(s) == 1:
        return False  # 错误：单个数字应该是回文数
    return s == s[::-1]"""
        else:  # example3
            # 生成一个有缺陷的斐波那契实现
            initial_code = """def fibonacci(n):
    # 错误的递归实现，没有正确的基础情况
    if n == 0:
        return 1  # 错误：应该是0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)"""
        
        steps.append({
            "step": 2,
            "title": "生成初始代码",
            "description": "使用LLM生成解决问题的初始代码",
            "content": initial_code,
            "prompt_tokens": 150,
            "completion_tokens": 200,
            "status": "completed"
        })
        
        # 步骤3: 执行代码
        # 模拟执行结果
        if example_id == "example1":
            exec_result_content = """测试执行结果：
assert two_sum([2, 7, 11, 15], 9) == [0, 1]  # 通过
assert two_sum([3, 2, 4], 6) == [1, 2]      # 通过
assert two_sum([3, 3], 6) == [0, 1]         # 通过

功能测试通过！但性能分析显示存在优化空间。"""
            execution_successful = True
        elif example_id == "example2":
            exec_result_content = """测试执行结果：
assert is_palindrome(121) == True   # 通过
assert is_palindrome(-121) == False # 通过
assert is_palindrome(10) == False   # 通过
assert is_palindrome(5) == True     # 失败：期望True，实际得到False
assert is_palindrome(0) == True     # 失败：期望True，实际得到False

功能测试失败：2/5测试用例未通过！
错误信息：
- 单个数字未被正确识别为回文数
- is_palindrome(5)返回False而不是True
- is_palindrome(0)返回False而不是True"""
            execution_successful = False
        else:  # example3
            exec_result_content = """测试执行结果：
assert fibonacci(0) == 0  # 失败：期望0，实际得到1
assert fibonacci(1) == 1  # 通过
assert fibonacci(2) == 1  # 失败：期望1，实际得到2
assert fibonacci(3) == 2  # 失败：期望2，实际得到3
assert fibonacci(4) == 3  # 失败：期望3，实际得到5
assert fibonacci(5) == 5  # 失败：期望5，实际得到8

功能测试失败：4/6测试用例未通过！
错误信息：
- fibonacci(0)返回1而不是0
- 多个结果不正确，表明基础情况处理有误"""
            execution_successful = False
        
        steps.append({
            "step": 3,
            "title": "执行代码",
            "description": "在沙箱环境中执行生成的代码",
            "content": exec_result_content,
            "status": "completed"
        })
        
        # 步骤4: 检查执行是否成功
        if execution_successful:
            steps.append({
                "step": 4,
                "title": "检查执行结果",
                "description": "检查代码执行是否成功通过所有测试",
                "content": "功能测试通过，但检测到性能问题需要优化",
                "status": "completed"
            })
        else:
            steps.append({
                "step": 4,
                "title": "检查执行结果",
                "description": "检查代码执行是否成功通过所有测试",
                "content": "功能测试失败，需要进入调试循环进行修复",
                "status": "completed"
            })
        
        # 根据执行结果决定流程
        if execution_successful:
            # 对于示例1，如果功能测试通过就直接结束，不进入性能优化流程
            if example_id == "example1":
                steps.append({
                    "step": 5,
                    "title": "解决方案完成",
                    "description": "TraceCoder流程完成",
                    "content": "问题已成功解决！代码功能正确，所有测试用例通过。",
                    "status": "completed"
                })
                return steps
            else:
                # 对于示例2和其他示例，继续进入调试循环以优化性能
                steps.append({
                    "step": 5,
                    "title": "代码插桩",
                    "description": "为代码添加诊断探针以收集运行时信息",
                    "content": "正在为代码添加性能分析探针...",
                    "status": "completed"
                })
            
            steps.append({
                "step": 6,
                "title": "运行插桩代码",
                "description": "执行插桩后的代码以收集运行时轨迹",
                "content": "正在运行插桩后的代码进行性能分析...",
                "status": "completed"
            })
            
            steps.append({
                "step": 7,
                "title": "收集执行轨迹",
                "description": "收集代码执行过程中的详细轨迹信息",
                "content": """收集到的性能分析数据：
1. 时间复杂度：O(n^2)
2. 空间复杂度：O(1)
3. 对于大数组输入，执行时间过长
4. 存在重复计算问题""",
                "status": "completed"
            })
            
            # 步骤8: 分析轨迹并生成修复计划
            analysis_result = """性能分析结果：
1. 问题定位：算法时间复杂度较高O(n^2)
2. 根本原因：使用了嵌套循环遍历数组
3. 修复建议：使用哈希表优化到O(n)时间复杂度
4. 预期改进：执行时间从平方级降低到线性级"""
            
            steps.append({
                "step": 8,
                "title": "分析轨迹并生成修复计划",
                "description": "分析执行轨迹并生成详细的修复计划",
                "content": analysis_result,
                "prompt_tokens": 300,
                "completion_tokens": 250,
                "status": "completed"
            })
            
            steps.append({
                "step": 9,
                "title": "提出新插桩方案",
                "description": "基于分析结果提出新的插桩建议",
                "content": "建议在哈希表操作关键位置添加性能监控探针",
                "status": "completed"
            })
            
            # 步骤10: 修复代码
            if example_id == "example1":
                repaired_code = """def two_sum(nums, target):
    # 使用哈希表优化时间复杂度到O(n)
    hash_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hash_map:
            return [hash_map[complement], i]
        hash_map[num] = i
    return []"""
            elif example_id == "example2":
                repaired_code = """def is_palindrome(x):
    # 负数不是回文数
    if x < 0:
        return False
    # 将数字转换为字符串并检查是否为回文
    s = str(x)
    # 修复：单个数字应该是回文数
    if len(s) == 1:
        return True  # 修复：单个数字是回文数
    return s == s[::-1]"""
            else:  # example3
                repaired_code = """def fibonacci(n):
    # 修复后的正确实现
    if n < 0:
        raise ValueError("输入必须是非负整数")
    elif n == 0:
        return 0  # 修复：正确返回0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)"""
            
            steps.append({
                "step": 10,
                "title": "修复代码",
                "description": "根据修复计划修改代码",
                "content": repaired_code,
                "prompt_tokens": 200,
                "completion_tokens": 220,
                "status": "completed"
            })
            
            # 步骤11: 测试修复后的代码
            if example_id == "example1":
                repaired_exec_result = """优化后代码性能测试结果：
1. 时间复杂度：O(n) ✓
2. 空间复杂度：O(n) ✓
3. 大数组输入处理时间显著降低 ✓
4. 所有功能测试通过 ✓

性能提升效果：
- 对于1000个元素的数组，执行时间从120ms降低到2ms
- 算法效率提升约60倍"""
            elif example_id == "example2":
                repaired_exec_result = """修复后代码测试结果：
assert is_palindrome(121) == True   # 通过
assert is_palindrome(-121) == False # 通过
assert is_palindrome(10) == False   # 通过
assert is_palindrome(5) == True     # 通过
assert is_palindrome(0) == True     # 通过

所有测试用例通过！
修复验证：
- 单个数字被正确识别为回文数
- is_palindrome(5)正确返回True
- is_palindrome(0)正确返回True"""
            else:  # example3
                repaired_exec_result = """修复后代码测试结果：
assert fibonacci(0) == 0  # 通过
assert fibonacci(1) == 1  # 通过
assert fibonacci(2) == 1  # 通过
assert fibonacci(3) == 2  # 通过
assert fibonacci(4) == 3  # 通过
assert fibonacci(5) == 5  # 通过

所有测试用例通过！
修复验证：
- fibonacci(0)正确返回0
- 递归计算结果正确
- 边界条件处理完善"""
            
            steps.append({
                "step": 11,
                "title": "测试修复后的代码",
                "description": "执行修复后的代码并验证结果",
                "content": repaired_exec_result,
                "status": "completed"
            })
            
            # 步骤12: 检查测试覆盖率是否改善
            repair_successful = True
            steps.append({
                "step": 12,
                "title": "检查测试结果改善情况",
                "description": "检查修复后的代码是否通过了更多测试",
                "content": "性能优化成功，功能测试全部通过",
                "status": "completed"
            })
            
            # 步骤13a或13b
            if repair_successful:
                if example_id == "example1":
                    hllm_content = "已将本次性能优化经验保存到HLLM（历史学习机制）中，供未来类似问题参考：\n- 暴力解法→哈希表优化模式\n- 时间复杂度从O(n^2)降至O(n)的通用方法"
                elif example_id == "example2":
                    hllm_content = "已将本次性能优化经验保存到HLLM（历史学习机制）中，供未来类似问题参考：\n- 字符串转换→数学计算优化模式\n- 时间复杂度从O(n)降至O(log n)的通用方法"
                else:  # example3
                    hllm_content = """HLLM（历史学习机制）记录：
问题模式：递归函数基础情况错误
错误特征：
- 基础情况返回值错误
- 导致递归链中所有结果错误
- 测试用例中多个失败案例具有相同模式

修复方案：
- 修正基础情况返回值
- 添加输入验证
- 增加边界条件检查

经验总结：
- 对于递归函数，必须仔细验证所有基础情况
- 基础情况的错误会以指数方式放大
- 插桩追踪是定位递归错误的有效方法

已将此经验保存到HLLM知识库，供未来类似问题参考。"""
                
                steps.append({
                    "step": 13,
                    "title": "保存修复经验到HLLM",
                    "description": "将成功的修复经验保存到历史学习机制中",
                    "content": hllm_content,
                    "status": "completed"
                })
            else:
                steps.append({
                    "step": 13,
                    "title": "触发RM回滚机制",
                    "description": "修复失败，触发回滚机制并更新失败模式",
                    "content": "修复未成功，触发RM（回滚机制），回滚到上一个最佳状态",
                    "status": "completed"
                })
            
            # 步骤14: 解决方案完成
            if example_id == "example1":
                final_status = "问题已成功解决！代码不仅功能正确，而且性能得到显著优化。"
            elif example_id == "example2":
                final_status = "问题已成功解决！代码功能正确，错误已修复。"
            else:  # example3
                final_status = "问题已成功解决！代码功能正确，错误已修复。"
                
            steps.append({
                "step": 14,
                "title": "解决方案完成",
                "description": "TraceCoder流程完成",
                "content": final_status,
                "status": "completed"
            })
        else:
            # 进入调试循环修复错误
            steps.append({
                "step": 5,
                "title": "代码插桩",
                "description": "插桩智能体为代码添加诊断探针以收集运行时信息",
                "content": """插桩智能体工作过程：
1. 分析代码结构，识别关键执行路径
2. 在函数入口和关键分支添加追踪代码
3. 添加变量状态监控点
4. 生成插桩后的代码用于调试

插桩后的代码：
def fibonacci(n):
    print(f"调用fibonacci({n})")
    # 错误的递归实现，没有正确的基础情况
    if n == 0:
        result = 1  # 错误：应该是0
        print(f"基础情况n=0，返回{result}")
        return result
    elif n == 1:
        result = 1
        print(f"基础情况n=1，返回{result}")
        return result
    else:
        result = fibonacci(n-1) + fibonacci(n-2)
        print(f"递归计算fibonacci({n}) = fibonacci({n-1}) + fibonacci({n-2}) = {result}")
        return result""",
                "status": "completed"
            })
            
            steps.append({
                "step": 6,
                "title": "运行插桩代码",
                "description": "执行插桩后的代码以收集运行时轨迹",
                "content": "正在运行插桩后的代码进行调试分析...",
                "status": "completed"
            })
            
            steps.append({
                "step": 7,
                "title": "收集执行轨迹",
                "description": "收集代码执行过程中的详细轨迹信息",
                "content": """收集到的执行轨迹信息：
调用堆栈跟踪：
1. 调用fibonacci(2)
2.   调用fibonacci(1) -> 返回1
3.   调用fibonacci(0) -> 返回1 (错误！应该返回0)
4. fibonacci(2)返回2 (错误！应该返回1)

调用堆栈跟踪：
1. 调用fibonacci(3)
2.   调用fibonacci(2) -> 返回2 (错误！应该返回1)
3.   调用fibonacci(1) -> 返回1
4. fibonacci(3)返回3 (错误！应该返回2)

问题定位：
- fibonacci(0)返回1而不是0
- 这导致所有后续计算结果都错误""",
                "status": "completed"
            })
            
            # 步骤8: 分析轨迹并生成修复计划
            analysis_result = """分析智能体诊断结果：
1. 问题定位：基础情况处理错误
2. 根本原因：fibonacci(0)应该返回0而不是1
3. 影响范围：由于递归特性，这个错误影响了所有n>0的计算结果
4. 修复建议：
   a. 修正fibonacci(0)的基础情况返回值
   b. 验证其他基础情况是否正确
   c. 添加边界条件检查防止无限递归"""
            
            steps.append({
                "step": 8,
                "title": "分析轨迹并生成修复计划",
                "description": "分析智能体分析执行轨迹并生成详细的修复计划",
                "content": analysis_result,
                "prompt_tokens": 300,
                "completion_tokens": 250,
                "status": "completed"
            })
            
            steps.append({
                "step": 9,
                "title": "提出新插桩方案",
                "description": "基于分析结果提出新的插桩建议",
                "content": """根据分析结果，提出新的插桩方案：
1. 在边界条件检查处添加输入验证探针
2. 在基础情况返回处添加详细日志
3. 在递归调用前后添加深度监控
4. 添加性能监控以检测潜在的无限递归""",
                "status": "completed"
            })
            
            # 步骤10: 修复代码
            repaired_code = """def fibonacci(n):
    # 修复后的正确实现
    if n < 0:
        raise ValueError("输入必须是非负整数")
    elif n == 0:
        return 0  # 修复：正确返回0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)"""
            
            steps.append({
                "step": 10,
                "title": "修复代码",
                "description": "修复智能体根据修复计划修改代码",
                "content": repaired_code,
                "prompt_tokens": 200,
                "completion_tokens": 220,
                "status": "completed"
            })
            
            # 步骤11: 测试修复后的代码
            repaired_exec_result = """修复后代码测试结果：
assert fibonacci(0) == 0  # 通过
assert fibonacci(1) == 1  # 通过
assert fibonacci(2) == 1  # 通过
assert fibonacci(3) == 2  # 通过
assert fibonacci(4) == 3  # 通过
assert fibonacci(5) == 5  # 通过

所有测试用例通过！
修复验证：
- fibonacci(0)正确返回0
- 递归计算结果正确
- 边界条件处理完善"""
            
            steps.append({
                "step": 11,
                "title": "测试修复后的代码",
                "description": "执行修复后的代码并验证结果",
                "content": repaired_exec_result,
                "status": "completed"
            })
            
            # 步骤12: 检查测试覆盖率是否改善
            repair_successful = True
            steps.append({
                "step": 12,
                "title": "检查测试结果改善情况",
                "description": "检查修复后的代码是否通过了更多测试",
                "content": "功能修复成功，所有测试用例通过",
                "status": "completed"
            })
            
            # 步骤13a: 保存修复经验到HLLM
            steps.append({
                "step": 13,
                "title": "保存修复经验到HLLM",
                "description": "将成功的修复经验保存到历史学习机制中",
                "content": """HLLM（历史学习机制）记录：
问题模式：递归函数基础情况错误
错误特征：
- 基础情况返回值错误
- 导致递归链中所有结果错误
- 测试用例中多个失败案例具有相同模式

修复方案：
- 修正基础情况返回值
- 添加输入验证
- 增加边界条件检查

经验总结：
- 对于递归函数，必须仔细验证所有基础情况
- 基础情况的错误会以指数方式放大
- 插桩追踪是定位递归错误的有效方法

已将此经验保存到HLLM知识库，供未来类似问题参考。""",
                "status": "completed"
            })
            
            # 步骤14: 解决方案完成
            final_status = "问题已成功解决！代码功能正确，错误已修复。"
            steps.append({
                "step": 14,
                "title": "解决方案完成",
                "description": "TraceCoder流程完成",
                "content": final_status,
                "status": "completed"
            })
        
        return steps
    
    def _simulate_complex_debugging_process(self, example: Dict[str, Any], model_name: str) -> List[Dict[str, Any]]:
        """模拟复杂的多轮调试过程，展示HLLM和RM机制"""
        steps = []
        debug_round = 1
        best_passed_count = 0
        consecutive_no_improvement = 0
        history_lessons = []
        
        # 步骤1: 问题描述输入
        steps.append({
            "step": 1,
            "title": "问题描述输入",
            "description": "接收并理解编程问题的描述",
            "content": example['description'],
            "status": "completed"
        })
        
        # 步骤2: 生成初始代码（有多个错误的实现）
        initial_code = """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less = [x for x in arr[1:] if x <= pivot]
        greater = [x for x in arr[1:] if x > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greater)"""
        
        steps.append({
            "step": 2,
            "title": "生成初始代码",
            "description": "使用LLM生成解决问题的初始代码",
            "content": initial_code,
            "prompt_tokens": 180,
            "completion_tokens": 220,
            "status": "completed"
        })
        
        # 步骤3: 执行代码
        exec_result_content = """测试执行结果：
assert quick_sort([]) == []                    # 通过
assert quick_sort([1]) == [1]                  # 通过
assert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]  # 通过
assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]  # 通过
assert quick_sort([3, 6, 8, 10, 1, 2, 1]) == [1, 1, 2, 3, 6, 8, 10]  # 通过
assert quick_sort([4, 2, 4, 2, 4, 2]) == [2, 2, 2, 4, 4, 4]  # 通过
assert quick_sort([-1, -3, 2, 0]) == [-3, -1, 0, 2]  # 失败

功能测试：6/7个测试用例通过！
错误信息：
- 包含负数的数组排序结果错误
- 表明分区逻辑存在问题"""
        
        steps.append({
            "step": 3,
            "title": "执行代码",
            "description": "在沙箱环境中执行生成的代码",
            "content": exec_result_content,
            "status": "completed"
        })
        
        # 步骤4: 检查执行是否成功
        steps.append({
            "step": 4,
            "title": "检查执行结果",
            "description": "检查代码执行是否成功通过所有测试",
            "content": "功能测试部分失败，需要进入调试循环进行修复",
            "status": "completed"
        })
        
        # 多轮调试循环
        current_code = initial_code
        current_passed_count = 6  # 初始通过6个测试用例
        best_code = current_code
        best_passed_count = current_passed_count  # 初始最佳通过数为6
        step_counter = 5  # 从步骤5开始
        
        # 模拟最多3轮调试
        for debug_round in range(1, 4):
            # 步骤5: 代码插桩
            steps.append({
                "step": step_counter,
                "title": f"代码插桩（第{debug_round}轮）",
                "description": "插桩智能体为代码添加诊断探针以收集运行时信息",
                "content": f"""插桩智能体工作过程（第{debug_round}轮）：
1. 分析代码结构，识别关键执行路径
2. 在分区和递归调用处添加追踪代码
3. 添加变量状态监控点
4. 生成插桩后的代码用于调试

插桩后的代码：
def quick_sort(arr):
    print(f"输入数组: {{arr}}")
    if len(arr) <= 1:
        print(f"基础情况，返回: {{arr}}")
        return arr
    else:
        pivot = arr[0]
        print(f"选择基准值: {{pivot}}")
        less = [x for x in arr[1:] if x <= pivot]
        greater = [x for x in arr[1:] if x > pivot]
        print(f"分区结果 - 小于等于基准: {{less}}, 大于基准: {{greater}}")
        result = quick_sort(less) + [pivot] + quick_sort(greater)
        print(f"合并结果: {{result}}")
        return result""",
                "status": "completed"
            })
            step_counter += 1
            
            # 步骤6: 运行插桩代码
            steps.append({
                "step": step_counter,
                "title": f"运行插桩代码（第{debug_round}轮）",
                "description": "执行插桩后的代码以收集运行时轨迹",
                "content": f"正在运行插桩后的代码进行第{debug_round}轮调试分析...",
                "status": "completed"
            })
            step_counter += 1
            
            # 步骤7: 收集执行轨迹
            if debug_round == 1:
                trace_content = """收集到的执行轨迹信息（第1轮）：
测试用例：quick_sort([-1, -3, 2, 0])
执行过程：
1. 输入数组: [-1, -3, 2, 0]
2. 选择基准值: -1
3. 分区结果 - 小于等于基准: [-3], 大于基准: [2, 0]
4. 递归调用quick_sort([-3])
   - 输入数组: [-3]
   - 基础情况，返回: [-3]
5. 递归调用quick_sort([2, 0])
   - 输入数组: [2, 0]
   - 选择基准值: 2
   - 分区结果 - 小于等于基准: [0], 大于基准: []
   - 递归调用quick_sort([0])
     - 输入数组: [0]
     - 基础情况，返回: [0]
   - 递归调用quick_sort([])
     - 输入数组: []
     - 基础情况，返回: []
   - 合并结果: [0] + [2] + [] = [0, 2]
6. 合并结果: [-3] + [-1] + [0, 2] = [-3, -1, 0, 2]

问题定位：
- 分区逻辑正确，但合并顺序错误
- 应该是less + [pivot] + greater，但实际结果是[-3, -1, 0, 2]而不是[-3, -1, 0, 2]
- 实际上结果是正确的！可能是测试用例期望的问题，让我们重新检查"""
            elif debug_round == 2:
                trace_content = """收集到的执行轨迹信息（第2轮）：
重新检查测试用例：quick_sort([-1, -3, 2, 0])
期望结果: [-3, -1, 0, 2]
实际结果: [-3, -1, 0, 2]

结果实际上是正确的！问题可能在测试用例本身或者我们的理解上。
让我们检查其他可能的问题...

通过更详细的分析，发现问题可能在于：
1. 算法逻辑本身是正确的
2. 但可能存在性能问题（最坏情况O(n^2)）
3. 或者在某些特殊输入下会出现问题"""
            else:  # 第3轮
                trace_content = """收集到的执行轨迹信息（第3轮）：
经过更深入的分析，发现了一个隐藏的问题：
当数组中有大量重复元素时，当前的分区方法效率很低。

测试用例：quick_sort([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
执行过程显示：
1. 每次选择第一个元素作为基准
2. 所有其他元素要么小于等于基准，要么大于基准
3. 由于所有元素都相等，greater数组始终为空
4. 这导致递归深度为O(n)，时间复杂度退化为O(n^2)

问题定位：
- 分区策略不够优化
- 对于大量重复元素的处理效率低
- 可能导致栈溢出或超时"""
            
            steps.append({
                "step": step_counter,
                "title": f"收集执行轨迹（第{debug_round}轮）",
                "description": "收集代码执行过程中的详细轨迹信息",
                "content": trace_content,
                "status": "completed"
            })
            step_counter += 1
            
            # 步骤8: 分析轨迹并生成修复计划
            if debug_round == 1:
                analysis_result = f"""分析智能体诊断结果（第{debug_round}轮）：
1. 问题定位：初步分析显示算法逻辑正确，但存在隐藏问题
2. 根本原因：由于没有历史修复经验（HLLM为空），分析基于当前执行轨迹
3. 影响范围：可能在特殊输入下表现不佳
4. 修复建议：
   a. 检查测试用例是否正确
   b. 优化分区策略处理重复元素
   c. 考虑三路分区法提高效率"""
            elif debug_round == 2:
                analysis_result = f"""分析智能体诊断结果（第{debug_round}轮）：
结合HLLM中的历史经验（第1轮修复记录）：
1. 问题定位：分区策略对重复元素处理效率低
2. 根本原因：传统二路分区在处理大量重复元素时性能差
3. 修复建议（基于HLLM学习）：
   a. 采用三路分区法（Dijkstra的解法）
   b. 将数组分为小于、等于、大于基准值三部分
   c. 递归只处理小于和大于部分"""
            else:  # 第3轮
                analysis_result = f"""分析智能体诊断结果（第{debug_round}轮）：
结合HLLM中的历史经验（第1、2轮修复记录）：
1. 问题定位：算法在大量重复元素下性能差
2. 根本原因：分区策略未优化
3. 修复建议（基于HLLM学习和当前分析）：
   a. 实现三路快排解决重复元素问题
   b. 添加随机化基准选择避免最坏情况
   c. 对小数组使用插入排序优化"""
            
            steps.append({
                "step": step_counter,
                "title": f"分析轨迹并生成修复计划（第{debug_round}轮）",
                "description": "分析智能体分析执行轨迹并生成详细的修复计划",
                "content": analysis_result,
                "prompt_tokens": 320,
                "completion_tokens": 260,
                "status": "completed"
            })
            step_counter += 1
            
            # 步骤9: 提出新插桩方案
            steps.append({
                "step": step_counter,
                "title": f"提出新插桩方案（第{debug_round}轮）",
                "description": "基于分析结果提出新的插桩建议",
                "content": f"""根据分析结果，提出新的插桩方案（第{debug_round}轮）：
1. 在分区函数处添加详细日志
2. 在基准选择处添加随机化监控
3. 在递归深度处添加栈深度监控
4. 在性能关键路径添加计时器""",
                "status": "completed"
            })
            step_counter += 1
            
            # 步骤10: 修复代码
            if debug_round == 1:
                repaired_code = """def quick_sort(arr):
    # 第1轮修复：添加注释和基本优化
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]  # 选择第一个元素作为基准
        # 二路分区
        less = [x for x in arr[1:] if x <= pivot]
        greater = [x for x in arr[1:] if x > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greater)"""
                current_code = repaired_code
                passed_count = 6  # 修复后通过数
            elif debug_round == 2:
                repaired_code = """def quick_sort(arr):
    # 第2轮修复：尝试使用中间元素作为基准
    if len(arr) <= 1:
        return arr
    else:
        # 选择中间元素作为基准
        pivot = arr[len(arr) // 2]
        less = [x for x in arr if x < pivot]
        equal = [x for x in arr if x == pivot]
        greater = [x for x in arr if x > pivot]
        return quick_sort(less) + equal + quick_sort(greater)"""
                current_code = repaired_code
                passed_count = 5  # 修复后通过数减少
            else:  # 第3轮
                repaired_code = """def quick_sort(arr):
    # 第3轮修复：采用随机基准的三路快排
    import random
    
    if len(arr) <= 1:
        return arr
    else:
        # 随机选择基准元素
        pivot_index = random.randint(0, len(arr) - 1)
        pivot = arr[pivot_index]
        
        # 三路分区
        less = [x for x in arr if x < pivot]
        equal = [x for x in arr if x == pivot]
        greater = [x for x in arr if x > pivot]
        
        return quick_sort(less) + equal + quick_sort(greater)"""
                current_code = repaired_code
                passed_count = 7  # 修复后通过数增加并全部通过
            
            steps.append({
                "step": step_counter,
                "title": f"修复代码（第{debug_round}轮）",
                "description": "修复智能体根据修复计划修改代码",
                "content": repaired_code,
                "prompt_tokens": 220,
                "completion_tokens": 240,
                "status": "completed"
            })
            step_counter += 1
            
            # 步骤11: 测试修复后的代码
            if debug_round == 1:
                test_result = f"""第{debug_round}轮修复后代码测试结果：
assert quick_sort([]) == []                    # 通过
assert quick_sort([1]) == [1]                  # 通过
assert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]  # 通过
assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]  # 通过
assert quick_sort([3, 6, 8, 10, 1, 2, 1]) == [1, 1, 2, 3, 6, 8, 10]  # 通过
assert quick_sort([4, 2, 4, 2, 4, 2]) == [2, 2, 2, 4, 4, 4]  # 通过
assert quick_sort([-1, -3, 2, 0]) == [-3, -1, 0, 2]  # 失败

测试通过数：6/7个测试用例通过！
性能分析显示仍有优化空间。"""
                passed_count = 6  # 修复后通过数
                repair_successful = True
            elif debug_round == 2:
                test_result = f"""第{debug_round}轮修复后代码测试结果：
assert quick_sort([]) == []                    # 通过
assert quick_sort([1]) == [1]                  # 通过
assert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]  # 通过
assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]  # 通过
assert quick_sort([3, 6, 8, 10, 1, 2, 1]) == [1, 1, 2, 3, 6, 8, 10]  # 失败
assert quick_sort([4, 2, 4, 2, 4, 2]) == [2, 2, 2, 4, 4, 4]  # 通过
assert quick_sort([-1, -3, 2, 0]) == [-3, -1, 0, 2]  # 通过

测试通过数：5/7个测试用例通过！
注意：通过测试用例数反而减少了！"""
                passed_count = 5  # 修复后通过数减少
                repair_successful = False
            else:  # 第3轮
                test_result = f"""第{debug_round}轮修复后代码测试结果：
assert quick_sort([]) == []                    # 通过
assert quick_sort([1]) == [1]                  # 通过
assert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]  # 通过
assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]  # 通过
assert quick_sort([3, 6, 8, 10, 1, 2, 1]) == [1, 1, 2, 3, 6, 8, 10]  # 通过
assert quick_sort([4, 2, 4, 2, 4, 2]) == [2, 2, 2, 4, 4, 4]  # 通过
assert quick_sort([-1, -3, 2, 0]) == [-3, -1, 0, 2]  # 通过

测试通过数：7/7个测试用例通过！
所有测试用例通过！"""
                passed_count = 7  # 修复后通过数增加并全部通过
                repair_successful = True
            
            steps.append({
                "step": step_counter,
                "title": f"测试修复后的代码（第{debug_round}轮）",
                "description": "执行修复后的代码并验证结果",
                "content": test_result,
                "status": "completed"
            })
            step_counter += 1
            
            # 步骤12: 检查测试覆盖率是否改善
            steps.append({
                "step": step_counter,
                "title": f"检查测试结果改善情况（第{debug_round}轮）",
                "description": "检查修复后的代码是否通过了更多测试",
                "content": f"第{debug_round}轮修复{'成功' if repair_successful else '失败'}，当前通过{passed_count}/7个测试用例，历史最佳{best_passed_count}/7个测试用例",
                "status": "completed"
            })
            step_counter += 1
            
            # 步骤13: HLLM和RM机制
            if debug_round == 1:
                # 第1轮：通过测试用例数量为6个，保存历史经验
                steps.append({
                    "step": step_counter,
                    "title": f"保存修复经验到HLLM（第{debug_round}轮）",
                    "description": "将第一轮修复经验保存到历史学习机制中",
                    "content": f"""HLLM（历史学习机制）记录（第{debug_round}轮）：
问题模式：快速排序算法基础实现问题
错误特征：
- 分区逻辑正确但基准选择不够优化
- 通过测试用例数：6/7个通过
- 性能分析显示仍有优化空间

修复方案：
- 第{debug_round}轮：优化分区逻辑，保持基准选择策略
- 添加详细注释提高代码可读性

经验总结：
- 基准选择对快排性能影响较大
- 分区逻辑需要正确处理边界情况
- 通过测试用例数是衡量修复效果的重要指标

已将此经验保存到HLLM知识库，供未来类似问题参考。""",
                    "status": "completed"
                })
                step_counter += 1
                
                # 更新最佳代码和通过数
                best_passed_count = passed_count
                best_code = current_code
                consecutive_no_improvement = 0
            elif debug_round == 2:
                # 第2轮：通过测试用例数量减少（从6个减少到5个），触发RM回滚机制
                steps.append({
                    "step": step_counter,
                    "title": f"触发RM回滚机制（第{debug_round}轮）",
                    "description": "修复后通过测试用例数减少，触发回滚机制",
                    "content": f"""RM（回滚机制）触发（第{debug_round}轮）：
1. 检测到修复后测试通过数减少（当前{passed_count}，历史最佳{best_passed_count}）
2. 第2轮修复尝试使用中间元素作为基准但效果不佳
3. 回滚到历史最佳版本（通过{best_passed_count}个测试用例）
4. 将本次失败的修复经验保存到HLLM：
   - 修复方案：使用中间元素作为基准的三路分区法
   - 失败原因：基准选择策略不当导致通过测试用例数减少
   - 后续策略：尝试随机基准选择或其他优化方向

HLLM（历史学习机制）更新记录：
- 添加了新的失败模式：中间元素基准选择问题
- 更新了基准选择策略的经验库
- 为下一轮修复提供参考：避免使用中间元素作为基准""",
                    "status": "completed"
                })
                step_counter += 1
                
                # 回滚到最佳代码（使用第1轮的代码）
                current_code = best_code
                consecutive_no_improvement += 1
            else:
                # 第3轮：通过测试用例数量增加并全部通过（从6个增加到7个），使用新修复的代码
                steps.append({
                    "step": step_counter,
                    "title": f"保存修复经验到HLLM（第{debug_round}轮）",
                    "description": "将第三轮成功的修复经验保存到历史学习机制中",
                    "content": f"""HLLM（历史学习机制）记录（第{debug_round}轮）：
问题模式：快速排序算法优化与鲁棒性提升
错误特征：
- 经过RM机制回滚后的版本通过6个测试用例
- 需要进一步优化以通过所有测试用例

修复方案：
- 第{debug_round}轮：采用随机基准的三路快排算法
- 有效处理重复元素和各种边界情况

经验总结：
- 随机基准选择可以避免最坏情况的发生
- 三路分区法在处理重复元素时效率更高
- 组合使用多种优化策略可以显著提升算法鲁棒性

已将此经验保存到HLLM知识库，供未来类似问题参考。""",
                    "status": "completed"
                })
                step_counter += 1
                
                # 更新最佳代码和通过数（第3轮修复成功，通过所有测试用例）
                best_passed_count = passed_count
                best_code = current_code
                consecutive_no_improvement = 0
            
            # 更新当前代码
            current_code = repaired_code
            current_passed_count = passed_count
            
            # 如果所有测试都通过了，结束循环
            if passed_count == 7 and debug_round == 3:
                break
        
        # 步骤14: 解决方案完成
        steps.append({
            "step": step_counter,
            "title": "解决方案完成",
            "description": "TraceCoder流程完成",
            "content": "问题已成功解决！代码功能正确，经过多轮优化后性能也得到显著提升。",
            "status": "completed"
        })
        
        return steps