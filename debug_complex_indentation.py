import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_demo.trace_runner import TraceCoderRunner

# 创建测试代码
code = """def task_func(numbers=None):
    if numbers is None:
        numbers = [1, 2, 3, 4, 5]
    if len(numbers) < 2:
        return 0.0
    # 简单实现：计算相邻元素差值的平均值的绝对值
    diffs = [abs(numbers[i+1] - numbers[i]) for i in range(len(numbers)-1)]
    return sum(diffs) / len(diffs)"""

# 创建有复杂缩进问题的unittest测试用例（包含with语句）
test_cases = """import unittest
from unittest.mock import patch
from random import seed, shuffle

class TestCases(unittest.TestCase):
    def test_default_numbers(self):
        # Test with default number range (1 to 10) to check that the result is a positive float.
        result = task_func()
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
    
    def test_specific_value_with_seed(self):
        # Set seed for reproducibility and check the computed value
        with patch('random.shuffle', side_effect=lambda x: seed(42) or shuffle(x)):
            result = task_func([1, 2, 3])
            self.assertAlmostEqual(result, 2.5, delta=0.5)"""

# 创建TraceCoderRunner实例
runner = TraceCoderRunner()

# 测试_simple_check_correctness方法
result = runner._simple_check_correctness(test_cases, code, 10)

print("测试结果:")
print(result)

# 让我们看看修复后的测试用例是什么样的
fixed_test_cases = runner._fix_unittest_indentation(test_cases)
print("\n修复后的测试用例:")
print(repr(fixed_test_cases))