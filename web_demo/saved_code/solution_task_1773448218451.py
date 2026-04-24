def threeSum(nums):
    # 首先对数组进行排序
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        # 如果当前数字大于0，则三数之和一定大于0，所以结束循环
        if nums[i] > 0:
            break
        # 跳过重复的数字
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        # 使用双指针来寻找另外两个数
        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                # 移动左指针并跳过重复的数字
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                # 移动右指针并跳过重复的数字
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    
    return result

# 示例用法
nums = [-1, 0, 1, 2, -1, -4]
print(threeSum(nums))  # 输出: [[-1, -1, 2], [-1, 0, 1]]