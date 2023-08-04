package top.vita.sort;

public class BinarySearch {

	public static void main(String[] args) {
//		int arr[] = { -1, 0, 3, 5, 9, 12 };
//		System.out.println(binarySearch(arr, 9));
//		int[] arr = {3,2,2,3};
//		System.out.println(removeElement(arr, 3));
		
//		int[] arr = {0,0,1,1,1,2,2,3,3,4};
//		System.out.println(removeDuplicates(arr));
		
//		int[] nums = {0,1,0,3,12};
//		moveZeroes(nums);
		
//		System.out.println(backspaceCompare("123#12", "1212"));
		
//		int[] nums = {-4,-1,0,3,10};
//		System.out.println(Arrays.toString(sortedSquares(nums)));
		
		int[] nums = {2,3,1,2,4,3};
		System.out.println(minSubArrayLen(7,nums));
		
//		int[] nums = {20,100,10,12,5,13};
//		System.out.println(increasingTriplet(nums));
		
//		int[] nums = {1,2,3,4};
//		System.out.println(Arrays.toString(productExceptSelf(nums)));
		
//		int[] nums = {1, 7, 3, 6, 5, 6};
//		System.out.println(pivotIndex(nums));
		
//		int[] nums = {1,1,1};
//		System.out.println(subarraySum(nums, 2));
	}

	public static int binarySearch(int[] arr, int target) {
		int left = 0;
		int right = arr.length;
		while (left < right) {
			int mid = (left + right) >> 1;
			if (target > arr[mid]) {
				left = mid + 1;
			} else if (target < arr[mid]) {
				right = mid;
			} else {
				return mid;
			}
		}
		return -1;
	}
	
	/*
	 * 27. 移除元素 给你一个数组 nums 和一个值 val，
	 * 你需要 原地 移除所有数值等于 val 的元素，
	 * 并返回移除后数组的新长度。
	 * 不要使用额外的数组空间，
	 * 你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
	 * 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
	 */
    public static int removeElement(int[] nums, int val) {
    	// 定义慢指针
    	int slow = 0;
    	// 快指针移动
    	for (int fast = 0; fast < nums.length; fast++) {
    		// 不等于要删除的元素时，慢指针才会移动
    		// 否则指向要删除元素的下标
			if (nums[fast] != val) {
				// 将快指针指向的值移动到慢指针指向的位置
				nums[slow] = nums[fast];
				// 慢指针移动
				slow ++;
			}
		}
    	// 此时slow的下标就是数组的长度
    	return slow;
    }
    
    /*
	 * 26. 删除有序数组中的重复项 给你一个 升序排列 的数组 nums ，
	 * 请你 原地 删除重复出现的元素，使每个元素 只出现一次
	 * ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。
	 */
    public static int removeDuplicates(int[] nums) {
    	int slow = 0;
    	for (int fast = 1; fast < nums.length; fast++) {
			if (nums[fast] != nums[slow]) {
				nums[++slow] = nums[fast];
			}
		}
    	return ++slow;
    }
    
    /*
	 * 283. 移动零 给定一个数组 nums，
	 * 编写一个函数将所有 0 移动到数组的末尾，
	 * 同时保持非零元素的相对顺序。
	 */
    public static void moveZeroes(int[] nums) {
    	int slow = 0;
    	for (int fast = 0; fast < nums.length; fast++) {
			if (nums[fast] != 0) {
				nums[slow++] = nums[fast];
			}
		}
    	while (slow < nums.length) {
    		nums[slow++] = 0;
    	}
    }

    /*
	 * 844. 比较含退格的字符串 给定 s 和 t 两个字符串，
	 * 当它们分别被输入到空白的文本编辑器后， 如果两者相等，
	 * 返回 true 。# 代表退格字符。
	 */
//    public static boolean backspaceCompare(String s, String t) {
//    	String str = "";
//    	char[] charArray = s.toCharArray();
//    	for (int i = 0; i < charArray.length; i++) {
//			if (charArray[i] == '#') {
//				str += charArray[i];
//			}else {
//			}
//		}
//    	
//		return s.equals(t);
//    }
    
    /*
	 * 977. 有序数组的平方 给你一个按 非递减顺序 排序的整数数组
	 *  nums，返回 每个数字的平方 组成的新数组，
	 *  要求也按 非递减顺序 排序。
	 */
    public static int[] sortedSquares(int[] nums) {
    	int[] res = new int[nums.length];
    	int j = nums.length - 1;
		int k = res.length - 1;
		// 此处for循环没有进行增加操作
		for (int i = 0; i <= j;) {
			if (nums[i] * nums[i] > nums[j] * nums[j]) {
				res[k--] = nums[i] * nums[i];
				i ++;
			}else {
				res[k--] = nums[j] * nums[j];
				j --;
			}
		}
    	return res;
    }
    
	/*
	 * 209. 长度最小的子数组 
	 * 给定一个含有 n 个正整数的数组和一个正整数 target 。
	 * 找出该数组中满足其和 ≥ target 的长度最小的 
	 * 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr]
	 * 并返回其长度。如果不存在符合条件的子数组，返回 0 。
	 */
	public static int minSubArrayLen(int target, int[] nums) {
		int res = Integer.MAX_VALUE;
		int left = 0; // 滑动窗口的起点
		int right = 0; // 滑动窗口的终点
		int sum = 0;
		for (right = 0; right < nums.length; right++) {
			sum += nums[right];
			while (sum >= target) {
//				res = res < right - left + 1 ? res : right - left + 1;
				res = Math.min(res, right - left + 1); // 取长度最小的作为结果
				sum -= nums[left++];
			}
		}
		return res == Integer.MAX_VALUE ? 0 : res;
	}
    
	/*
	 * 334. 递增的三元子序列 给你一个整数数组 nums ，
	 * 判断这个数组中是否存在长度为 3 的递增子序列。
	 */
    public static boolean increasingTriplet(int[] nums) {
    	int first = nums[0];
    	int second = Integer.MAX_VALUE;
    	for (int i = 0; i < nums.length; i++) {
			int num = nums[i];
			// 当找到比second大的数时，代表找到此序列
			if (num > second) {
				return true;
			}else if (num > first) { 
				// 当num比first大时，将second重新赋值
				// 尽可能的让second的数小
				second = num;
			}else{
				// 小于first的情况，将first也重新赋值
				// 尽可能的让first的数小
				first = num;
			}
		}
    	return false;
    }
    
    /*
	 * 238. 除自身以外数组的乘积 给你一个整数数组 nums，
	 * 返回 数组 res ，其中 res[i] 等于 nums 中除 nums[i]
	 * 之外其余各元素的乘积 。
	 */
    public static int[] productExceptSelf(int[] nums) {
		int[] res = new int[nums.length];
		for (int i = 0; i < res.length; i++) {
			res[i] = 1;
		}

		int left = 1;
		int right = 1;
		for (int i = 0; i < nums.length; i++) {
			// 1 1 1 1
			// left = 1 right = 4
			// 1 1 4 1
			// left = 2 right = 12 
			// 1 12 8 1
			// left = 6 right = 24
			// 24 12 8 6
			// left = 24 right = 24
			res[i] *= left;
			left *= nums[i];

			res[nums.length - i - 1] *= right;
			right *= nums[nums.length - i - 1];
		}
		return res;
    }

	/*
	 * 724. 寻找数组的中心下标
	 *  给你一个整数数组 nums ，请计算数组的 中心下标 。
	 */
    public static int pivotIndex(int[] nums) {
    	// 记录总和
    	int presum = 0;
    	for (int i : nums) {
			presum += i;
		}
    	
    	// 记录前缀和
    	int leftsum = 0;
    	for (int i = 0; i < nums.length; i++) {
    		// leftsum代表左半部分，presum - leftsum代表右半部分
    		// 因为还没有加此次循环的nums[i]，所以减去
			if (leftsum == presum - nums[i] - leftsum) {
				return i;
			}
			leftsum += nums[i];
		}
    	
    	return -1;
    }
    
    /*
	 * 560. 和为 K 的子数组 
	 * 给你一个整数数组 nums 和一个整数 k ，
	 * 请你统计并返回 该数组中和为 k 的连续子数组的个数 。
	 */
	public static int subarraySum(int[] nums, int k) {
		int sum = 0;
		int count = 0;
		for (int i = 0; i < nums.length; i++) {
			for (int j = i; j < nums.length; j++) {
				sum += nums[j];
				if (sum == k) {
					count ++;
				}
			}
			sum = 0;
		}
		return count;
	}
    
    
    
    
}
