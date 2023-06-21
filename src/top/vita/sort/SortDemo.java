package top.vita.sort;

import java.util.*;

public class SortDemo {

	public static void main(String[] args) {
//		int[] nums = {2,2,1};
//		System.out.println(singleNumber(nums));
//		int[] nums = { 3, 2, 3 };
//		System.out.println(majorityElement(nums));
//		int[] nums = { -1, 0, 1, 2, -1, -4 };
//		List<List<Integer>> threeSum = threeSum(nums);
//		System.out.println(threeSum);
//		System.out.println(hammingDistance(1,4));

//		System.out.println(hasAlternatingBits(5));
		int[] arr = {2,0,2,1,1,0,2,2};
		sortColors(arr);
//		int[][] intervals = {{1,3},{2,6},{8,10},{15,18}};
//		int[][] merge = merge(intervals);
//		for (int[] row : merge) {
//			System.out.println(Arrays.toString(row));
//		}
	}

	/*
	 * 136. 只出现一次的数字 给定一个非空整数数组，
	 * 除了某个元素只出现一次以外，其余每个元素均出现两次。
	 * 找出那个只出现了一次的元素。
	 */
	 public static int singleNumber(int[] nums) {
	        int res = 0;
	        for(int i = 0;i<nums.length;i++){
	            res ^= nums[i];
	        }
	        return res;
	    }
	
	public static int majorityElement(int[] nums) {
		// 从第一个数开始count=1，
		// 遇到相同的就加1，遇到不同的就减1，
		// 减到0就重新换个数开始计数，总能找到最多的那个
		int count = 0;
		int maj = nums[0];
		for (int i = 0; i < nums.length; i++) {
			if (nums[i] == maj) {
				count++;
			} else {
				count--;
				if (count == 0) {
					count = 1;
					maj = nums[i];
				}
			}
		}
		return maj;
	}

	/*
	 * 给你一个包含 n 个整数的数组 nums， 判断 nums 中是否存在三个元素 a，b，c ， 使得 a + b + c = 0 ？请你找出所有和为 0
	 * 且不重复的三元组。
	 */
	public static List<List<Integer>> threeSum(int[] nums) {
		List<List<Integer>> res = new ArrayList<>();
		Arrays.sort(nums);
		for (int i = 0; i < nums.length - 2; i++) {
			// 去重，因为排序后相同的数字都在一起，所以可以这样用
			if (i > 0 && nums[i] == nums[i - 1])
				continue;
			// 保存目标数
			int target = -nums[i];
			int left = i + 1;
			int right = nums.length - 1;
			while (left < right) {
				// 两数相加等于目标数则找到
				int sum = nums[left] + nums[right];
				if (sum == target) {
					res.add(Arrays.asList(nums[i], nums[left], nums[right]));
					// left++;
					// right--;
					// 去重，前后有相同数时跳过
					while (left < right && nums[left] == nums[++left])
						;
					while (left < right && nums[right] == nums[--right])
						;
				}else if (sum < target)
					left++;
				else
					right--;
			}
		}
		return res;
	}
	
	/*
	 * 461. 汉明距离
	 * 两个整数之间的 汉明距离 
	 * 指的是这两个数字对应二进制位不同的位置的数目。
	 *	给你两个整数 x 和 y，计算并返回它们之间的汉明距离。
	 */
	public static int hammingDistance(int x, int y) {
		int n = (x^y);
		int count = 0;
		while (n != 0) {
			n &= (n - 1);
			count ++;
		}
		return count;
    }
	
	/*
	 * 693. 交替位二进制数
	 * 给定一个正整数，检查它的二进制表示是否总是 0、1 交替出现：
	 * 换句话说，就是二进制表示中相邻两位的数字永不相同。
	 */
    public static boolean hasAlternatingBits(int n) {
		while (n != 0) {
			if ((n & 3) == 3 || (n & 3) == 0) {
				return false;
			}
			System.out.println(n);
			// 右移 0101 => 0010
			n >>= 1;
			System.out.println(n);
		}
		return true;
    }
    
    /*
	 * 75. 颜色分类 给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums
	 * ，原地对它们进行排序，使得相同颜色的元素相邻，
	 * 并按照红色、白色、蓝色顺序排列。
	 *  我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
	 * 必须在不使用库的sort函数的情况下解决这个问题。
	 */
    public static void sortColors(int[] nums) {
    	int lastZero = -1;
    	int firstTwo = nums.length;
    	int current = 0;
    	for (int i = 0; i < nums.length; i++) {
			if (nums[current] == 0) {
				exchange(nums, ++lastZero, current++);
			}else if(nums[current] == 1) {
				current ++;
			}else {
				exchange(nums,--firstTwo, current);
			}
		}
    	System.out.println(Arrays.toString(nums));
    }
    
    private static void exchange(int[] nums,int left,int right) {
    	int temp = nums[left];
    	nums[left] = nums[right];
    	nums[right] = temp;
    }
    
	/*
	 * 56. 合并区间 以数组 intervals 表示若干个区间的集合，
	 * 其中单个区间为 intervals[i] = [starti, endi]
	 * 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，
	 * 该数组需恰好覆盖输入中的所有区间 。
	 */
    public static int[][] merge(int[][] intervals) {
    	// 按第一位数升序排序
    	Arrays.sort(intervals,new Comparator<int[]>() {
    		@Override
    		public int compare(int[] o1, int[] o2) {
    			return o1[0]-o2[0];
    		}
		});
    	List<int[]> out = new ArrayList<int[]>();
    	// 直接将第一个区间加入
    	out.add(intervals[0]);
		for (int i = 1; i < intervals.length; i++) {
			// 当区间的左区间大于上一个区间的右区间，则代表区间不重复
			if (intervals[i][0] > out.get(out.size() - 1)[1]) {
				out.add(intervals[i]);
			} else { // 否则区间重复，取两个区间的右区间最大值
				out.get(out.size() -1)[1] = Math.max(intervals[i][1], out.get(out.size() - 1)[1]);
			}
		}
    	return out.toArray(new int[out.size()][]);
    }

	public String minNumber(int[] nums) {
		String[] strs = new String[nums.length];
		for (int i = 0; i < nums.length; i++) {
			strs[i] = nums[i] + "";
		}
		Arrays.sort(strs, (a, b) -> (a + b).compareTo(b + a));
		StringBuilder sb = new StringBuilder();
		for (String str : strs) {
			sb.append(str);
		}
		return sb.toString();
	}

	public boolean isStraight(int[] nums) {
		Set<Integer> set = new HashSet<>();
		int min = 14;
		int max = 0;
		// 满足max - min < 5 即可组成顺子
		for (int i : nums) {
			if (i == 0) {
				continue;
			}
			min = Math.min(min, i);
			max = Math.max(max, i);
			if (set.contains(i)) {
				return false;
			}
			set.add(i);
		}
		return max - min < 5;
	}

	public boolean isStraight1(int[] nums) {
		int joker = 0;
		Arrays.sort(nums);
		for (int i = 0; i < nums.length - 1; i++) {
			// 可以有多个大小王
			if (nums[i] == 0) {
				joker++;
			} else if (nums[i] == nums[i + 1]) {
				return false;
			}
		}
		// max - min < 5
		return nums[nums.length - 1] - nums[joker] < 5;
	}
}
