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
	 * 136. ֻ����һ�ε����� ����һ���ǿ��������飬
	 * ����ĳ��Ԫ��ֻ����һ�����⣬����ÿ��Ԫ�ؾ��������Ρ�
	 * �ҳ��Ǹ�ֻ������һ�ε�Ԫ�ء�
	 */
	 public static int singleNumber(int[] nums) {
	        int res = 0;
	        for(int i = 0;i<nums.length;i++){
	            res ^= nums[i];
	        }
	        return res;
	    }
	
	public static int majorityElement(int[] nums) {
		// �ӵ�һ������ʼcount=1��
		// ������ͬ�ľͼ�1��������ͬ�ľͼ�1��
		// ����0�����»�������ʼ�����������ҵ������Ǹ�
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
	 * ����һ������ n ������������ nums�� �ж� nums ���Ƿ��������Ԫ�� a��b��c �� ʹ�� a + b + c = 0 �������ҳ����к�Ϊ 0
	 * �Ҳ��ظ�����Ԫ�顣
	 */
	public static List<List<Integer>> threeSum(int[] nums) {
		List<List<Integer>> res = new ArrayList<>();
		Arrays.sort(nums);
		for (int i = 0; i < nums.length - 2; i++) {
			// ȥ�أ���Ϊ�������ͬ�����ֶ���һ�����Կ���������
			if (i > 0 && nums[i] == nums[i - 1])
				continue;
			// ����Ŀ����
			int target = -nums[i];
			int left = i + 1;
			int right = nums.length - 1;
			while (left < right) {
				// ������ӵ���Ŀ�������ҵ�
				int sum = nums[left] + nums[right];
				if (sum == target) {
					res.add(Arrays.asList(nums[i], nums[left], nums[right]));
					// left++;
					// right--;
					// ȥ�أ�ǰ������ͬ��ʱ����
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
	 * 461. ��������
	 * ��������֮��� �������� 
	 * ָ�������������ֶ�Ӧ������λ��ͬ��λ�õ���Ŀ��
	 *	������������ x �� y�����㲢��������֮��ĺ������롣
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
	 * 693. ����λ��������
	 * ����һ����������������Ķ����Ʊ�ʾ�Ƿ����� 0��1 ������֣�
	 * ���仰˵�����Ƕ����Ʊ�ʾ��������λ������������ͬ��
	 */
    public static boolean hasAlternatingBits(int n) {
		while (n != 0) {
			if ((n & 3) == 3 || (n & 3) == 0) {
				return false;
			}
			System.out.println(n);
			// ���� 0101 => 0010
			n >>= 1;
			System.out.println(n);
		}
		return true;
    }
    
    /*
	 * 75. ��ɫ���� ����һ��������ɫ����ɫ����ɫ���� n ��Ԫ�ص����� nums
	 * ��ԭ�ض����ǽ�������ʹ����ͬ��ɫ��Ԫ�����ڣ�
	 * �����պ�ɫ����ɫ����ɫ˳�����С�
	 *  ����ʹ������ 0�� 1 �� 2 �ֱ��ʾ��ɫ����ɫ����ɫ��
	 * �����ڲ�ʹ�ÿ��sort����������½��������⡣
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
	 * 56. �ϲ����� ������ intervals ��ʾ���ɸ�����ļ��ϣ�
	 * ���е�������Ϊ intervals[i] = [starti, endi]
	 * ������ϲ������ص������䣬������ һ�����ص����������飬
	 * ��������ǡ�ø��������е��������� ��
	 */
    public static int[][] merge(int[][] intervals) {
    	// ����һλ����������
    	Arrays.sort(intervals,new Comparator<int[]>() {
    		@Override
    		public int compare(int[] o1, int[] o2) {
    			return o1[0]-o2[0];
    		}
		});
    	List<int[]> out = new ArrayList<int[]>();
    	// ֱ�ӽ���һ���������
    	out.add(intervals[0]);
		for (int i = 1; i < intervals.length; i++) {
			// ������������������һ������������䣬��������䲻�ظ�
			if (intervals[i][0] > out.get(out.size() - 1)[1]) {
				out.add(intervals[i]);
			} else { // ���������ظ���ȡ������������������ֵ
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
		// ����max - min < 5 �������˳��
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
			// �����ж����С��
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
