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
	 * 27. �Ƴ�Ԫ�� ����һ������ nums ��һ��ֵ val��
	 * ����Ҫ ԭ�� �Ƴ�������ֵ���� val ��Ԫ�أ�
	 * �������Ƴ���������³��ȡ�
	 * ��Ҫʹ�ö��������ռ䣬
	 * ������ʹ�� O(1) ����ռ䲢 ԭ�� �޸��������顣
	 * Ԫ�ص�˳����Ըı䡣�㲻��Ҫ���������г����³��Ⱥ����Ԫ�ء�
	 */
    public static int removeElement(int[] nums, int val) {
    	// ������ָ��
    	int slow = 0;
    	// ��ָ���ƶ�
    	for (int fast = 0; fast < nums.length; fast++) {
    		// ������Ҫɾ����Ԫ��ʱ����ָ��Ż��ƶ�
    		// ����ָ��Ҫɾ��Ԫ�ص��±�
			if (nums[fast] != val) {
				// ����ָ��ָ���ֵ�ƶ�����ָ��ָ���λ��
				nums[slow] = nums[fast];
				// ��ָ���ƶ�
				slow ++;
			}
		}
    	// ��ʱslow���±��������ĳ���
    	return slow;
    }
    
    /*
	 * 26. ɾ�����������е��ظ��� ����һ�� �������� ������ nums ��
	 * ���� ԭ�� ɾ���ظ����ֵ�Ԫ�أ�ʹÿ��Ԫ�� ֻ����һ��
	 * ������ɾ����������³��ȡ�Ԫ�ص� ���˳�� Ӧ�ñ��� һ�� ��
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
	 * 283. �ƶ��� ����һ������ nums��
	 * ��дһ������������ 0 �ƶ��������ĩβ��
	 * ͬʱ���ַ���Ԫ�ص����˳��
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
	 * 844. �ȽϺ��˸���ַ��� ���� s �� t �����ַ�����
	 * �����Ƿֱ����뵽�հ׵��ı��༭���� ���������ȣ�
	 * ���� true ��# �����˸��ַ���
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
	 * 977. ���������ƽ�� ����һ���� �ǵݼ�˳�� �������������
	 *  nums������ ÿ�����ֵ�ƽ�� ��ɵ������飬
	 *  Ҫ��Ҳ�� �ǵݼ�˳�� ����
	 */
    public static int[] sortedSquares(int[] nums) {
    	int[] res = new int[nums.length];
    	int j = nums.length - 1;
		int k = res.length - 1;
		// �˴�forѭ��û�н������Ӳ���
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
	 * 209. ������С�������� 
	 * ����һ������ n ���������������һ�������� target ��
	 * �ҳ���������������� �� target �ĳ�����С�� 
	 * ���������� [numsl, numsl+1, ..., numsr-1, numsr]
	 * �������䳤�ȡ���������ڷ��������������飬���� 0 ��
	 */
	public static int minSubArrayLen(int target, int[] nums) {
		int res = Integer.MAX_VALUE;
		int left = 0; // �������ڵ����
		int right = 0; // �������ڵ��յ�
		int sum = 0;
		for (right = 0; right < nums.length; right++) {
			sum += nums[right];
			while (sum >= target) {
//				res = res < right - left + 1 ? res : right - left + 1;
				res = Math.min(res, right - left + 1); // ȡ������С����Ϊ���
				sum -= nums[left++];
			}
		}
		return res == Integer.MAX_VALUE ? 0 : res;
	}
    
	/*
	 * 334. ��������Ԫ������ ����һ���������� nums ��
	 * �ж�����������Ƿ���ڳ���Ϊ 3 �ĵ��������С�
	 */
    public static boolean increasingTriplet(int[] nums) {
    	int first = nums[0];
    	int second = Integer.MAX_VALUE;
    	for (int i = 0; i < nums.length; i++) {
			int num = nums[i];
			// ���ҵ���second�����ʱ�������ҵ�������
			if (num > second) {
				return true;
			}else if (num > first) { 
				// ��num��first��ʱ����second���¸�ֵ
				// �����ܵ���second����С
				second = num;
			}else{
				// С��first���������firstҲ���¸�ֵ
				// �����ܵ���first����С
				first = num;
			}
		}
    	return false;
    }
    
    /*
	 * 238. ��������������ĳ˻� ����һ���������� nums��
	 * ���� ���� res ������ res[i] ���� nums �г� nums[i]
	 * ֮�������Ԫ�صĳ˻� ��
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
	 * 724. Ѱ������������±�
	 *  ����һ���������� nums ������������ �����±� ��
	 */
    public static int pivotIndex(int[] nums) {
    	// ��¼�ܺ�
    	int presum = 0;
    	for (int i : nums) {
			presum += i;
		}
    	
    	// ��¼ǰ׺��
    	int leftsum = 0;
    	for (int i = 0; i < nums.length; i++) {
    		// leftsum������벿�֣�presum - leftsum�����Ұ벿��
    		// ��Ϊ��û�мӴ˴�ѭ����nums[i]�����Լ�ȥ
			if (leftsum == presum - nums[i] - leftsum) {
				return i;
			}
			leftsum += nums[i];
		}
    	
    	return -1;
    }
    
    /*
	 * 560. ��Ϊ K �������� 
	 * ����һ���������� nums ��һ������ k ��
	 * ����ͳ�Ʋ����� �������к�Ϊ k ������������ĸ��� ��
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
