package top.vita.dp;

import java.util.List;

public class DPQuestion {

	public static void main(String[] args) {

	}

	/*
	 * 509. 쳲�������
	 */
    public int fib(int n) {
    	if (n == 0) {
    		return 0;
    	}
    	if (n <= 2) {
    		return 1;
    	}
    	int[] dp = new int[n + 1];
    	dp[0] = 0;
    	dp[1] = 1;
    	for (int i = 2; i <= n; i++) {
    		dp[i] = dp[i - 1] + dp[i - 2];
		}
    	return dp[n];
    }
	
	/*
	 * 70. ��¥��
	 */
    public int climbStairs(int n) {
    	if (n <= 1) {
    		return n;
    	}
    	int[] dp = new int[n + 1];
    	// dp[0] �����壬������ʼ��
    	dp[1] = 1;
    	dp[2] = 2;
    	for (int i = 3; i <= n; i++) {
			dp[i] = dp[i - 1] + dp[i - 2];
		}
    	return dp[n];
    }
    
    /*
     * 198. ��ҽ���
     */
    public int rob(int[] nums) {
    	int[] dp = new int[nums.length];
    	dp[0] = nums[0];
    	for (int i = 1; i < nums.length; i++) {
    		// ��ֹ��������
    		if (i == 1) {
    			dp[1] = Math.max(nums[0], nums[1]);
    		}else {
    			dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
    		}
		}
    	return dp[nums.length - 1];
    }
    
    /*
     * 33. ������ת��������
     */
    public int search(int[] nums, int target) {
    	int left = 0;
    	int right = nums.length - 1;
    	while (left <= right) {
    		int mid = left + (right - left) / 2;
    		if (nums[mid] == target) {
    			return mid;
    		}else if(nums[mid] < nums[right]) { // �м�ֵ���������ұߵ���С(�ұ�����)
    			// �м�ֵ��Ŀ�����ұ���Ŀ�������������ұߵ���С
    			if (nums[mid] < target && target <= nums[right]) {
    				left = mid + 1;
    			}else {
    				right = mid - 1;
    			}
    		}else { // �м�ֵ���������ұߵ�����(�������)
    			// �м�ֵ��Ŀ�����ұ���Ŀ��������������ߵ�����
    			if (nums[mid] > target && nums[left] <= target) {
    				right = mid - 1;
    			}else {
    				left = mid + 1;
    			}
    		}
    	}
    	return -1;
    }
    
    /*
     * 120. ��������С·����
     */
    public int minimumTotal(List<List<Integer>> triangle) {
    	int size = triangle.size();
    	// dp[i][j]��ʾ�ߵ���λ�ö�Ӧ����С·����
    	int[][] dp = new int[size][size];
    	dp[0][0] = triangle.get(0).get(0);
    	for (int i = 1; i < size; i++) {
    		// j = 0 ʱ j-1 ������
    		// ���ϵ�ǰλ�õ�ֵ
			dp[i][0] = dp[i - 1][0] + triangle.get(i).get(0);
			for (int j = 1; j < i; j++) {
				// ȡ���Ϻ����ϵ���Сֵ
				dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle.get(i).get(j);
			}
			// j = i ʱ j ������
			dp[i][i] = dp[i - 1][i - 1] + triangle.get(i).get(i);
		}
    	// ���һ��ÿ��λ�þ��Ǹ���·������С·����
    	int min = dp[size - 1][0];
    	for (int i = 1; i < size; i++) {
			min = Math.min(min, dp[size - 1][i]);
		}
    	return min;
    }

	/**
	 * 121. ������Ʊ�����ʱ��
	 */
	public int maxProfit1(int[] prices) {
		int min = prices[0];
		int max = 0;
		for (int i = 1; i < prices.length; i++) {
			// ÿ��ѭ��������Сֵ������һ��ѭ������Сֵ�����
			max = Math.max(max, prices[i] - min);
			min = Math.min(min, prices[i]);
		}
		return max;
	}

	public int maxProfit2(int[] prices) {
		if (prices.length <= 1){
			return 0;
		}
		// dp[i][0] ��ʾ��i�����(�ѹ���)��Ʊ��������ֽ�
		// dp[i][1] ��ʾ��i�첻����(����)��Ʊ��������ֽ�
		int[][] dp = new int[prices.length][2];
		// ����ʱ0Ԫ,�������Ϊ-prices[0]
		dp[0][0] = -prices[0];
		// û�������Ĺ�Ʊ
		dp[0][1] = 0;
		for (int i = 1; i < prices.length; i++){
			// �����Ʊ��ʣ�µ��ֽ�,Խ��Խ��
			dp[i][0] = Math.max(dp[i - 1][0], -prices[i]);
			// ������Ʊ��ʣ�µ��ֽ�,Խ��Խ��
			dp[i][1] = Math.max(dp[i - 1][1], prices[i] + dp[i - 1][0]);
		}
		return dp[prices.length - 1][1];
	}

	public int maxProfit3(int[] prices) {
		if (prices.length <= 1){
			return 0;
		}
		// dp[0] ��ʾ��i�����(�ѹ���)��Ʊ��������ֽ�
		// dp[1] ��ʾ��i�첻����(����)��Ʊ��������ֽ�
		int[] dp = new int[2];
		// ����ʱ0Ԫ,�������Ϊ-prices[0]
		dp[0] = -prices[0];
		// ����û�������Ĺ�Ʊ
		dp[1] = 0;
		// ������Ҫ�õ�ǰһ��,��Χ[1, prices.length]
		for (int i = 1; i <= prices.length; i++){
			// ǰһ�����,��������
			dp[0] = Math.max(dp[0], -prices[i - 1]);
			// ǰһ������,��������, ����Ҫ����,��ǰһ����в���
			dp[1] = Math.max(dp[1], prices[i - 1] + dp[0]);
		}
		return dp[1];
	}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
}
