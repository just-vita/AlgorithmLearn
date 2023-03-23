package top.vita.dp;

import java.util.List;

public class DPQuestion {

	public static void main(String[] args) {

	}

	/*
	 * 509. 斐波那契数
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
	 * 70. 爬楼梯
	 */
    public int climbStairs(int n) {
    	if (n <= 1) {
    		return n;
    	}
    	int[] dp = new int[n + 1];
    	// dp[0] 无意义，不作初始化
    	dp[1] = 1;
    	dp[2] = 2;
    	for (int i = 3; i <= n; i++) {
			dp[i] = dp[i - 1] + dp[i - 2];
		}
    	return dp[n];
    }
    
    /*
     * 198. 打家劫舍
     */
    public int rob(int[] nums) {
    	int[] dp = new int[nums.length];
    	dp[0] = nums[0];
    	for (int i = 1; i < nums.length; i++) {
    		// 防止错误用例
    		if (i == 1) {
    			dp[1] = Math.max(nums[0], nums[1]);
    		}else {
    			dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
    		}
		}
    	return dp[nums.length - 1];
    }
    
    /*
     * 33. 搜索旋转排序数组
     */
    public int search(int[] nums, int target) {
    	int left = 0;
    	int right = nums.length - 1;
    	while (left <= right) {
    		int mid = left + (right - left) / 2;
    		if (nums[mid] == target) {
    			return mid;
    		}else if(nums[mid] < nums[right]) { // 中间值比数组最右边的数小(右边有序)
    			// 中间值在目标数右边且目标数比数组最右边的数小
    			if (nums[mid] < target && target <= nums[right]) {
    				left = mid + 1;
    			}else {
    				right = mid - 1;
    			}
    		}else { // 中间值比数组最右边的数大(左边有序)
    			// 中间值在目标数右边且目标数比数组最左边的数大
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
     * 120. 三角形最小路径和
     */
    public int minimumTotal(List<List<Integer>> triangle) {
    	int size = triangle.size();
    	// dp[i][j]表示走到此位置对应的最小路径和
    	int[][] dp = new int[size][size];
    	dp[0][0] = triangle.get(0).get(0);
    	for (int i = 1; i < size; i++) {
    		// j = 0 时 j-1 无意义
    		// 加上当前位置的值
			dp[i][0] = dp[i - 1][0] + triangle.get(i).get(0);
			for (int j = 1; j < i; j++) {
				// 取左上和正上的最小值
				dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle.get(i).get(j);
			}
			// j = i 时 j 无意义
			dp[i][i] = dp[i - 1][i - 1] + triangle.get(i).get(i);
		}
    	// 最后一行每个位置就是各自路径的最小路径和
    	int min = dp[size - 1][0];
    	for (int i = 1; i < size; i++) {
			min = Math.min(min, dp[size - 1][i]);
		}
    	return min;
    }

	/**
	 * 121. 买卖股票的最佳时机
	 */
	public int maxProfit1(int[] prices) {
		int min = prices[0];
		int max = 0;
		for (int i = 1; i < prices.length; i++) {
			// 每次循环计算最小值，用上一次循环的最小值计算差
			max = Math.max(max, prices[i] - min);
			min = Math.min(min, prices[i]);
		}
		return max;
	}

	public int maxProfit2(int[] prices) {
		if (prices.length <= 1){
			return 0;
		}
		// dp[i][0] 表示第i天持有(已购买)股票所得最多现金
		// dp[i][1] 表示第i天不持有(卖出)股票所得最多现金
		int[][] dp = new int[prices.length][2];
		// 买入时0元,所得最多为-prices[0]
		dp[0][0] = -prices[0];
		// 没有能卖的股票
		dp[0][1] = 0;
		for (int i = 1; i < prices.length; i++){
			// 购买股票后剩下的现金,越多越好
			dp[i][0] = Math.max(dp[i - 1][0], -prices[i]);
			// 卖出股票后剩下的现金,越多越好
			dp[i][1] = Math.max(dp[i - 1][1], prices[i] + dp[i - 1][0]);
		}
		return dp[prices.length - 1][1];
	}

	public int maxProfit3(int[] prices) {
		if (prices.length <= 1){
			return 0;
		}
		// dp[0] 表示第i天持有(已购买)股票所得最多现金
		// dp[1] 表示第i天不持有(卖出)股票所得最多现金
		int[] dp = new int[2];
		// 买入时0元,所得最多为-prices[0]
		dp[0] = -prices[0];
		// 当天没有能卖的股票
		dp[1] = 0;
		// 计算需要用到前一天,范围[1, prices.length]
		for (int i = 1; i <= prices.length; i++){
			// 前一天持有,或当天买入
			dp[0] = Math.max(dp[0], -prices[i - 1]);
			// 前一天卖出,或当天卖出, 当天要卖出,得前一天持有才行
			dp[1] = Math.max(dp[1], prices[i - 1] + dp[0]);
		}
		return dp[1];
	}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
}
