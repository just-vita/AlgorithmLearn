package top.vita.dp;

import java.util.Arrays;
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

	/**
	 * 122. 买卖股票的最佳时机 II
	 */
	public int maxProfit4(int[] prices) {
		if (prices.length <= 1){
			return 0;
		}
		int[][] dp = new int[prices.length][2];
		dp[0][0] = -prices[0];
		dp[0][1] = 0;
		for (int i = 1; i < prices.length; i++){
			dp[i - 1][0] = Math.max(dp[i - 1][0], dp[i - 1][1] - prices[i - 1]);
			dp[i - 1][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i - 1]);
		}
		return dp[prices.length - 1][1];
	}

	public int maxProfit5(int[] prices) {
		if (prices.length <= 1){
			return 0;
		}
		int[] dp = new int[2];
		dp[0] = -prices[0];
		dp[1] = 0;
		for (int i = 1; i <= prices.length; i++){
			dp[0] = Math.max(dp[0], dp[1] - prices[i - 1]);
			dp[1] = Math.max(dp[1], dp[0] + prices[i - 1]);
		}
		return dp[1];
	}

	/**
	 * 123. 买卖股票的最佳时机 III
	 */
	public int maxProfit6(int[] prices) {
		if (prices.length <= 1){
			return 0;
		}
		// 0 没有操作
		// 1 第一次持有股票
		// 2 第一次不持有股票
		// 3 第二次持有股票
		// 4 第二次不持有股票
		int[][] dp = new int[prices.length][5];
		// 买入
		dp[0][1] = -prices[0];
		// 买入、卖出后，再买入
		dp[0][3] = -prices[0];

		for (int i = 1; i < prices.length; i++){
			// 第一次买或不买
			dp[i][1] = Math.max(dp[i - 1][1], dp[i][0] - prices[i]);
			// 第一次卖或不卖
			dp[i][2] = Math.max(dp[i - 1][2], dp[i][1] + prices[i]);
			// 第二次买或不买
			dp[i][3] = Math.max(dp[i - 1][3], dp[i][2] - prices[i]);
			// 第二次卖或不卖
			dp[i][4] = Math.max(dp[i - 1][4], dp[i][3] + prices[i]);
		}
		return dp[prices.length - 1][4];
	}

	public int maxProfit7(int[] prices) {
		if (prices.length <= 1){
			return 0;
		}
		// 1 第一次持有股票
		// 2 第一次不持有股票
		// 3 第二次持有股票
		// 4 第二次不持有股票
		int[] dp = new int[5];
		// 买入
		dp[1] = -prices[0];
		// 买入、卖出后，再买入
		dp[3] = -prices[0];

		for (int i = 1; i < prices.length; i++){
			// 第一次买或不买
			dp[1] = Math.max(dp[1], -prices[i]);
			// 第一次卖或不卖
			dp[2] = Math.max(dp[2], dp[1] + prices[i]);
			// 第二次买或不买
			dp[3] = Math.max(dp[3], dp[2] - prices[i]);
			// 第二次卖或不卖
			dp[4] = Math.max(dp[4], dp[3] + prices[i]);
		}
		return dp[4];
	}

	/**
	 * 188. 买卖股票的最佳时机 IV
	 */
	public int maxProfit8(int k, int[] prices) {
		if (prices.length <= 1){
			return 0;
		}
		int n = prices.length;
		// 每次交易都有买入、卖出两个状态，所以要乘2
		int[][] dp = new int[n][2 * k + 1];
		// 奇数为购入 偶数为卖出
		for (int j = 1; j < 2 * k; j += 2){
			dp[0][j] = -prices[0];
		}
		for (int i = 1; i < n; i++){
			for (int j = 0; j < 2 * k - 1; j += 2){
				dp[i][j + 1] = Math.max(dp[i - 1][j + 1], dp[i - 1][j] - prices[i]);
				dp[i][j + 2] = Math.max(dp[i - 1][j + 2], dp[i - 1][j + 1] + prices[i]);
			}
		}
		return dp[n - 1][2 * k];
	}

	/**
	 * 300. 最长递增子序列
	 */
	public int lengthOfLIS(int[] nums) {
		int n = nums.length;
		if (n <= 1){
			return n;
		}
		int[] dp = new int[n];
		Arrays.fill(dp, 1);
		int res = 0;
		for (int i = 1; i < n; i++) {
			for (int j = 0; j < i; j++) {
				// 多次比较dp[j]
				if (nums[j] < nums[i]) {
					dp[i] = Math.max(dp[i], dp[j] + 1);
				}
			}
			// 记录最大值，防止dp[n - 1]不是最大值
			if (res < dp[i]){
				res = dp[i];
			}
		}
		return res;
	}

	/**
	 * 674. 最长连续递增序列
	 */
	public int findLengthOfLCIS(int[] nums) {
		int n = nums.length;
		if (n <= 1){
			return n;
		}
		int[] dp  = new int[n];
		Arrays.fill(dp, 1);
		// 连续子序列最少是一个
		int res = 1;
		for (int i = 1; i < n; i++) {
			if (nums[i] > nums[i - 1]){
				dp[i] = dp[i - 1] + 1;
			}
			if (dp[i] > res){
				res = dp[i];
			}
		}
		return res;
	}

	public int findLengthOfLCIS2(int[] nums) {
		int n = nums.length;
		if (n <= 1){
			return n;
		}
		// 连续子序列最少是一个
		int res = 1;
		int count = 1;
		for (int i = 1; i < n; i++) {
			if (nums[i] > nums[i - 1]){
				count++;
			}else{
				count = 1;
			}
			if (count > res){
				res = count;
			}
		}
		return res;
	}

	/**
	 * 718. 最长重复子数组
	 */
	public int findLength(int[] nums1, int[] nums2) {
		int res = 0;
		for (int i = 0; i < nums1.length; i++){
			for (int j = 0; j < nums2.length; j++){
				int k = 0;
				while (i + k < nums1.length && j + k < nums2.length && nums1[i + k] == nums2[j + k]){
					k++;
				}
				if (res < k){
					res = k;
				}
			}
		}
		return res;
	}

	/**
	 * 巧妙写法
	 */
	public int findLength2(int[] nums1, int[] nums2) {
		int n1 = nums1.length;
		int n2 = nums2.length;
		// 以下标i-1为结尾的A，和以下标j-1为结尾的B，最长重复子数组长度为dp[i][j]
		int[][] dp = new int[n1 + 1][n2 + 1];
		int res = 0;
		for (int i = 1; i <= n1; i++) {
			for (int j = 1; j <= n2; j++) {
				if (nums1[i - 1] == nums2[j - 1]){
					dp[i][j] = dp[i - 1][j - 1] + 1;
				}
				if (res < dp[i][j]){
					res = dp[i][j];
				}
			}
		}
		return res;
	}

	/**
	 * 正常写法
	 */
	public int findLength4(int[] nums1, int[] nums2) {
		int n1 = nums1.length;
		int n2 = nums2.length;
		// 以下标i为结尾的A，和以下标j为结尾的B，最长重复子数组长度为dp[i - 1][j - 1]
		int[][] dp = new int[n1][n2];
		// 给行列上已经相同的数初始化
		for (int i = 0; i < n1; i++) {
			if (nums1[i] == nums2[0]){
				dp[i][0] = 1;
			}
		}
		for (int j = 0; j < n1; j++) {
			if (nums1[0] == nums2[j]){
				dp[0][j] = 1;
			}
		}
		int res = 0;
		for (int i = 0; i < n1; i++) {
			for (int j = 0; j < n2; j++) {
				if (nums1[i] == nums2[j] && i > 0 && j > 0){
					dp[i][j] = dp[i - 1][j - 1] + 1;
				}
				if (res < dp[i][j]){
					res = dp[i][j];
				}
			}
		}
		return res;
	}

	public int findLength3(int[] nums1, int[] nums2) {
		int n1 = nums1.length;
		int n2 = nums2.length;
		int[] dp = new int[n1 + 1];
		int res = 0;
		for (int i = 1; i <= n1; i++) {
			for (int j = n2; j >= 1; j--) {
				if (nums1[i - 1] == nums2[j - 1]){
					dp[j] = dp[j - 1] + 1;
				}else {
					// 不相同
					dp[j] = 0;
				}
				if (res < dp[j]){
					res = dp[j];
				}
			}
		}
		return res;
	}

	/**
	 * 1143. 最长公共子序列
	 */
	public int longestCommonSubsequence(String text1, String text2) {
		int[][] dp = new int[text1.length() + 1][text2.length() + 1];

		for (int i = 1; i <= text1.length(); i++){
			// 直接从i - 1和j - 1开始，省去初始化操作
			char ch1 = text1.charAt(i - 1);
			for (int j = 1; j <= text2.length(); j++){
				char ch2 = text2.charAt(j - 1);
				if (ch1 == ch2){
					dp[i][j] = dp[i - 1][j - 1] + 1;
				}else{
					dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
				}
			}
		}
		return dp[text1.length()][text2.length()];
	}

	/**
	 * 1035. 不相交的线
	 */
	public int maxUncrossedLines(int[] nums1, int[] nums2) {
		int[][] dp = new int[nums1.length + 1][nums2.length + 1];
		for(int i = 1; i <= nums1.length; i++){
			for(int j = 1; j <= nums2.length; j++){
				if(nums1[i - 1] == nums2[j - 1]){
					dp[i][j] = dp[i - 1][j - 1] + 1;
				}else{
					dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
				}
			}
		}
		return dp[nums1.length][nums2.length];
	}

	/**
	 * 53. 最大子数组和
	 */
	public int maxSubArray(int[] nums) {
		if (nums.length == 1){
			return nums[0];
		}
		int[] dp = new int[nums.length];
		dp[0] = nums[0];
		int res = dp[0];
		for (int i = 1; i < nums.length; i++){
			dp[i] = Math.max(dp[i - 1] + nums[i], nums[i]);
			if (res < dp[i]){
				res = dp[i];
			}
		}
		return res;
	}

    /**
     * 392. 判断子序列
     */
    public boolean isSubsequence(String s, String t) {
        int[][] dp = new int[s.length() + 1][t.length() + 1];
        for (int i = 1; i <= s.length(); i++){
            char ch1 = s.charAt(i - 1);
            for (int j = 1; j <= t.length(); j++){
                char ch2 = t.charAt(j - 1);
                if (ch1 == ch2){
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else{
                    // 相比较下
                    // 1143.最长公共子序列 是两个字符串都可以删元素
                    // 而这题只需要删除t字符串的元素
                    // 所以不相等时的递推公式从
                    // Math.max(dp[i - 1][j], dp[i][j - 1]) 考虑删哪边字符串的
                    // 变为了 dp[i][j - 1] 只用删t字符串的
                    dp[i][j] = dp[i][j - 1];
                }
            }
        }
        return dp[s.length()][t.length()] == s.length();
    }

    /**
     * 115. 不同的子序列
     */
    public int numDistinct(String s, String t) {
        int n = s.length();
        int m = t.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 0; i <= n; i++){
            dp[i][0] = 1;
        }
        for (int i = 1; i <= n; i++){
            char ch1 = s.charAt(i - 1);
            for (int j = 1; j <= m; j++){
                char ch2 = t.charAt(j - 1);
                if (ch1 == ch2){
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                } else{
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[n][m];
    }

	/**
	 * 583. 两个字符串的删除操作
	 */
	public int minDistance1(String word1, String word2) {
		int n = word1.length();
		int m = word2.length();
		// dp[i-1, j-1] 想要得到相等所需要删除的最少次数，这样dp[0][0]可以表示为空串
		int[][] dp = new int[n + 1][m + 1];
		// 作为空字符串时，另一边最少要删除 i 个
		for (int i = 0; i <= n; i++) {
			dp[i][0] = i;
		}
		// 作为空字符串时，另一边最少要删除 j 个
		for (int j = 0; j <= m; j++) {
			dp[0][j] = j;
		}
		for (int i = 1; i <= n; i++){
			for (int j = 1; j <= m; j++) {
				if (word1.charAt(i - 1) == word2.charAt(j - 1)){
					// 相等，不删
					dp[i][j] = dp[i - 1][j - 1];
				}else {
					// 情况1：两边都删，次数+2
					// 情况2：删word1，次数+1
					// 情况3：删word2，次数+1
					dp[i][j] = Math.min(dp[i - 1][j - 1] + 2, Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1));
				}
			}
		}
		return dp[n][m];
	}

	public int minDistance2(String word1, String word2) {
		int n = word1.length();
		int m = word2.length();
		// dp[i-1, j-1] 想要得到相等所需要删除的最少次数，这样dp[0][0]可以表示为空串
		// 最长公共子序列解法
		int[][] dp = new int[n + 1][m + 1];
		for (int i = 1; i <= n; i++){
			for (int j = 1; j <= m; j++) {
				if (word1.charAt(i - 1) == word2.charAt(j - 1)){
					// 找到一个子序列
					dp[i][j] = dp[i - 1][j - 1] + 1;
				}else {
					dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
				}
			}
		}
		// 两个字符串的长度减去最长公共子序列的长度就是需要删除的最少个数了
		return n + m - dp[n][m] * 2;
	}

	/**
	 * 72. 编辑距离
	 */
	public int minDistance(String word1, String word2) {
		int n = word1.length();
		int m = word2.length();
		int[][] dp = new int[n + 1][m + 1];
		// 如果为空字符串，则变成另一个字符串需要编辑的步数为 i
		for (int i = 0; i <= n; i++) {
			dp[i][0] = i;
		}
		// 如果为空字符串，则变成另一个字符串需要编辑的步数为 j
		for (int j = 0; j <= m; j++) {
			dp[0][j] = j;
		}
		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= m; j++) {
				if (word1.charAt(i - 1) == word2.charAt(j - 1)){
					// 相等则不用编辑
					dp[i][j] = dp[i - 1][j - 1];
				} else {
					// 不相等有三种操作情况
					// 增加字符 增加一个字符相当于另一个字符串删除一个字符，所以直接可以不算添加，交给删除
					// 替换字符 dp[i - 1][j - 1] + 1
					// 删除字符 选择删word1还是word2 dp[i - 1][j], dp[i][j - 1]
					dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
				}
			}
		}
		return dp[n][m];
	}

	/**
	 * 647. 回文子串
	 */
	public int countSubstrings(String s) {
		int[] dp = new int[s.length()];
		Arrays.fill(dp, 1);
		for (int i = 1; i < s.length(); i++) {
			for (int j = 0; j <= i; j++) {
				if (s.charAt(i) != s.charAt(j)){
					continue;
				}
				if (isPalindrome(s, i, j)){
					dp[i] = Math.max(dp[i], dp[i - 1]) + 1;
				}
			}
		}
		return dp[s.length() - 1];
	}

	private boolean isPalindrome(String s, int i, int j) {
		while (j < i && s.charAt(i) == s.charAt(j)){
			i--;
			j++;
		}
		return true;
	}

	public int countSubstrings1(String s) {
		int res = 0;
		for (int i = 0; i < s.length(); i++) {
			// 以i为中心
			res += getPalindrome(s, i, i);
			// 以i + 1为中心
			res += getPalindrome(s, i, i + 1);
		}
		return res;
	}

	private int getPalindrome(String s, int i, int j) {
		int res = 0;
		while (i >= 0 && j <s.length() && s.charAt(i) == s.charAt(j)){
			i--;
			j++;
			res++;
		}
		return res;
	}

	public int countSubstrings2(String s) {
		int n = s.length();
		boolean[][] dp = new boolean[n][n];
		// 需要使用到已修改的数据，所以要使用这种遍历顺序
		for (int j = 0; j < n; j++) {
			for (int i = 0; i <= j; i++) {
				if (s.charAt(j) == s.charAt(i)){
					// 两个相等的字符是回文串，单个字符也是回文串
					if (j - i < 3){
						dp[i][j] = true;
					}else{
						dp[i][j] = dp[i + 1][j - 1];
					}
				}
			}
		}
		int res = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (dp[i][j]){
					res++;
				}
			}
		}		return res;
	}

	/**
	 * 516. 最长回文子序列
	 */
	public int longestPalindromeSubseq(String s) {
		int n = s.length();
		int[][] dp = new int[n + 1][n];
		// 倒序遍历i，保证前面的数据被计算过
		for (int i = n - 1; i >= 0; i--) {
			// 当i和j相同时，代表指向了同一字符，则将子序列长度初始化为1
			dp[i][i] = 1;
			// j指针在i指针后面
			for (int j = i + 1; j < n; j++) {
				if (s.charAt(i) == s.charAt(j)){
					// 指针指向的两个字符都相同，代表多了两个回文子串
					// 获取往中间靠一格后回文子串的个数
					dp[i][j] = dp[i + 1][j - 1] + 2;
				} else{
					// 如果两个字符不相同，则分开判断哪边回文子串最长，作为目前的最长子串长度
					dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
				}
			}
		}
		return dp[0][n - 1];
	}

	public int fib2(int n) {
		if (n == 0) {
			return 0;
		}
		int[] dp = new int[n + 1];
		dp[1] = 1;
		for (int i = 2; i <= n; i++) {
			dp[i] = dp[i - 1] + dp[i - 2];
			dp[i] %= 1000000007;
		}
		return dp[n];
	}

	public int maxValue(int[][] grid) {
		for (int i = 0; i < grid.length; i++) {
			for (int j = 0; j < grid[0].length; j++) {
				if (i == 0 && j == 0) {
					continue;
				}
				if (i == 0) {
					grid[i][j] = grid[i][j] + grid[i][j - 1];
				} else if (j == 0) {
					grid[i][j] = grid[i][j] + grid[i - 1][j];
				} else {
					grid[i][j] =  grid[i][j] + Math.max(grid[i][j - 1], grid[i - 1][j]);
				}
			}
		}
		return grid[grid.length - 1][grid[0].length - 1];
	}













































































































}
