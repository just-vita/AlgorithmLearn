package top.vita.dp;

import java.util.Arrays;
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

	/**
	 * 122. ������Ʊ�����ʱ�� II
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
	 * 123. ������Ʊ�����ʱ�� III
	 */
	public int maxProfit6(int[] prices) {
		if (prices.length <= 1){
			return 0;
		}
		// 0 û�в���
		// 1 ��һ�γ��й�Ʊ
		// 2 ��һ�β����й�Ʊ
		// 3 �ڶ��γ��й�Ʊ
		// 4 �ڶ��β����й�Ʊ
		int[][] dp = new int[prices.length][5];
		// ����
		dp[0][1] = -prices[0];
		// ���롢������������
		dp[0][3] = -prices[0];

		for (int i = 1; i < prices.length; i++){
			// ��һ�������
			dp[i][1] = Math.max(dp[i - 1][1], dp[i][0] - prices[i]);
			// ��һ��������
			dp[i][2] = Math.max(dp[i - 1][2], dp[i][1] + prices[i]);
			// �ڶ��������
			dp[i][3] = Math.max(dp[i - 1][3], dp[i][2] - prices[i]);
			// �ڶ���������
			dp[i][4] = Math.max(dp[i - 1][4], dp[i][3] + prices[i]);
		}
		return dp[prices.length - 1][4];
	}

	public int maxProfit7(int[] prices) {
		if (prices.length <= 1){
			return 0;
		}
		// 1 ��һ�γ��й�Ʊ
		// 2 ��һ�β����й�Ʊ
		// 3 �ڶ��γ��й�Ʊ
		// 4 �ڶ��β����й�Ʊ
		int[] dp = new int[5];
		// ����
		dp[1] = -prices[0];
		// ���롢������������
		dp[3] = -prices[0];

		for (int i = 1; i < prices.length; i++){
			// ��һ�������
			dp[1] = Math.max(dp[1], -prices[i]);
			// ��һ��������
			dp[2] = Math.max(dp[2], dp[1] + prices[i]);
			// �ڶ��������
			dp[3] = Math.max(dp[3], dp[2] - prices[i]);
			// �ڶ���������
			dp[4] = Math.max(dp[4], dp[3] + prices[i]);
		}
		return dp[4];
	}

	/**
	 * 188. ������Ʊ�����ʱ�� IV
	 */
	public int maxProfit8(int k, int[] prices) {
		if (prices.length <= 1){
			return 0;
		}
		int n = prices.length;
		// ÿ�ν��׶������롢��������״̬������Ҫ��2
		int[][] dp = new int[n][2 * k + 1];
		// ����Ϊ���� ż��Ϊ����
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
	 * 300. �����������
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
				// ��αȽ�dp[j]
				if (nums[j] < nums[i]) {
					dp[i] = Math.max(dp[i], dp[j] + 1);
				}
			}
			// ��¼���ֵ����ֹdp[n - 1]�������ֵ
			if (res < dp[i]){
				res = dp[i];
			}
		}
		return res;
	}

	/**
	 * 674. �������������
	 */
	public int findLengthOfLCIS(int[] nums) {
		int n = nums.length;
		if (n <= 1){
			return n;
		}
		int[] dp  = new int[n];
		Arrays.fill(dp, 1);
		// ����������������һ��
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
		// ����������������һ��
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
	 * 718. ��ظ�������
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
	 * ����д��
	 */
	public int findLength2(int[] nums1, int[] nums2) {
		int n1 = nums1.length;
		int n2 = nums2.length;
		// ���±�i-1Ϊ��β��A�������±�j-1Ϊ��β��B����ظ������鳤��Ϊdp[i][j]
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
	 * ����д��
	 */
	public int findLength4(int[] nums1, int[] nums2) {
		int n1 = nums1.length;
		int n2 = nums2.length;
		// ���±�iΪ��β��A�������±�jΪ��β��B����ظ������鳤��Ϊdp[i - 1][j - 1]
		int[][] dp = new int[n1][n2];
		// ���������Ѿ���ͬ������ʼ��
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
					// ����ͬ
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
	 * 1143. �����������
	 */
	public int longestCommonSubsequence(String text1, String text2) {
		int[][] dp = new int[text1.length() + 1][text2.length() + 1];

		for (int i = 1; i <= text1.length(); i++){
			// ֱ�Ӵ�i - 1��j - 1��ʼ��ʡȥ��ʼ������
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
	 * 1035. ���ཻ����
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
	 * 53. ����������
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
     * 392. �ж�������
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
                    // ��Ƚ���
                    // 1143.����������� �������ַ���������ɾԪ��
                    // ������ֻ��Ҫɾ��t�ַ�����Ԫ��
                    // ���Բ����ʱ�ĵ��ƹ�ʽ��
                    // Math.max(dp[i - 1][j], dp[i][j - 1]) ����ɾ�ı��ַ�����
                    // ��Ϊ�� dp[i][j - 1] ֻ��ɾt�ַ�����
                    dp[i][j] = dp[i][j - 1];
                }
            }
        }
        return dp[s.length()][t.length()] == s.length();
    }

    /**
     * 115. ��ͬ��������
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
	 * 583. �����ַ�����ɾ������
	 */
	public int minDistance1(String word1, String word2) {
		int n = word1.length();
		int m = word2.length();
		// dp[i-1, j-1] ��Ҫ�õ��������Ҫɾ�������ٴ���������dp[0][0]���Ա�ʾΪ�մ�
		int[][] dp = new int[n + 1][m + 1];
		// ��Ϊ���ַ���ʱ����һ������Ҫɾ�� i ��
		for (int i = 0; i <= n; i++) {
			dp[i][0] = i;
		}
		// ��Ϊ���ַ���ʱ����һ������Ҫɾ�� j ��
		for (int j = 0; j <= m; j++) {
			dp[0][j] = j;
		}
		for (int i = 1; i <= n; i++){
			for (int j = 1; j <= m; j++) {
				if (word1.charAt(i - 1) == word2.charAt(j - 1)){
					// ��ȣ���ɾ
					dp[i][j] = dp[i - 1][j - 1];
				}else {
					// ���1�����߶�ɾ������+2
					// ���2��ɾword1������+1
					// ���3��ɾword2������+1
					dp[i][j] = Math.min(dp[i - 1][j - 1] + 2, Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1));
				}
			}
		}
		return dp[n][m];
	}

	public int minDistance2(String word1, String word2) {
		int n = word1.length();
		int m = word2.length();
		// dp[i-1, j-1] ��Ҫ�õ��������Ҫɾ�������ٴ���������dp[0][0]���Ա�ʾΪ�մ�
		// ����������нⷨ
		int[][] dp = new int[n + 1][m + 1];
		for (int i = 1; i <= n; i++){
			for (int j = 1; j <= m; j++) {
				if (word1.charAt(i - 1) == word2.charAt(j - 1)){
					// �ҵ�һ��������
					dp[i][j] = dp[i - 1][j - 1] + 1;
				}else {
					dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
				}
			}
		}
		// �����ַ����ĳ��ȼ�ȥ����������еĳ��Ⱦ�����Ҫɾ�������ٸ�����
		return n + m - dp[n][m] * 2;
	}

	/**
	 * 72. �༭����
	 */
	public int minDistance(String word1, String word2) {
		int n = word1.length();
		int m = word2.length();
		int[][] dp = new int[n + 1][m + 1];
		// ���Ϊ���ַ�����������һ���ַ�����Ҫ�༭�Ĳ���Ϊ i
		for (int i = 0; i <= n; i++) {
			dp[i][0] = i;
		}
		// ���Ϊ���ַ�����������һ���ַ�����Ҫ�༭�Ĳ���Ϊ j
		for (int j = 0; j <= m; j++) {
			dp[0][j] = j;
		}
		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= m; j++) {
				if (word1.charAt(i - 1) == word2.charAt(j - 1)){
					// ������ñ༭
					dp[i][j] = dp[i - 1][j - 1];
				} else {
					// ����������ֲ������
					// �����ַ� ����һ���ַ��൱����һ���ַ���ɾ��һ���ַ�������ֱ�ӿ��Բ�����ӣ�����ɾ��
					// �滻�ַ� dp[i - 1][j - 1] + 1
					// ɾ���ַ� ѡ��ɾword1����word2 dp[i - 1][j], dp[i][j - 1]
					dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
				}
			}
		}
		return dp[n][m];
	}

	/**
	 * 647. �����Ӵ�
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
			// ��iΪ����
			res += getPalindrome(s, i, i);
			// ��i + 1Ϊ����
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
		// ��Ҫʹ�õ����޸ĵ����ݣ�����Ҫʹ�����ֱ���˳��
		for (int j = 0; j < n; j++) {
			for (int i = 0; i <= j; i++) {
				if (s.charAt(j) == s.charAt(i)){
					// ������ȵ��ַ��ǻ��Ĵ��������ַ�Ҳ�ǻ��Ĵ�
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
	 * 516. �����������
	 */
	public int longestPalindromeSubseq(String s) {
		int n = s.length();
		int[][] dp = new int[n + 1][n];
		// �������i����֤ǰ������ݱ������
		for (int i = n - 1; i >= 0; i--) {
			// ��i��j��ͬʱ������ָ����ͬһ�ַ����������г��ȳ�ʼ��Ϊ1
			dp[i][i] = 1;
			// jָ����iָ�����
			for (int j = i + 1; j < n; j++) {
				if (s.charAt(i) == s.charAt(j)){
					// ָ��ָ��������ַ�����ͬ������������������Ӵ�
					// ��ȡ���м俿һ�������Ӵ��ĸ���
					dp[i][j] = dp[i + 1][j - 1] + 2;
				} else{
					// ��������ַ�����ͬ����ֿ��ж��ı߻����Ӵ������ΪĿǰ����Ӵ�����
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
