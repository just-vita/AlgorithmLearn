package top.vita.zuo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Stack;

public class GreedyDemo {
	/*
	 * 807. ���ֳ��������
	 */
    public int maxIncreaseKeepingSkyline(int[][] grid) {
    	int n = grid.length;
    	int m = grid[0].length;
    	int[] row = new int[n];
    	int[] col = new int[m];
    	for (int i = 0; i < row.length; i++) {
			for (int j = 0; j < col.length; j++) {
				// ����������ж�Ӧ�����ֵ
				row[i] = Math.max(row[i], grid[i][j]);
				col[j] = Math.max(col[j], grid[i][j]);
			}
		}
    	int res = 0;
    	for (int i = 0; i < row.length; i++) {
			for (int j = 0; j < col.length; j++) {
				// �����е���Сֵ�͵�ǰֵ�Ĳ�ֵ���ó���������Ӷ���
				res += Math.min(row[i], col[j]) - grid[i][j];
			}
		}
    	return res;
    }
    
    /*
     * 1689. ʮ-����������������Ŀ
     */
    public int minPartitions(String n) {
        // �ַ����е������...
    	int res = 0;
    	for (int i = 0; i < n.length(); i++) {
			res = Math.max(res, n.charAt(i) - '0');
		}
        return res;
    }

    /*
     * 1221. �ָ�ƽ���ַ���
     */
    public int balancedStringSplit(String s) {
    	int LR = 0;
    	int res = 0;
    	for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == 'R') LR++;
			else LR--;
			if (LR == 0) res++;
		}
    	return res;
    }
    
    /*
     * 1877. ������������Ժ͵���Сֵ
     */
    public int minPairSum(int[] nums) {
        Arrays.sort(nums);
        int res = 0;
        for (int i = 0, j = nums.length - 1; i < j; i++, j--) 
        	// ��β���
            res = Math.max(res, nums[i] + nums[j]);
        return res;
    }
    
    
    
    /*
     * 502. IPO
     */
    public int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
    	// ��������Ĵ����
    	PriorityQueue<Node> maxProfit = new PriorityQueue<Node>(new Comparator<Node>() {
			@Override
			public int compare(Node o1, Node o2) {
				return o2.p - o1.p;
			}
		});
    	// �������ѵ�С����
    	PriorityQueue<Node> minCapital = new PriorityQueue<Node>(new Comparator<Node>() {
    		@Override
    		public int compare(Node o1, Node o2) {
    			return o1.c - o2.c;
    		}
    	});
    	// �Ƚ��ڵ�ȫ������С���� ����ס��
    	for (int i = 0; i < capital.length; i++) {
			minCapital.add(new Node(profits[i],capital[i]));
		}
    	// ����k��
    	for (int i = 0; i < k; i++) {
    		// ��С�������ó�����������ʽ��㹻�ķ������� ��������
			while (!minCapital.isEmpty() && minCapital.peek().c <= w) {
				maxProfit.add(minCapital.poll());
			}
			// û���ʽ������ֱ�ӷ���
			if (maxProfit.isEmpty()) {
				return w;
			}
			w += maxProfit.poll().p;
		}
    	
    	return w;
    }
    
    public static class Node{
    	public int p;
    	public int c;
		public Node(int p, int c) {
			this.p = p;
			this.c = c;
		}
    }
    
    /*
             * һ���������У���ʱ����ȡ����λ��
     */
    public static class MedianHolder {
    	
    	PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>((o1,o2)->{
    		return o2 - o1;
    	});
    	PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>((o1,o2)->{
    		return o1 - o2;
    	});
    	
		private void modifyTwoHeapsSize() {
			if (this.maxHeap.size() == this.minHeap.size() + 2) {
				this.minHeap.add(this.maxHeap.poll());
			}
			if (this.minHeap.size() == this.maxHeap.size() + 2) {
				this.maxHeap.add(this.minHeap.poll());
			}
		}
    	
    	public void addNumber(int num) {
    		if (maxHeap.size() == minHeap.size() + 2) {
    			maxHeap.add(num);
    		}else {
    			minHeap.add(num);
    		}
    		modifyTwoHeapsSize();
    	}
    	
	    public Integer getMedian() {
	    	int maxSize = maxHeap.size();
	    	int minSize = minHeap.size();
	    	if(maxSize + minSize == 0) {
	    		return 0;
	    	}
	    	Integer maxHeapHead = maxHeap.peek();
	    	Integer minHeapHead = minHeap.peek();
	    	if (((maxSize+ minSize) & 1) == 0){
	    		return (maxHeapHead + minHeapHead) / 2;
	    	}
	    	return maxHeapHead > minHeapHead ? maxHeapHead : minHeapHead;
	    }
    }
    
    /*
     * 08.12. �˻ʺ� 51. N �ʺ�
     */
    List<List<String>> res = new ArrayList<List<String>>();
    
    public List<List<String>> solveNQueens(int n) {
    	int[] record = new int[n];
    	// ��ʼ�����̣�Ĭ�϶��� .
        char[][] grid = new char[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				grid[i][j] = '.';
			}
		}
        process(0, record, grid, n);
        return res;
    }

	private void process(int i, int[] record, char[][] grid, int n) {
		// �Ѿ��ڷ���
		if (i == n) {
			// �������ÿһ�����̵�������������
			ArrayList<String> list = new ArrayList<String>();
			for (int j = 0; j < grid.length; j++) {
				list.add(new String(grid[j]));
			}
			res.add(list);
			return;
		}
		
		StringBuilder sb = new StringBuilder();
		for (int j = 0; j < n; j++) {
			if(isValid(record, i, j)) {
				// ��i�е�j�аڷŻʺ�
				record[i] = j;
				// ��ǻʺ�
				grid[i][j] = 'Q';
				// �ݹ���һ��
				process(i + 1, record, grid, n);
				// ������������Ӱ���´εݹ�
				grid[i][j] = '.';
			}
		}
	}

	private boolean isValid(int[] record, int i, int j) {
		for (int k = 0; k < i; k++) {
			// �С��С�б�߶��������
			if (j == record[k] || Math.abs(record[k] - j) == Math.abs(i - k)) {
				return false;
			}
		}
		return true;
	}
    
	/*
	 * 52. N�ʺ� II
	 */
    public int totalNQueens(int n) {
    	int[] record = new int[n];
    	return process1(0, record, n);
    }

	private int process1(int i, int[] record, int n) {
		// �ڷ����
		if (i == n) {
			// ���������һ
			return 1;
		}
		int res = 0;
		for (int j = 0; j < n; j++) {
			if (isValid(record, i, j)) {
				// ��i�е�j�аڷŻʺ�
				record[i] = j;
				// �ݹ���һ��
				res += process1(i + 1, record, n);
			}
		}
		
		return res;
	}
	
    /*
	 * 1222. ���Թ��������Ļʺ�
	 */
    public List<List<Integer>> queensAttacktheKing(int[][] queens, int[] king) {
    	// �ù��������ʺ�
    	List<List<Integer>> res = new ArrayList<List<Integer>>();
    	boolean[][] queenPos = new boolean[8][8];
    	// ��ǻʺ�λ��
    	for (int i = 0; i < queens.length; i++) {
			queenPos[queens[i][0]][queens[i][1]] = true;
		}
    	// ������������İ˸�����
		int[][] directions = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 }, { -1, -1 }, { 1, -1 }, { -1, 1 }, { 1, 1 } };
		for (int i = 0; i < directions.length; i++) {
			int x = king[0];
			int y = king[1];
			while(x >= 0 && y >= 0 && x < 8 && y < 8) {
				// �����ҵ��ʺ�
				if (queenPos[x][y]) {
					ArrayList<Integer> list = new ArrayList<Integer>();
					list.add(x);
					list.add(y);
					res.add(list);
					break;
				}else {
					// ������������
					x += directions[i][0];
					y += directions[i][1];
				}
			}
		}
    	return res;
    }

    /*
     * 455. �ַ�����
     */
    public int findContentChildren(int[] g, int[] s) {
    	// С������ιСθ��
    	Arrays.sort(g);
    	Arrays.sort(s);
    	int res = 0;
    	int index = 0;
    	
    	for (int i = 0; i < s.length && index < g.length; i++) {
    		if (s[i] >= g[index]) {
    			res++;
    			index++;
    		}
		}
    	return res;
    }
	
    public int findContentChildren2(int[] g, int[] s) {
    	// �������ι��θ��
        Arrays.sort(g);
        Arrays.sort(s);
        int count = 0;
        int start = s.length - 1;
        for (int index = g.length - 1; index >= 0; index--) {
            if(start >= 0 && g[index] <= s[start]) {
                start--;
                count++;
            }
        }
        return count;
    }
	
	/*
	 * 376. �ڶ�����
	 */
    public int wiggleMaxLength(int[] nums) {
    	// ��¼��ֵ
        int cur = 0;
        int pre = 0;
        // Ĭ�����ұ���һ����ֵ
        int res = 1;
        for (int i = 0; i < nums.length - 1; i++) {
        	cur = nums[i + 1] - nums[i];
        	// �ҵ���ֵ
			if ((cur > 0 && pre <= 0) || (cur < 0 && pre >= 0)) {
				res++;
				pre = cur;
			}
		}
        return res;
    }
	
	/*
	 * 122. ������Ʊ�����ʱ�� II
	 */
    public int maxProfit(int[] prices) {
    	int res = 0;
    	for (int i = 1; i < prices.length; i++) {
			res += Math.max(prices[i] - prices[i - 1], 0);
		}
    	return res;
    }
	
	/*
	 * 55. ��Ծ��Ϸ
	 */
    public boolean canJump(int[] nums) {
    	int cover = 0;
        if (nums.length == 1) return true;
    	for (int i = 0; i <= cover; i++) {
    		// �������Ӹ��ǵ����򣬼�i����˼�Ǽ����Ѿ��ƶ���λ��
			cover = Math.max(nums[i] + i, cover);
			// �ɵ����յ�
			if (cover >= nums.length - 1) {
				return true;
			}
		}
    	return false;
    }
	
	/*
	 * 45. ��Ծ��Ϸ II
	 */
    public int jump(int[] nums) {
    	int step = 0;
    	int curCover = 0;
    	int nextCover = 0;
    	for (int i = 0; i < nums.length; i++) {
			nextCover = Math.max(nextCover, nums[i] + i);
			// ���ﵱǰ����󸲸�λ��
			if (curCover == i) {
				if (curCover != nums.length - 1) {
					step++;
					curCover = nextCover;
					if (nextCover > nums.length - 1) {
						break;
					}
				}else {
					break;
				}
			}
		}
    	return step;
    }
    
    /*
     * 1005. K ��ȡ������󻯵������
     */
    public int largestSumAfterKNegations(int[] nums, int k) {
        Arrays.sort(nums);
    	for (int i = 0; i < nums.length && k > 0; i++) {
			if (nums[i] < 0) {
				nums[i] = -nums[i];
				k--;
			}
		}
    	if (k > 0) {
    		Arrays.sort(nums);
            if (k % 2 == 1){
                nums[0] = -nums[0];
            }
    	}
    	int sum = 0;
    	for (int i = 0; i < nums.length; i++) {
			sum += nums[i];
		}
    	return sum;
    }
    
    /*
     * 11. ʢ���ˮ������
     */
    public int maxArea(int[] height) {
    	int left = 0;
    	int right = height.length - 1;
    	int max = 0;
    	while(left < right) {
    		max = Math.max((right - left) * Math.min(height[left], height[right]), max);
    		if (height[left] > height[right]) {
    			right--;
    		}else {
    			left++;
    		}
		}
    	return max;
    }
    
    /*
     * 14. �����ǰ׺
     */
    public String longestCommonPrefix(String[] strs) {
    	String prefix = strs[0];
    	for (String s : strs) {
    		while (!s.startsWith(prefix)) {
    			if (prefix.length() == 0) {
    				return "";
    			}
    			prefix = prefix.substring(0, prefix.length() - 1);
    		}
		}
    	return prefix;
    }
    
    /*
     * 1450. �ڼȶ�ʱ������ҵ��ѧ������
     */
    public int busyStudent(int[] startTime, int[] endTime, int queryTime) {
    	int count = 0;
    	for (int i = 0; i < endTime.length; i++) {
			if (startTime[i] <= queryTime && endTime[i] >= queryTime) {
				count++;
			}
		}
    	return count;
    }
    
    /*
     * 08.06. ��ŵ������
     */
    public void hanota(List<Integer> A, List<Integer> B, List<Integer> C) {
    	// A �ƶ��� C ��Ҫ�õ� B
    	func(A,B,C,A.size());
    }

	private void func(List<Integer> A, List<Integer> B, List<Integer> C, int num) {
		if (num == 1) {
			C.add(A.remove(A.size() - 1));
			return;
		}else {
			func(A,C,B,num - 1);
            C.add(A.remove(A.size() - 1)); 
			func(B,A,C,num - 1);
		}
	}
    
    /*
     * 392. �ж�������
     */
    public boolean isSubsequence(String s, String t) {
    	int slow = 0;
    	char[] s_chs = s.toCharArray();
    	char[] t_chs = t.toCharArray();
    	int fast = 0;
    	while (slow < s.length() && fast < t.length()) {
    		if (s_chs[slow] == t_chs[fast]) {
    			slow++;
    		}
			fast++;
    	}
    	if (slow < s.length()) {
    		return false;
    	}
    	return true;
    }
    
    /*
     * 46. ȫ����
     */
    public List<List<Integer>> permute(int[] nums) {
    	List<List<Integer>> res = new ArrayList<List<Integer>>();
		ArrayList<Integer> path = new ArrayList<Integer>();
		boolean[] visited = new boolean[nums.length];
		process2(nums, path, res, visited);
		return res;
	}

	private void process2(int[] nums, List<Integer> path, List<List<Integer>> res, boolean[] visited) {
		if (path.size() == nums.length) {
			res.add(new ArrayList<Integer>(path));
			return;
		}
		for (int j = 0; j < nums.length; j++) {
			if (!visited[j]) {
				visited[j] = true;
				path.add(nums[j]);
				process2(nums, path, res, visited);
				path.remove(path.size() - 1);
				visited[j] = false;
			}
		}
	}

	/*
	 * 486. Ԥ��Ӯ��
	 */
    public int PredictTheWinner(int[] nums) {
    	return Math.max(f(nums,0,nums.length - 1), s(nums,0,nums.length - 1));
    }

	private int f(int[] nums, int L, int R) {
		if (L == R) {
			return nums[L];
		}
		return Math.max(nums[L] + s(nums, L + 1, R), nums[R] + s(nums, L, R - 1));
	}

	private int s(int[] nums, int L, int R) {
		if (L == R) {
			return 0;
		}
		return Math.max(f(nums, L + 1, R), f(nums, L, R - 1));
	}
    
	/*
	 * ����ջ
	 */
    public static void reverse(Stack<Integer> stack) {
    	if (stack.isEmpty()) {
    		return;
    	}
    	int i = f(stack);
    	reverse(stack);
    	stack.push(i);
    }

	private static int f(Stack<Integer> stack) {
		Integer res = stack.pop();
		if (stack.isEmpty()) {
			return res;
		}else {
			int last = f(stack);
			stack.push(res);
			return last;
		}
	}
	
	public static void main(String[] args) {
		Stack<Integer> stack = new Stack<Integer>();
		stack.push(1);
		stack.push(2);
		stack.push(3);
		reverse(stack);
		System.out.println(stack);
	}
    
    /*
     * 91. ���뷽�� ����ʱ��
     */
    public int numDecodings(String s) {
    	char[] chs = s.toCharArray();
    	return process3(chs,0);
    }

	private int process3(char[] chs, int i) {
		if (i == chs.length) {
			return 1;
		}
		if (chs[i] == '0') {
			return 0;
		}
		if (chs[i] == '1') {
			int res = process3(chs, i + 1);
			if (i + 1 < chs.length) {
				res += process3(chs, i + 2);
			}
			return res;
		}
		if (chs[i] == '2') {
			int res = process3(chs, i + 1);
			if (i + 1 < chs.length && (chs[i + 1] >= '0' && chs[i + 1] <= '6')) {
				res += process3(chs, i + 2);
			}
			return res;
		}
		// 3 ~ 9
		return process3(chs, i + 1);
	}
    
    /*
     * 200. ��������
     */
    public int numIslands(char[][] grid) {
    	int n = grid.length;
    	int m = grid[0].length;
    	int res = 0;
    	for (int i = 0; i < n; i++) {
    		for (int j = 0; j < m; j++) {
				if (grid[i][j] == '1') {
					res++;
					infect(grid,n,m,i,j);
				}
			}
		}
    	return res;
    }

	private void infect(char[][] grid, int n, int m, int i, int j) {
		if (i < 0 || i >= n || j < 0 || j >= m || grid[i][j] != '1') {
			return;
		}
		// ���Ϊ2���´η���ֱ�ӷ���
		grid[i][j] = '2';
		// �������ҵݹ��Ⱦ
		infect(grid, n, m, i - 1, j);
		infect(grid, n, m, i + 1, j);
		infect(grid, n, m, i, j - 1);
		infect(grid, n, m, i, j + 1);
	}
    
    /*
     * ��ָ Offer II 105. �����������
     */
	int max = 0;
	int cur = 0;
    public int maxAreaOfIsland(int[][] grid) {
    	int n = grid.length;
    	int m = grid[0].length;
    	for (int i = 0; i < n; i++) {
    		for (int j = 0; j < m; j++) {
    			if (grid[i][j] == 1) {
    				cur = 0;
    				infect1(grid,n,m,i,j);
    				max = Math.max(cur, max);
    			}
			}
		}
    	return max;
    }

	private void infect1(int[][] grid, int n, int m, int i, int j) {
		if (i < 0 || i >= n || j < 0 || j >= m || grid[i][j] == 1) {
			return;
		}
		if (grid[i][j] == 1) {
			cur++;
			grid[i][j] = 2;
		}
		// �������ҵݹ��Ⱦ
		infect1(grid, n, m, i - 1, j);
		infect1(grid, n, m, i + 1, j);
		infect1(grid, n, m, i, j - 1);
		infect1(grid, n, m, i, j + 1);
	}
    
    /*
     * 34. �����������в���Ԫ�صĵ�һ�������һ��λ��
     */
    public int[] searchRange(int[] nums, int target) {
    	int[] res = new int[2];
    	res[0] = -1;
    	res[1] = -1;
    	for (int i = 0; i < nums.length; i++) {
    		if (nums[i] == target) {
	    		if (res[0] == -1) {
	    			res[0] = i;
	    		}
    			res[1] = i;
    		}
		}
    	return res;
    }
    
    /*
     * 239. �����������ֵ
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
    	Deque<Integer> que = new LinkedList<Integer>();
    	int[] res = new int[nums.length - k + 1];
    	for (int i = 0; i < res.length; i++) {
    		while (nums[i] >= que.peekLast()) {
    			que.removeLast();
    		}
    		
		}
    	
    	return res;
    }
    
    
    
    
    
    
    
    
    
    
    
    
}
