package top.vita.array;

import java.util.*;

@SuppressWarnings("all")
public class ArrayQuestion {

    public static void main(String[] args) {
//		int[] nums1 = {1,2,3,0,0,0};
//		int m = 3;
//		int[] nums2 = {2,5,6};
//		int n = 3;
//		merge(nums1,m,nums2,n);

//		int[] arr = new int[]{3,2,6,5,0,3};
//		System.out.println(maxProfit(arr));

//		int[] arr = new int[]{1,3,5};
//		System.out.println(searchInsert(arr, 4));

//		generate(5);

//		System.out.println(-1 / 2);


//		int[][] matrix = {{1,1,1},{1,0,1},{1,1,1}};
//		setZeroes(matrix);

//		int[] arr = {0,0,3,4};
//		int[] twoSum = twoSum(arr,0);
//		System.out.println(Arrays.toString(twoSum));

//		checkInclusion("ab","eidbaooo");


    }

    /*
     * 217. 存在重复元素
     * 给你一个整数数组 nums 。如果任一值在数组中出现 至少两次 ，
     * 返回 true ；如果数组中每个元素互不相同，返回 false
     */
    public boolean containsDuplicate(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i = 0; i < nums.length; i++) {
            if (map.get(nums[i]) != null) {
                if (map.get(nums[i]) + 1 > 1) return true;
                map.put(nums[i], map.get(nums[i]) + 1);
            } else {
                map.put(nums[i], 1);
            }
        }
        return false;
    }

    /*
     * 53. 最大子数组和
     *  给你一个整数数组 nums ，
     *  请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），
     *  返回其最大和。
     */
    public int maxSubArray(int[] nums) {
        int maxSum = nums[0];
        int curSum = nums[0];
        for (int i = 0; i < nums.length; i++) {
            if (curSum < 0) {
                curSum = 0;
            }
            curSum += nums[i];
            maxSum = Math.max(maxSum, curSum);
        }
        return maxSum;
    }

    /*
     * 2351. 第一个出现两次的字母
     *  给你一个由小写英文字母组成的字符串 s ，
     *  请你找出并返回第一个出现 两次 的字母。
     */
    public char repeatedCharacter(String s) {
        HashMap<Character, Integer> map = new HashMap<>();
        for (char ch : s.toCharArray()) {
            if (map.get(ch) != null) {
                if (map.get(ch) + 1 > 1) return ch;
                map.put(ch, map.get(ch) + 1);
            } else {
                map.put(ch, 1);
            }
        }
        return 'a';
    }

    /*
     * 88. 合并两个有序数组
     * 给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，
     * 另有两个整数 m 和 n ，分别表示 nums1 和
     * nums2 中的元素数目。
     */
    public static void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1;
        int j = n - 1;
        for (int k = m + n - 1; k >= 0; k--) {
            // 当nums2遍历完后，也会走这个if
            if (j < 0 || (i >= 0 && nums1[i] >= nums2[j])) {
                nums1[k] = nums1[i];
                i--;
            } else {
                nums1[k] = nums2[j];
                j--;
            }
        }
    }

    /*
     * 350. 两个数组的交集 II
     * 给你两个整数数组 nums1 和 nums2
     * 请你以数组形式返回两数组的交集。返回结果中每个元素出现的次数，
     * 应与元素在两个数组中都出现的次数一致（如果出现次数不一致，则考虑取较小值）。
     * 可以不考虑输出结果的顺序。
     */
    public int[] intersect(int[] nums1, int[] nums2) {
//    	HashMap<Integer,Integer> map = new HashMap<Integer,Integer>();
//    	
//		for (int i : nums1) {
//			if (map.containsKey(i)) {
//				map.put(i, map.get(i) + 1);
//			} else {
//				map.put(i, 1);
//			}
//		}
//    	ArrayList<Integer> res = new ArrayList<Integer>();
//		for (int i : nums2) {
//			Integer num = map.get(i);
//			if (num != null && num > 0) {
//				res.add(i);
//				map.put(i,map.get(i) - 1);
//			}
//		}
//		int[] arr = new int[res.size()];
//		for (int i = 0; i < res.size(); i++) {
//			arr[i] = res.get(i);
//		}
//		return arr;
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        /*
         * [4,5,9]
         * [4,4,8,9,9]
         */
        int i = 0;
        int j = 0;
        ArrayList<Integer> res = new ArrayList<Integer>();
        for (int k = 0; k < nums1.length; k++) {
            if (i > nums1.length - 1 || j > nums2.length - 1) break;

            if (nums1[i] == nums2[j]) {
                res.add(nums1[i]);
                i++;
                j++;
            } else if (nums1[i] > nums2[j]) {
                j++;
            } else {
                i++;
            }
        }
        int[] arr = new int[res.size()];
        for (int k = 0; k < res.size(); k++) {
            arr[k] = res.get(k);
        }
        return arr;
    }

    /*
     * 121. 买卖股票的最佳时机
     * 给定一个数组 prices ，
     * 它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
     */
    public static int maxProfit(int[] prices) {
        int min = prices[0];
        int max = 0;
        for (int i = 1; i < prices.length; i++) {
            // 每次循环计算最小值，用上一次循环的最小值计算差
            max = Math.max(max, prices[i] - min);
            min = Math.min(min, prices[i]);
        }
        return max;
    }

    /*
     * 278. 第一个错误的版本
     * 你是产品经理，目前正在带领一个团队开发新的产品。
     * 不幸的是，你的产品的最新版本没有通过质量检测。由于每个版本都是基于之前的版本开发的，
     * 所以错误的版本之后的所有版本都是错的。
     */
    public int firstBadVersion(int n) {
        int left = 1;
        int right = n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (isBadVersion(mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    boolean isBadVersion(int version) {
        return true;
    }

    ;

    /*
     * 35. 搜索插入位置
     * 给定一个排序数组和一个目标值，
     * 在数组中找到目标值，并返回其索引。
     * 如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
     */
    public static int searchInsert(int[] nums, int target) {
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == target) {
                return i;
            }
            if (nums[i] < target && i + 1 < nums.length && nums[i + 1] > target) {
                return i + 1;
            }
            if (nums[i] > target) {
                return i;
            }
        }
        return nums.length;
    }

    /*
     * 118. 杨辉三角
     * 给定一个非负整数 numRows，
     * 生成「杨辉三角」的前 numRows 行。
     * 在「杨辉三角」中，每个数是它左上方和右上方的数的和。
     */
    public static List<List<Integer>> generate(int numRows) {
        ArrayList<List<Integer>> res = new ArrayList<List<Integer>>();
        for (int i = 0; i < numRows; i++) {
            ArrayList<Integer> row = new ArrayList<Integer>();
            for (int j = 0; j <= i; j++) {
                // 加入作为边的 1
                if (j == 0 || j == i) {
                    row.add(1);
                } else {
                    // 上一层的 j - 1 的值
                    Integer beforeJ = res.get(i - 1).get(j - 1);
                    // 上一层的 j 的值
                    Integer afterJ = res.get(i - 1).get(j);
                    // 相加得到当前位置的值
                    row.add(beforeJ + afterJ);
                }
            }
            res.add(row);
        }
        return res;
    }

    /*
     * 189. 轮转数组
     * 给你一个数组，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
     */
    public void rotate(int[] nums, int k) {
        if (nums.length < 2) return;
        // 对 k 取余，k 大于数组长度时也能翻转
        k %= nums.length;
        // 整个数组翻转
        reverse(nums, 0, nums.length - 1);
        // 翻转前 k 个
        reverse(nums, 0, k - 1);
        // 翻转后 k 个
        reverse(nums, k, nums.length - 1);
    }

    /*
     * 36. 有效的数独
     * 请你判断一个 9 x 9 的数独是否有效。只需要 根据以下规则 ，
     * 验证已经填入的数字是否有效即可。
     */
    public boolean isValidSudoku(char[][] board) {
        boolean[][] row = new boolean[9][9];
        boolean[][] col = new boolean[9][9];
        boolean[][] block = new boolean[9][9];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    int num = board[i][j] - '1';
                    // 获取3x3九宫格下标
                    int blockIndex = i / 3 * 3 + j / 3;
                    // 当前行、列、3x3九宫格已经存在这个数字
                    if (row[i][num] || col[j][num] || block[blockIndex][num]) {
                        return false;
                    } else { // 标记为此数字存在
                        row[i][num] = true;
                        col[j][num] = true;
                        block[blockIndex][num] = true;
                    }

                }
            }
        }
        return true;
    }

    /*
     * 73. 矩阵置零
     * 给定一个 m x n 的矩阵，如果一个元素为 0 ，
     * 则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
     */
    public static void setZeroes(int[][] matrix) {
        // 判断首行是否有0
        boolean rowflag = false;
        for (int i = 0; i < matrix[0].length; i++) {
            if (matrix[0][i] == 0) {
                rowflag = true;
                break;
            }
        }
        // 判断首列是否有0
        boolean colflag = false;
        for (int i = 0; i < matrix.length; i++) {
            if (matrix[i][0] == 0) {
                colflag = true;
                break;
            }
        }
        // 判断除首行首列外是否有0，有则将0加到当前的首行首列上
        for (int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[0].length; j++) {
                if (matrix[i][j] == 0) {
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }
        // 判断首行是否有0，有则将此<列>全部设为0
        for (int i = 1; i < matrix[0].length; i++) {
            if (matrix[0][i] == 0) {
                for (int j = 1; j < matrix.length; j++) {
                    matrix[j][i] = 0;
                }
            }
        }
        // 判断首列是否有0，有则将此<行>全部设为0
        for (int i = 1; i < matrix.length; i++) {
            if (matrix[i][0] == 0) {
                for (int j = 1; j < matrix[0].length; j++) {
                    matrix[i][j] = 0;
                }
            }
        }
        // 首行有0，则将首行全部设为0
        if (rowflag) {
            for (int i = 0; i < matrix[0].length; i++) {
                matrix[0][i] = 0;
            }
        }
        // 首列有0，则将首列全部设为0
        if (colflag) {
            for (int i = 0; i < matrix.length; i++) {
                matrix[i][0] = 0;
            }
        }
    }

    public void moveZeroes(int[] nums) {
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
     * 给你一个下标从 1 开始的整数数组 numbers ，
     * 该数组已按 非递减顺序排列 ，请你从数组中找出满足相加之和等于目标数 target 的两个数。
     */
    public static int[] twoSum(int[] numbers, int target) {
        /*
         * 二分
         */
        int left = 0;
        int right = numbers.length - 1;
        while (left < right) {
            int mid = (left + (right - left)) / 2;
            if (numbers[mid] >= target && mid - 1 > 0) {
                right = mid - 1;
            } else {
                if (numbers[left] + numbers[right] > target) {
                    right--;
                } else if (numbers[left] + numbers[right] < target) {
                    left++;
                } else {
                    return new int[]{left + 1, right + 1};
                }
            }
        }
        return new int[2];
    	/* 双指针
    	for (int i = 0,j = numbers.length - 1; i < j;) {
			int sum = numbers[i] + numbers[j];
			if (sum == target) {
				return new int[] {i + 1,j + 1};
			}else if(sum > target) {
				j--;
			}else {
				i++;
			}
		}
    	return null;
    	*/
    }

    /*
     * 56. 合并区间
     */
    public int[][] merge(int[][] intervals) {
        // 按第一位数升序排序
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
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
                out.get(out.size() - 1)[1] = Math.max(intervals[i][1], out.get(out.size() - 1)[1]);
            }
        }

        return out.toArray(new int[out.size()][]);
    }

    /*
     * 658. 找到 K 个最接近的元素
     */
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        int left = 0;
        int right = arr.length - 1;
        int removeCount = arr.length - k;
        while (removeCount > 0) {
            // 绝对值的表示方式
            if (x - arr[left] <= arr[right] - x) {
                right--;
            } else {
                left++;
            }
            removeCount--;
        }

        List<Integer> res = new ArrayList<>();
        for (int i = left; i < left + k; i++) {
            res.add(arr[i]);
        }

        return res;
    }

    /*
     * 1672. 最富有客户的资产总量
     */
    public int maximumWealth(int[][] accounts) {
        TreeSet s = new TreeSet();
        for (int i = 0; i < accounts.length; i++) {
            s.add(Arrays.stream(accounts[i]).sum());
        }
        return (int) s.last();
    }

    /*
     * 1929. 数组串联
     */
    public int[] getConcatenation(int[] nums) {
        int length = nums.length;
        int[] res = Arrays.copyOf(nums, length << 1);
        System.arraycopy(nums, 0, res, 0, length);
        System.arraycopy(nums, 0, res, length, length);
        System.gc();
        return res;
    }

    /*
     * 1464. 数组中两元素的最大乘积
     */
    public int maxProduct(int[] nums) {
        int max = 0;
        int index = 0;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > max) {
                max = nums[i];
                index = i;
            }
        }
        int max2 = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i != index && nums[i] > max2) {
                max2 = nums[i];
            }
        }
        return (max - 1) * (max2 - 1);
    }

    /*
     * 567. 字符串的排列
     */
    public static boolean checkInclusion(String s1, String s2) {
        int n = s1.length();
        int m = s2.length();
        if (n > m) {
            return false;
        }
        int[] cur = new int[26];
        int[] window = new int[26];
        // 创建窗口0——n-1
        for (int i = 0; i < n; i++) {
            cur[s1.charAt(i) - 'a']++;
            window[s2.charAt(i) - 'a']++;
        }
        if (Arrays.equals(cur, window)) {
            return true;
        }
        // 窗口从0——n-1开始
        for (int i = n; i < m; i++) {
            // 窗口移动 0——n
            window[s2.charAt(i) - 'a']++;
            // 移出窗口 1——n
            window[s2.charAt(i - n) - 'a']--;
            if (Arrays.equals(cur, window)) {
                return true;
            }
        }
        return false;
    }

    /*
     * 77. 组合
     */
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> path = new ArrayList<Integer>();
        int startIndex = 1;
        backtracking(n, k, startIndex, res, path);
        return res;
    }

    private void backtracking(int n, int k, int startIndex, List<List<Integer>> res, List<Integer> path) {
        if (path.size() == k) {
            res.add(new ArrayList<>(path));
            return;
        }
        // 剪枝
        // k - path.size(): 还需要加入多少个元素
        // 最少要从 n - (k - path.size() 开始遍历才能获得符合个数为k的结果
        for (int i = startIndex; i <= n - (k - path.size()) + 1; i++) {
            path.add(i);
            backtracking(n, k, i + 1, res, path);
            path.remove(path.size() - 1);
        }
    }

    /*
     * 784. 字母大小写全排列
     */
    public List<String> letterCasePermutation(String s) {
        List<String> res = new ArrayList<String>();
        // 将s转成StringBuilder作为路径
        backtracking(0, s.toCharArray(), res, new StringBuilder(s));
        return res;
    }

    private void backtracking(int startIndex, char[] chs, List<String> res, StringBuilder path) {
        if (startIndex == chs.length) {
            res.add(path.toString());
            return;
        }

        // 初始化当前字符
        path.setCharAt(startIndex, chs[startIndex]);
        // 数字直接加
        if (Character.isDigit(chs[startIndex])) {
            backtracking(startIndex + 1, chs, res, path);
        } else {
            // 修改为大写
            path.setCharAt(startIndex, Character.toUpperCase(chs[startIndex]));
            backtracking(startIndex + 1, chs, res, path);
            // 修改为小写
            path.setCharAt(startIndex, Character.toLowerCase(chs[startIndex]));
            backtracking(startIndex + 1, chs, res, path);
        }
    }

    /*
     * 78. 子集
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        res.add(new ArrayList<Integer>());
        backtracking(0, res, new ArrayList<Integer>(), nums);
        return res;
    }

    private void backtracking(int startIndex, List<List<Integer>> res, ArrayList<Integer> path, int[] nums) {
        if (startIndex == nums.length) {
            res.add(new ArrayList<Integer>(path));
            return;
        }
        path.add(nums[startIndex]);
        // 考虑选择当前位置
        backtracking(startIndex + 1, res, path, nums);
        path.remove(path.size() - 1);
        // 考虑不选择当前位置
        backtracking(startIndex + 1, res, path, nums);
    }

    /*
     * 74. 搜索二维矩阵
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int row = 0;
        int col = matrix[0].length - 1;
        while (row < matrix.length && col >= 0) {
            if (matrix[row][col] > target) {
                col--;
            } else if (matrix[row][col] < target) {
                row++;
            } else {
                return true;
            }
        }
        return false;
    }

    /*
     * 946. 验证栈序列
     */
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Deque<Integer> que = new ArrayDeque<Integer>();
        int j = 0;
        for (int i = 0; i < popped.length; i++) {
            que.push(pushed[i]);
            while (!que.isEmpty() && popped[j] == que.peek()) {
                j++;
                que.poll();
            }
        }
        return que.isEmpty();
    }

    /*
     * 153. 寻找旋转排序数组中的最小值 3,4,5,1,2
     */
    public int findMin(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        int min = Integer.MAX_VALUE;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < min) {
                min = nums[mid];
            }
            // 右边有序,最小值在左边
            if (nums[mid] <= nums[right]) {
                right = mid - 1;
            } else if (nums[mid] > nums[right]) { // 左边有序,最小值在右边
                left = mid + 1;
            }
        }
        return min;
    }

    /*
     * 162. 寻找峰值
     */
    public int findPeakElement(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    /*
     * 374. 猜数字大小
     */
    public int guessNumber(int n) {
        int left = 0;
        int right = n;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (guess(mid) == -1) {
                right = mid - 1;
            } else if (guess(mid) == 1) {
                left = mid + 1;
            } else {
                return mid;
            }
        }
        return -1;
    }

    int guess(int num) {
        return 0;
    }

    /*
     * 852. 山脉数组的峰顶索引 [0,2,1,0]
     */
    public int peakIndexInMountainArray(int[] arr) {
        // 数组最小长度为3，且 0 和 arr.length - 1绝对不会是最大值
        int left = 1;
        int right = arr.length - 2;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] < arr[mid + 1]) {
                left = mid + 1; // 更新后， left 的左侧的前一个值(mid)小于后一个值(mid + 1)
            } else {
                right = mid; // 更新后， right 的右侧的前一个值(mid)大于后一个值(mid + 1)
            }
        }
        // 循环不变量，left/right都指向最大值
        return right;
    }

    /*
     * 844. 比较含退格的字符串
     */
    public boolean backspaceCompare(String s, String t) {
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        char[] chs1 = s.toCharArray();
        char[] chs2 = t.toCharArray();
        for (int i = 0; i < chs1.length; i++) {
            if (chs1[i] != '#') {
                sb1.append(chs1[i]);
            } else if (sb1.length() > 0) {
                sb1.deleteCharAt(sb1.length() - 1);
            }
        }
        for (int i = 0; i < chs2.length; i++) {
            if (chs2[i] != '#') {
                sb2.append(chs2[i]);
            } else if (sb2.length() > 0) {
                sb2.deleteCharAt(sb2.length() - 1);
            }
        }
        return sb1.toString().equals(sb2.toString());
    }

    /*
     * 367. 有效的完全平方数
     */
    public boolean isPerfectSquare(int num) {
        int left = 0;
        int right = num;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            long square = (long) mid * mid;
            if (square > num) {
                right = mid - 1;
            } else if (square < num) {
                left = mid + 1;
            } else {
                return true;
            }
        }
        return false;
    }

    /*
     * 1385. 两个数组间的距离值
     */
    public int findTheDistanceValue(int[] arr1, int[] arr2, int d) {
        Arrays.sort(arr2);
        int res = 0;
        for (int i = 0; i < arr1.length; i++) {
            int low = arr1[i] - d;
            int high = arr1[i] + d;
            if (!binarySearch(arr2, low, high)) {
                res++;
            }
        }
        return res;
    }

    private boolean binarySearch(int[] arr, int low, int high) {
        int left = 0;
        int right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            // num − d <= arr[mid] <= num + d
            if (arr[mid] >= low && arr[mid] <= high) {
                return true;
            } else if (arr[mid] < low) {
                left = mid + 1;
            } else if (arr[mid] > high) {
                right = mid - 1;
            }
        }
        return false;
    }

    /*
     * 646. 最长数对链
     */
    public int findLongestChain(int[][] pairs) {
        Arrays.sort(pairs, (o1, o2) -> o1[1] - o2[1]);
        int res = 1;
        int temp = pairs[0][1];
        for (int i = 1; i < pairs.length; i++) {
            if (pairs[i][0] > temp) {
                res++;
                temp = pairs[i][1];
            }
        }
        return res;
    }

    /*
     * 1582. 二进制矩阵中的特殊位置
     */
    public int numSpecial(int[][] mat) {
        int res = 0;
        int[] row = new int[mat.length];
        int[] col = new int[mat[0].length];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                if (mat[i][j] == 1) {
                    row[i] += 1;
                    col[j] += 1;
                }
            }
        }
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                if (mat[i][j] == 1 && row[i] == 1 && col[j] == 1) {
                    res++;
                }
            }
        }
        return res;
    }

    /*
     * 986. 区间列表的交集
     */
    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        List<int[]> res = new ArrayList<int[]>();
        int i = 0;
        int j = 0;
        while (i < firstList.length && j < secondList.length) {
            int[] temp = new int[2];
            int a = firstList[i][0];
            int b = firstList[i][1];
            int x = secondList[j][0];
            int y = secondList[j][1];
            // [1,3],[2,3] temp => [2,3]
            temp[0] = Math.max(a, x);
            temp[1] = Math.min(b, y);
            if (temp[1] >= temp[0]) {
                res.add(temp);
            }
            if (y >= b) {
                i++;
            } else {
                j++;
            }
        }
        return res.toArray(new int[res.size()][]);
    }

    /*
     * 11. 盛最多水的容器
     */
    public int maxArea(int[] height) {
        int max = 0;
        int slow = 0;
        int fast = height.length - 1;
        while (slow <= fast) {
            max = Math.max(Math.min(height[slow], height[fast]) * (fast - slow), max);
            if (height[slow] <= height[fast]) {
                slow++;
            } else {
                fast--;
            }
        }
        return max;
    }

    /*
     * 438. 找到字符串中所有字母异位词
     */
    public static List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<Integer>();
        int[] target = new int[26];
        for (int i = 0; i < p.length(); i++) {
            target[p.charAt(i) - 'a']++;
        }
        int slow = 0;
        int fast = 0;
        int[] window = new int[26];
        while (fast < s.length()) {
            window[s.charAt(fast) - 'a']++;
            // 窗口长度达到p
            if (fast - slow + 1 == p.length()) {
                // 比较集合
                if (Arrays.equals(target, window)) {
                    res.add(slow);
                }
                // 窗口缩小，窗口数值减少
                window[s.charAt(slow) - 'a']--;
                // 长度到达p时开始缩小窗口
                slow++;
            }
            fast++;
        }
        return res;
    }

    /*
     * 713. 乘积小于 K 的子数组
     */
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k == 0 || k == 1) return 0;
        int res = 0;
        int slow = 0;
        int fast = 0;
        int temp = 1;
        while (fast < nums.length) {
            temp *= nums[fast];
            // 大于k则去掉窗口最左侧的数
            while (temp >= k) {
                temp /= nums[slow];
                slow++;
            }
            // 加上当前窗口中的数字个数
            res += fast - slow + 1;
            fast++;
        }
        return res;
    }

    /*
     * 547. 省份数量
     */
    int n;
    int[] father;

    public int findCircleNum(int[][] isConnected) {
        n = isConnected.length;
        father = new int[n];
        // 初始化并查集
        for (int i = 0; i < father.length; i++) {
            father[i] = i;
        }

        for (int i = 0; i < isConnected.length; i++) {
            for (int j = 0; j < isConnected[0].length; j++) {
                if (isConnected[i][j] == 1) {
                    union(i, j);
                }
            }
        }
        HashSet<Integer> res = new HashSet<>();
        for (int i = 0; i < father.length; i++) {
            // 看看有几个顶点
            res.add(find(i));
        }
        return res.size();
    }

    int find(int u) {
        if (u == father[u]) {
            return u;
        }
        father[u] = find(father[u]);
        return father[u];
    }

    void union(int u, int v) {
        u = find(u);
        v = find(v);
        if (u == v) {
            return;
        }
        father[v] = u;
    }

    /*
     * 1592. 重新排列单词间的空格
     */
    public String reorderSpaces(String text) {
        if (text.length() <= 1) {
            return text;
        }
        int space = 0;
        int word = 0;
        boolean isWord = false;
        for (int i = 0; i < text.length(); i++) {
            if (text.charAt(i) == ' ') {
                space++;
                isWord = false;
            } else if (!isWord) {
                word++;
                isWord = true;
            }
        }

        int spaceNum = space / (word - 1 == 0 ? 1 : word - 1);
        StringBuilder res = new StringBuilder();
        String[] split = text.split(" ");
        for (int i = 0; i < split.length; i++) {
            if (split[i] != "") {
                res.append(split[i]);
                boolean hasWord = false;
                for (int j = i + 1; j < split.length; j++) {
                    if (split[i] != "") {
                        hasWord = true;
                    }
                }
                if (hasWord) {
                    for (int j = 0; j < spaceNum; j++) {
                        res.append(" ");
                    }
                }
            }
        }
        while (res.length() < text.length()) {
            res.append(" ");
        }
        return res.toString();
    }

    /*
     * 1539. 第 k 个缺失的正整数
     */
    public int findKthPositive(int[] arr, int k) {
        int left = 0;
        int right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            // 数组的中位数并不是原本的中位数，当前数与原本的中位数的差大于k则找到
            // 减1是因为索引从0开始
            if (arr[mid] - mid - 1 < k) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left + k;
    }

    /**
     * 27. 移除元素
     */
    public int removeElement(int[] nums, int val) {
        int n = nums.length;
        int slow = 0;
        for (int fast = 0; fast < n; fast++) {
            if (nums[fast] != val) {
                // 每次都赋值是因为 在找到目标值后slow指针不会变，fast指针会将后面的数值赋值到slow指针的位置上
                nums[slow] = nums[fast];
                slow++;
            }
        }
        return slow;
    }

    /**
     * 977. 有序数组的平方
     */
    public int[] sortedSquares(int[] nums) {
        int[] res = new int[nums.length];
        int i = 0;
        int j = nums.length - 1;
        int k = res.length - 1;
        while (i <= j) {
            if (nums[i] * nums[i] < nums[j] * nums[j]) {
                res[k] = nums[j] * nums[j];
                j--;
            } else {
                res[k] = nums[i] * nums[i];
                i++;
            }
            k--;
        }
        return res;
    }

    /**
     * 209. 长度最小的子数组
     */
    public int minSubArrayLen(int target, int[] nums) {
        int left = 0;
        int right = 0;
        int sum = 0;
        int res = Integer.MAX_VALUE;
        for (right = 0; right < nums.length; right++) {
            sum += nums[right];
            while (sum >= target) {
                res = Math.min(res, right - left + 1);
                sum -= nums[left];
                left++;
            }
        }

        return res == Integer.MAX_VALUE ? 0 : res;
    }

    /**
     * 59. 螺旋矩阵 II
     */
    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        int num = 1;
        int loop = 0;
        int start = 0;
        int i, j;
        while (loop++ < n / 2) {
            for (j = start; j < n - loop; j++) {
                res[start][j] = num++;
            }
            for (i = start; i < n - loop; i++) {
                res[i][j] = num++;
            }

            for (; j >= loop; j--) {
                res[i][j] = num++;
            }
            for (; i >= loop; i--) {
                res[i][j] = num++;
            }
            start++;
        }

        if (n % 2 == 1) {
            res[start][start] = num;
        }

        return res;
    }

    public int[] exchange(int[] nums) {
        int[] res = new int[nums.length];
        int left = 0;
        int right = res.length - 1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] % 2 == 1) {
                res[left++] = nums[i];
            } else {
                res[right--] = nums[i];
            }
        }
        return res;
    }

    public int[] twoSum1(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int sum = nums[left] + nums[right];
            if (sum > target) {
                right--;
            } else if (sum < target) {
                left++;
            } else {
                return new int[]{nums[left], nums[right]};
            }
        }
        return new int[0];
    }

    public int[] spiralOrder(int[][] matrix) {
        int row = matrix.length;
        if (row == 0) {
            return new int[0];
        }
        int col = matrix[0].length;
        int[] res = new int[row * col];
        int left = 0;
        int right = col - 1;
        int top = 0;
        int bottom = row - 1;
        int index = 0;
        while (true) {
            // 从左往右
            for (int i = left; i <= right; i++) {
                res[index++] = matrix[top][i];
            }
            if (++top > bottom) {
                break;
            }
            // 从上往下
            for (int i = top; i <= bottom; i++) {
                res[index++] = matrix[i][right];
            }
            if (--right < left) {
                break;
            }
            // 从右往左
            for (int i = right; i >= left; i--) {
                res[index++] = matrix[bottom][i];
            }
            if (--bottom < top) {
                break;
            }
            // 从下往上
            for (int i = bottom; i >= top; i--) {
                res[index++] = matrix[i][left];
            }
            if (++left > right) {
                break;
            }
        }
        return res;
    }

    public int findRepeatNumber(int[] nums) {
        int i = 0;
        while (i < nums.length) {
            // 已经将数字和下标下的数字交换过，进行下一个数字的操作
            if (nums[i] == i) {
                i++;
                continue;
            }
            // 根据题意 nums[i] < nums.length，也就是每个数字都会对应一个下标
            // 如果下标下已经有数字了，那么就找到了一个重复的数
            if (nums[nums[i]] == nums[i]) {
                return nums[i];
            }
            // 交换数字与下标下的数字，实现下标对应数字
            int temp = nums[i];
            nums[i] = nums[temp];
            nums[temp] = temp;
        }
        return -1;
    }

    public int search(int[] nums, int target) {
        if (nums.length == 0) {
            return 0;
        }
        int left = 0;
        int right = nums.length - 1;
        int count = 0;
        while (left <= right) {
            int mid = left + ((right - left) / 2);
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                int l = mid - 1;
                int r = mid;
                while (l >= 0 && nums[l--] == target) {
                    count++;
                }
                while (r < nums.length && nums[r++] == target) {
                    count++;
                }
                return count;
            }
        }
        return 0;
    }

    public int missingNumber(int[] nums) {
        int i;
        for (i = 0; i < nums.length; i++) {
            if (i != nums[i]) {
                return i;
            }
        }
        return i;
    }

    public int missingNumber1(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = (right + (right - left)) / 2;
            if (mid == nums[mid]) {
                left = mid + 1;
            } else if (mid < nums[mid]) {
                // 只有小于的情况
                right = mid - 1;
            }
        }
        return left;
    }

    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix.length < 1) {
            return false;
        }
        int i = 0;
        int j = matrix[0].length - 1;
        while (i < matrix.length && j >= 0) {
            if (target == matrix[i][j]) {
                return true;
            } else if (target < matrix[i][j]) {
                // 从同一层往前找
                j--;
            } else {
                // 从同一列往下找
                i++;
            }
        }
        return false;
    }

    public int minArray(int[] numbers) {
        int left = 0;
        int right = numbers.length - 1;
        while (left < right) {
            int mid = (right + left) / 2;
            if (numbers[mid] > numbers[right]) {
                // 中间数比右边大，那可以认为右边是有序的
                left = mid + 1;
            } else if (numbers[mid] < numbers[right]) {
                // 比右边小，那可以认为中间是有序的
                right = mid;
            } else {
                // 跟当前数相等，移动右指针
                right--;
            }
        }
        return numbers[left];
    }

    public int longestConsecutive(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        Arrays.sort(nums);
        int count = 1;
        int max = 1;
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] + 1 == nums[i + 1]) {
                count++;
            } else if (nums[i] == nums[i + 1]) {
                continue;
            } else {
                count = 1;
            }
            max = Math.max(max, count);
        }
        return max;
    }

    public void moveZeroes1(int[] nums) {
        int slow = 0;
        int fast = 0;
        while (fast < nums.length) {
            // 将有效的数字都放到慢指针的位置
            if (nums[fast] != 0) {
                nums[slow++] = nums[fast];
            }
            fast++;
        }
        // 循环结束后慢指针往后的都是0
        while (slow < nums.length) {
            nums[slow++] = 0;
        }
    }

    public int maxArea1(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int max = 0;
        while (left < right) {
            int area = Math.min(height[left], height[right]) * (right - left);
            max = Math.max(max, area);
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return max;
    }

    public int subarraySum(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int presum = 0;
        int res = 0;
        // 前缀和初始化，类似动规
        map.put(0, 1);
        for (int i = 0; i < nums.length; i++) {
            // 前缀和
            presum += nums[i];
            // 如果pre-k的结果被包含在map中，说明邻近的一个或几个数的和刚好等于k
            if (map.containsKey(presum - k)) {
                res += map.get(presum - k);
            }
            map.put(presum, map.getOrDefault(presum, 0) + 1);
        }
        return res;
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums.length == 0 || k == 0) {
            return new int[0];
        }
        Deque<Integer> queue = new LinkedList<>();
        // 滑动窗口长度为k，需要先达到k位数字时才能得到结果，所以减去k
        int[] res = new int[nums.length - k + 1];
        // 左窗口前移k-1位，方便在形成窗口前将数据加入queue中
        int left = 1 - k;
        int right = 0;
        while (right < nums.length) {
            // 队头元素（最大值）已经不在窗口中，将其从队列中移除
            if (left > 0 && queue.peekFirst() == nums[left - 1]) {
                queue.removeFirst();
            }
            // 删除比新数值小的元素，让队头永远是最大值，保持递减
            while (!queue.isEmpty() && queue.peekLast() < nums[right]) {
                queue.removeLast();
            }
            queue.addLast(nums[right]);
            if (left >= 0) {
                // res中加入的永远是窗口中的最大值
                res[left] = queue.peekFirst();
            }
            left++;
            right++;
        }
        return res;
    }

    public int maxAbsoluteSum(int[] nums) {
        int max = 0;
        int min = 0;
        int sum = 0;
        for (int num : nums) {
            sum += num;
            // 正数的累计和尽可能大，负数的累计和尽可能小
            if (sum > max) {
                max = sum;
            } else if (sum < min) {
                min = sum;
            }
        }
        // 返回 max - min 时，实际上就是在计算了数组中的一个子数组
        // 使得其中正数和最大、负数和最小，从而得到这个子数组的最大绝对和
        return max - min;
    }

    public int subtractProductAndSum(int n) {
        int sum = 0;
        int prod = 1;
        while (n != 0) {
            sum += n % 10;
            prod *= n % 10;
            n /= 10;
        }
        return prod - sum;
    }

    public int[][] merge1(int[][] intervals) {
        // 按第一位数升序排序
        Arrays.sort(intervals, (a, b) -> {
            return a[0] - b[0];
        });
        List<int[]> res = new ArrayList<>();
        // 直接将第一个区间加入
        res.add(intervals[0]);
        for (int i = 1; i < intervals.length; i++) {
            // 当前区间的最小值大于上一个区间的最大值，如[0, 1]和[2, 3]，不是重复区间
            if (intervals[i][0] > res.get(res.size() - 1)[1]) {
                res.add(intervals[i]);
            } else {
                // 否则是重复区间，直接将右区间改为两个区间的最大值
                res.get(res.size() - 1)[1] = Math.max(intervals[i][1], res.get(res.size() - 1)[1]);
            }
        }
        return res.toArray(new int[res.size()][]);
    }

    public void rotate1(int[] nums, int k) {
        if (nums.length < 2){
            return;
        }
        // 对 k 取余，k 大于数组长度时也能翻转
        k %= nums.length;
        // 整个数组翻转
        reverse(nums, 0, nums.length - 1);
        // 翻转前 k 个
        reverse(nums, 0, k - 1);
        // 翻转后 k 个
        reverse(nums, k, nums.length - 1);
    }

    private void reverse(int[] nums, int begin, int end) {
        while (begin < end) {
            int temp = nums[begin];
            nums[begin] = nums[end];
            nums[end] = temp;
            begin++;
            end--;
        }
    }

    public int diagonalSum(int[][] mat) {
        int res = 0;
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat.length; j++) {
                // 在实例1的情况下 i == j 能处理1 5 9 i + j 能处理3 5 7（对角线
                if (i == j || i + j == mat.length - 1) {
                    res += mat[i][j];
                }
            }
        }
        return res;
    }

    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        // 前缀积
        int[] res = new int[n];
        res[0] = 1;
        for (int i = 1; i < n; i++) {
            res[i] = res[i - 1] * nums[i - 1];
        }
        int right = 1;
        for (int i = n - 1; i >= 0; i--) {
            res[i] = res[i] * right;
            right *= nums[i];
        }
        return res;
    }

    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            // nums[i] != nums[nums[i] - 1] 当前位置不是它应该在的位置
            while (nums[i] > 0 && nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
                // 调整数组，将数字放到正确的位置上
                int temp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = temp;
            }
        }
        // 因为上面的循环已经把数字都尽量调整到了正确的位置
        // 所以可以直接从0开始遍历来找到第一个位置不正确的数字
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        // 最小的不正确的数不在[0, n]中，则返回n+1
        return n + 1;
    }
}
