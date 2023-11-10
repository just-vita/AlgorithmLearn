package top.vita.array;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static java.lang.Thread.sleep;

public class ArrayDemo {

    /*
     * 1351. 统计有序矩阵中的负数
     */
    @SuppressWarnings("all")
    public int countNegatives(int[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            int left = 0;
            int right = grid[0].length - 1;
            while (left < right){
                int mid = left + (right - left) / 2;
                if (grid[i][mid] >= 0){
                    left = mid + 1;
                }else {
                    right = mid;
                }
            }
            if (grid[i][left] < 0){
                // left的位置代表有几个大于0的数
                res += grid[0].length - left;
            }
        }
        return res;
    }

    /*
     * 74. 搜索二维矩阵
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int n = matrix.length;
        int m = matrix[0].length;
        int left = 0;
        int right = n * m - 1;
        while (left <= right){
            // 每一行衔接上一行的末尾，二分这个数组的下标
            int mid = left + (right - left) / 2;
            // 映射行和列
            int x = matrix[mid / m][mid % m];
            if (x < target){
                left = mid + 1;
            }else if (x > target){
                right = mid - 1;
            }else {
                return true;
            }
        }
        return false;
    }

    /*
     * 1337. 矩阵中战斗力最弱的 K 行
     */
    public int[] kWeakestRows(int[][] mat, int k) {
        List<int[]> power = new ArrayList<>();
        for (int i = 0; i < mat.length; i++) {
            int left = 0;
            int right = mat[0].length - 1;
            // 存放最后一个1的位置
            int pos = -1;
            while (left <= right){
                int mid = left + (right - left) / 2;
                if (mat[i][mid] == 0){
                    right = mid - 1;
                }else{
                    pos = mid;
                    left = mid + 1;
                }
            }
            // 存放[权利(1的数量),层数]
            power.add(new int[]{pos + 1, i});
        }
        // 构建小根堆
        PriorityQueue<int[]> que = new PriorityQueue<>((a, b) -> {
            // 权利相同则比较层数
            if (a[0] == b[0]){
                return a[1] - b[1];
            }else{
                return a[0] - b[0];
            }
        });
        // 将权利数组加入小根堆
        que.addAll(power);

        int[] res = new int[k];
        for (int i = 0; i < res.length; i++) {
            // 将权利最小的弹出加入结果集
            res[i] = que.poll()[1];
        }
        return res;
    }

    /*
     * 1346. 检查整数及其两倍数是否存在
     */
    public boolean checkIfExist(int[] arr) {
        Arrays.sort(arr);
        for (int i = 0; i < arr.length; i++) {
            int left = 0;
            int right = arr.length - 1;
            while (left <= right){
                int mid = left + (right - left) / 2;
                if (arr[mid] == arr[i] * 2 && mid != i){
                    return true;
                }else if(arr[mid] > arr[i] * 2){
                    right = mid - 1;
                }else{
                    left = mid + 1;
                }
            }
        }
        return false;
    }

    /*
     * 633. 平方数之和
     */
    public boolean judgeSquareSum(int c) {
        int maxC = (int) Math.sqrt(c);
        for (int i = 0; i <= maxC; i++) {
            long left = 0;
            long right = maxC;
            while (left <= right){
                long mid = left + (right - left) / 2;
                if (i * i + mid * mid == c){
                    return true;
                }else if (i * i + mid * mid > c){
                    right = mid - 1;
                }else{
                    left = mid + 1;
                }
            }
        }
        return false;
    }

    /*
     * 40. 组合总和 II
     */
    public List<List<Integer>> combinationSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> path = new ArrayList<Integer>();
        Arrays.sort(nums);
        dfs(0, nums, res, path, target);
        return res;
    }

    private void dfs(int startIndex, int[] nums, List<List<Integer>> res, List<Integer> path, int target) {
        if (target < 0){
            return;
        }
        if (target == 0){
            res.add(new ArrayList<>(path));
            return;
        }

        for (int j = startIndex; j < nums.length; j++) {
            // 去重
            if(j > startIndex && nums[j] == nums[j - 1]) {
                continue;
            }
            path.add(nums[j]);
            // 一个数只能用一次，将startIndex设置为下一个数，且target越来越少
            dfs(j + 1, nums, res, path, target - nums[j]);
            path.remove(path.size() - 1);
        }
    }

    /*
     *  79. 单词搜索
     */
    boolean flag = false;
    public boolean exist(char[][] board, String word) {
        boolean[][] visited = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == word.charAt(0)){
                    backtracking(0, i, j, board, word);
                    if (flag){
                        return true;
                    }
                }
            }
        }
        return flag;
    }

    private void backtracking(int startIndex, int i, int j, char[][] board, String word) {
        if (startIndex >= word.length()){
            flag = true;
            return;
        }
        if (flag || i < 0 || i >= board.length || j < 0 || j >= board[0].length
                || board[i][j] != word.charAt(startIndex)){
            return;
        }
        // 标记为访问过
        board[i][j] += 256;
        backtracking(startIndex + 1, i + 1, j, board, word);
        backtracking(startIndex + 1, i - 1, j, board, word);
        backtracking(startIndex + 1, i, j + 1, board, word);
        backtracking(startIndex + 1, i, j - 1, board, word);
        board[i][j] -= 256;
    }

    /*
     * 417. 太平洋大西洋水流问题
     */
    int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        int n = heights.length;
        int m = heights[0].length;
        // 记录是否能到达太平洋/大西洋
        boolean[][] pacific = new boolean[n][m];
        boolean[][] atlantic = new boolean[n][m];
        // 第一列
        for (int i = 0; i < n; i++) {
            bfs(i, 0, pacific, heights);
        }
        // 第一行
        // 从1开始，去掉对角，此点无法同时到达两个洋
        for (int j = 1; j < m; j++) {
            bfs(0, j, pacific, heights);
        }
        // 最后一列
        for (int i = 0; i < n; i++) {
            bfs(i, m - 1, atlantic, heights);
        }
        // 最后一行
        // 到m-1结束，去掉对角，此点无法同时到达两个洋
        for (int j = 0; j < m - 1; j++) {
            bfs(n - 1, j, atlantic, heights);
        }

        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                // 同时到达
                if (pacific[i][j] && atlantic[i][j]){
                    res.add(Arrays.asList(i, j));
                }
            }
        }
        return res;
    }

    private void bfs(int row, int col, boolean[][] ocean, int[][] heights) {
        if (ocean[row][col]){
            return;
        }
        ocean[row][col] = true;
        Queue<int[]> que = new LinkedList<>();
        que.add(new int[]{row, col});
        while (!que.isEmpty()){
            int[] cell = que.poll();
            for (int[] dir : dirs) {
                int newRow = cell[0] + dir[0];
                int newCol = cell[1] + dir[1];
                // 此处每次都要判断是否已访问
                if (newCol >= 0 && newCol < heights[0].length && newRow >= 0 && newRow < heights.length && heights[newRow][newCol] >= heights[cell[0]][cell[1]] && !ocean[newRow][newCol]) {
                    // 与DFS不同之处，BFS是在此时做标记
                    ocean[newRow][newCol] = true;
                    que.offer(new int[]{newRow, newCol});
                }
            }
        }
    }

    private void dfs(int row, int col, boolean[][] ocean, int[][] heights) {
        if (ocean[row][col]){
            return;
        }
        ocean[row][col] = true;
        for (int[] dir : dirs) {
            int newRow = row + dir[0];
            int newCol = col + dir[1];
            // 当前点大于等于前一个点
            if (newCol >= 0 && newCol < heights[0].length && newRow >= 0 && newRow < heights.length && heights[newRow][newCol] >= heights[row][col]) {
                dfs(newRow, newCol, ocean, heights);
            }
        }
    }

    /*
     * 1855. 下标对中的最大距离
     */
    public int maxDistance(int[] nums1, int[] nums2) {
        int res = 0;
        int left = 0;
        int right;
        for (int i = 0; i < nums1.length; i++) {
            // 直接从上一个循环的位置开始，不用声明
            // left = i;
            right = nums2.length - 1;
            while (left <= right){
                int mid = left + (right - left) / 2;
                if (nums1[i] <= nums2[mid]){
                    left = mid + 1;
                }else {
                    right = mid - 1;
                }
            }
            // left 指向了目标下标的后一位
            res = Math.max(res, left - i - 1);
        }
        return res;
    }

    /*
     * 934. 最短的桥
     */
    public int shortestBridge(int[][] grid) {
        // outer: 跳出整个循环
        outer:for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1){
                    // 将此岛相连的所有1改为2
                    dfs(grid, i, j);
                    break outer;
                }
            }
        }
        return bfs(grid);
    }

    private int bfs(int[][] grid) {
        Queue<int[]> que = new LinkedList<>();
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1){
                    que.offer(new int[]{i, j});
                }
            }
        }
        int count = 0;
        while (!que.isEmpty()){
            count++;
            int size = que.size();
            while (size-- > 0) {
                int[] rc = que.poll();
                for (int[] dir : dirs) {
                    int row = rc[0] + dir[0];
                    int col = rc[1] + dir[1];
                    // 遇到的1都是自己的岛上的
                    if (row < 0 || row >= grid.length || col < 0 || col >= grid[0].length || grid[row][col] == 1) {
                        continue;
                    }
                    // 触碰到另一方的岛
                    if (grid[row][col] == 2) {
                        // 减去终点的一次
                        return count - 1;
                    }
                    grid[row][col] = 1;
                    que.offer(new int[]{row, col});
                }
            }
        }
        return 0;
    }

    private void dfs(int[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != 1){
            return;
        }
        grid[i][j] = 2;
        for (int[] dir : dirs) {
            dfs(grid, i + dir[0], j + dir[1]);
        }
    }

    /*
     * 1926. 迷宫中离入口最近的出口
     */
    public int nearestExit(char[][] maze, int[] entrance) {
        Queue<int[]> que = new LinkedList<>();
        que.offer(entrance);
        maze[entrance[0]][entrance[1]] = '+';
        int step = 0;
        while (!que.isEmpty()){
            step++;
            int size = que.size();
            while (size-- > 0){
                int[] rc = que.poll();
                if ((rc[0] == 0 || rc[0] == maze.length - 1 || rc[1] == 0 || rc[1] == maze[0].length - 1)
                        && !Arrays.equals(rc,entrance)
                ){
                    // 减去到达终点的那一次
                    return step - 1;
                }
                // 相当于一次向四周扩散一格
                for (int[] dir : dirs) {
                    int row = dir[0] + rc[0];
                    int col = dir[1] + rc[1];
                    if (row >= 0 && row < maze.length && col >= 0 && col < maze[0].length && maze[row][col] == '.'){
                        // 当前格不是终点，将其设为墙
                        maze[row][col] = '+';
                        que.offer(new int[]{row, col});
                    }
                }
            }
        }
        return -1;
    }

    /*
     * 1319. 连通网络的操作次数
     */
    public int makeConnected(int n, int[][] connections) {
        if (connections.length < n - 1) {
            return -1;
        }
        // 构建无向图
        List<Integer>[] graph = new List[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }
        // 连接两个点，无向图两边都要连接
        for (int[] connection : connections) {
            graph[connection[0]].add(connection[1]);
            graph[connection[1]].add(connection[0]);
        }

        int res = 0;
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; i++) {
            // 没找到，代表之前访问不到
            if(!visited[i]) {
                // 将连接的点全部标记为可访问
                dfs(i, visited, graph);
                // 结果加一
                res++;
            }
        }
        // 减去连接最后一个点的一次
        return res - 1;
    }

    private void dfs(int i, boolean[] visited, List<Integer>[] graph) {
        visited[i] = true;
        for (int g : graph[i]) {
            if (!visited[g]){
                dfs(g, visited, graph);
            }
        }
    }

    /*
     * 1376. 通知所有员工所需的时间
     */
    int max = 0;
    public int numOfMinutes(int n, int headID, int[] manager, int[] informTime) {
        List<Integer>[] graph = new ArrayList[n];
        for (int i = 0; i < graph.length; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int i = 0; i < n; i++) {
            // 传递信息的上司
            int from = manager[i];
            if (from == -1){
                continue;
            }
            // 下属
            int to = i;
            graph[from].add(to);
        }
        dfs(graph, informTime, headID, 0);
        return max;
    }

    private void dfs(List<Integer>[] graph, int[] informTime, int cur, int sum) {
        // 需要花最长时间的传播路线就是最少所需时间
        max = Math.max(max, sum);
        for (Integer manager : graph[cur]) {
            // 加上此次用到的时间
            dfs(graph, informTime, manager, sum + informTime[cur]);
        }
    }

    /*
     * 802. 找到最终的安全状态
     */
    public List<Integer> eventualSafeNodes(int[][] graph) {
        // 0 ：该节点尚未被访问
        // 1 ：该节点位于递归栈中，或者在某个环上
        // 2 ：该节点搜索完毕，是一个安全节点
        int[] state = new int[graph.length];
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < graph.length; i++) {
            if (safe_dfs(graph, state, i)){
                res.add(i);
            }
        }
        return res;
    }

    private boolean safe_dfs(int[][] graph, int[] state, int i) {
        // 不是第一次访问，如果为1则代表入环，不为1代表安全
        if (state[i] > 0){
            return state[i] == 2;
        }
        // 标记为被访问过
        state[i] = 1;
        for (Integer next : graph[i]) {
            // 判断下一个点
            if (!safe_dfs(graph, state, next)){
                return false;
            }
        }
        // 递归完没有入环，标记为安全节点
        state[i] = 2;
        return true;
    }

    /*
     * 1129. 颜色交替的最短路径
     */
    public int[] shortestAlternatingPaths(int n, int[][] redEdges, int[][] blueEdges) {
        // 边的集合转换为邻接表
        ArrayList<Integer>[] redList = new ArrayList[n];
        ArrayList<Integer>[] blueList = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            redList[i] = new ArrayList<>();
            blueList[i] = new ArrayList<>();
        }
        for (int[] edge : redEdges) {
            redList[edge[0]].add(edge[1]);
        }
        for (int[] edge : blueEdges) {
            blueList[edge[0]].add(edge[1]);
        }
        Queue<int[]> que = new LinkedList<>();
        // 节点 颜色 距离
        // 0-red,1-blue
        // 初始0可以走红色和蓝色两条路
        que.add(new int[]{0, 0, 0});
        que.add(new int[]{0, 1, 0});
        // 注意，因为有红蓝两种选择，则应该有两种标记
        boolean[] redVisited = new boolean[n];
        boolean[] blueVisited = new boolean[n];
        redVisited[0] = true;
        blueVisited[0] = true;
        int[] res = new int[n];
        Arrays.fill(res, -1);
        while (!que.isEmpty()){
            int[] x = que.poll();
            int v = x[0], color = x[1], len = x[2];
            // 第一次访问
            if (res[v] == -1){
                res[v] = len;
            }
            if (color == 0){ // red
                for (int w : blueList[v]) {
                    if (!blueVisited[w]){
                        blueVisited[w] = true;
                        que.add(new int[] {w, 1, len + 1});
                    }
                }
            }else{ // blue
                for (int w : redList[v]) {
                    if (!redVisited[w]){
                        redVisited[w] = true;
                        que.add(new int[] {w, 0, len + 1});
                    }
                }
            }
        }

        return res;
    }

    /*
     * 1466. 重新规划路线
     */
    int res = 0;
    public int minReorder(int n, int[][] connections) {
        // 有向图改无向图
        List<int[]>[] graph = new List[n];
        for (int i = 0; i < graph.length; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int[] c : connections) {
            // 0为发出的一方，1为被进入的一方
            graph[c[0]].add(new int[] {c[1], 0});
            graph[c[1]].add(new int[] {c[0], 1});
        }
        boolean[] visited = new boolean[n];
        dfs(0, graph, visited);

        return res;
    }

    private void dfs(int i, List<int[]>[] graph, boolean[] visited) {
        visited[i] = true;
        for (int[] j : graph[i]) {
            if (!visited[j[0]]){
                // 如果是发出的一方
                if (j[1] == 0){
                    res++;
                }
                dfs(j[0], graph, visited);
            }
        }
    }

    /*
     * 1306. 跳跃游戏 III
     */
    public boolean canReach(int[] arr, int start) {
        return dfs(arr, start);
    }

    private boolean dfs(int[] arr, int start) {
        if (start < 0 || start >= arr.length || arr[start] == -1){
            return false;
        }
        int step = arr[start];
        arr[start] = -1;
        return step == 0 || dfs(arr, start + step) || dfs(arr, start - step);
    }

    /*
     * 670. 最大交换
     */
    public int maximumSwap(int num) {
        char[] chs = String.valueOf(num).toCharArray();
        // 存储当前阶段最大值的索引
        int[] maxIndex = new int[chs.length];
        int max = chs.length - 1;
        for (int i = chs.length - 1; i >= 0; i--) {
            if (chs[i] - '0' > chs[max] - '0'){
                max = i;
            }
            // 存入阶段最大值索引
            maxIndex[i] = max;
        }

        // 找到第一个不是最大的数，将该位置和右边最大数换位置
        for (int i = 0; i < chs.length; i++) {
            char iValue = chs[i];
            char maxValue = chs[maxIndex[i]];
            if (iValue != maxValue){
                chs[i] = maxValue;
                chs[maxIndex[i]] = iValue;
                break;
            }
        }
        return Integer.parseInt(new String(chs));
    }

    /*
     * 213. 打家劫舍 II
     */
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        if (n == 1) return nums[0];
        int[] dp1 = new int[n];
        int[] dp2 = new int[n];
        // 不选第一个
        dp1[0] = 0;
        dp1[1] = nums[1];
        for (int i = 2; i < dp1.length; i++) {
            dp1[i] = Math.max(dp1[i - 1], dp1[i - 2] + nums[i]);
        }
        // 不选最后一个
        dp2[0] = 0;
        dp2[1] = nums[0];
        for (int i = 2; i < dp2.length; i++) {
            // nums[i - 1] 因为不能选最后一个，所以始终是 i - 1
            dp2[i] = Math.max(dp2[i - 1], dp2[i - 2] + nums[i - 1]);
        }
        return Math.max(dp1[n - 1], dp2[n - 1]);
    }

    /*
     * 746. 使用最小花费爬楼梯
     */
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        // 走到第i层楼梯的最小花费
        int[] dp = new int[n];
        dp[0] = cost[0];
        dp[1] = cost[1];
        for (int i = 2; i < n; i++) {
            dp[i] = Math.min(dp[i - 1], dp[i - 2]) + cost[i];
        }
        return Math.min(dp[n - 1], dp[n - 2]);
    }

    /*
     * 62. 不同路径
     */
    public int uniquePaths(int m, int n) {
        // 只能走右下方向
        // dp[i][j]表示左格子和上格子的步数的和
        int[][] dp = new int[m][n];
        // 第一列无法被访问，填充1
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        // 第一行无法被访问，填充1
        for (int j = 0; j < n; j++) {
            dp[0][j] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    /*
     * 45. 跳跃游戏 II
     */
    public int jump(int[] nums) {
        // dp[i]表示到达当前步的最小值
        int[] dp = new int[nums.length];
        dp[0] = 0;
        // 赋最大值
        for (int i = 1; i < dp.length; i++) {
            dp[i] = nums.length + 1;
        }
        for (int i = 0; i < nums.length; i++) {
            // 循环可到达步数次
            for (int j = 1; j <= nums[i]; j++) {
                // 到达最后数直接返回
                if (i + j >= nums.length){
                    return dp[dp.length - 1];
                }
                dp[i + j] = Math.min(dp[i + j], dp[i] + 1);
            }
        }
        return dp[dp.length - 1];
    }

    /*
     * 413. 等差数列划分
     */
    public int numberOfArithmeticSlices(int[] nums) {
        int n = nums.length;
        if (n == 1 || n == 2){
            return 0;
        }
        // dp[i]表示当前位置能够组成多少个等差数列
        int[] dp = new int[n];
        dp[0] = 0;
        dp[1] = 0;
        int sum = 0;
        for (int i = 2; i < n; i++) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]){
                dp[i] = dp[i - 1] + 1;
                sum += dp[i];
            }
        }
        return sum;
    }

    /*
     * 5. 最长回文子串
     */
    public String longestPalindrome(String s) {
        int n = s.length();
        if (n < 2){
            return s;
        }
        // dp[i][j]：[i...j]处是否为回文子串
        boolean[][] dp = new boolean[n][n];
        // 初始化对角线，因为只有单个字符的时候必定是回文子串
        for (int i = 0; i < dp.length; i++) {
            dp[i][i] = true;
        }

        char[] chs = s.toCharArray();
        int begin = 0;
        int maxEnd = 1;
        // 绘制表格
        // 右边界，对角线已经初始化过不用计算
        for (int j = 1; j < n; j++) {
            // 左边界
            for (int i = 0; i < j; i++) {
                if (chs[i] != chs[j]){
                    dp[i][j] = false;
                }else{
                    // 字符相等且头尾去掉以后没有字符剩余，或者剩下一个字符的时候，肯定是回文串
                    // j - i + 1 < 4
                    if (j - i < 3){
                        dp[i][j] = true;
                    }else {
                        // 长度为3以上则采用表格左下角中(子串)的计算结果
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                // 为true代表是回文，记录长度和起始位置
                if (dp[i][j] && j - i + 1 > maxEnd){
                    begin = i;
                    maxEnd = j - i + 1;
                }
            }
        }
        return s.substring(begin, begin + maxEnd);
    }

    /*
     * 91. 解码方法
     */
    public int numDecodings(String s) {
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            if (s.charAt(i - 1) != '0'){
                dp[i] += dp[i - 1];
            }
            if (i > 1 && s.charAt(i - 2) != '0' && ((s.charAt(i - 2) - '0') * 10 + (s.charAt(i - 1) - '0') <= 26)){
                dp[i] += dp[i - 2];
            }
        }
        return dp[n];
    }

    /*
     * 139. 单词拆分
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        // dp[i]表示字符串s的前i个字符能否拆分成wordDict
        boolean[] dp = new boolean[n + 1];
        // 假定dp[0]为空串合法
        dp[0] = true;
        // i <= n 是因为subString是左闭右开，右区间取值需要+1
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                // 判断[0...i]在wordDict中是否存在
                if (dp[j] && wordDict.contains(s.substring(j, i))){
                    dp[i] = true;
                    break;
                }
            }
        }
        // dp[n]表示整个字符串是否能拆分成wordDict
        return dp[n];
    }

    /*
     * 300. 最长递增子序列
     */
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        // dp[i]表示[0...i]的递增子序列长度
        int[] dp = new int[n];
        dp[0] = 1;
        int max = 1;
        for (int i = 1; i < n; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]){
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            max = Math.max(max, dp[i]);
        }
        return max;
    }

    /*
     * 673. 最长递增子序列的个数
     */
    public int findNumberOfLIS(int[] nums) {
        int n = nums.length, maxSize = 0, res = 0;
        int[] dp1 = new int[n];
        int[] dp2 = new int[n];
        for (int i = 0; i < n; i++) {
            dp1[i] = 1;
            dp2[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    if (dp1[i] < dp1[j] + 1) {
                        dp1[i] = dp1[j] + 1;
                        dp2[i] = dp2[j];
                    } else if (dp1[i] == dp1[j] + 1) {
                        dp2[i] += dp2[j];
                    }
                }
            }
            if (dp1[i] > maxSize){
                maxSize = dp1[i];
                res = dp2[i];
            } else if (dp1[i] == maxSize) {
                res += dp2[i];
            }
        }
        return res;
    }

    /*
     * 1624. 两个相同字符之间的最长子字符串
     */
    public int maxLengthBetweenEqualCharacters(String s) {
        int n = s.length();
        int[] dp = new int[n];
        dp[0] = 0;
        boolean flag = false;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if(s.charAt(i) == s.charAt(j)){
                    dp[i] = Math.max(dp[i - 1], i - j - 1);
                    flag = true;
                    break;
                } else{
                    dp[i] = dp[i - 1];
                }
            }
        }
        if (!flag){
            return -1;
        }
        return dp[n - 1];
    }

    /*
     * 322. 零钱兑换
     */
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.length; j++) {
                if (coins[j] <= i){
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }

    /*
     * 1143. 最长公共子序列
     */
    public int longestCommonSubsequence(String text1, String text2) {
        int n = text1.length();
        int m = text2.length();
        // dp[i][j]代表text1,text2在[i][j]处有几个公共子序列
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)){
                    // text1[0..i-1] 和 text2[0..j-1] 处的公共子序列数加一 左上角格
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else{
                    // text1[0..i-1] 和 text2[0..j] 处的公共子序列数 上一格
                    // text1[0..i] 和 text2[0..j-1] 处的公共子序列数 左一格
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[n][m];
    }

    /*
     * 115. 不同的子序列
     */
    public int numDistinct(String s, String t) {
        int n = s.length();
        int m = t.length();
        // 这样dp可以表示s/t为空串
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 0; i < n; i++) {
            dp[i][0] = 1;
        }
        // [0][0]为特殊位置，对应表格中的空字符串
        for (int j = 1; j < m; j++) {
            dp[0][j] = 0;
        }

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)){
                    // 用现在这个相同字符(j - 1) + 不用现在这个(j - 1),而是用后面的(j)
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                }else {
                    // 现在这个字符(i)不相同就用前一个(i - 1)字符做计算结果
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[n][m];
    }

    /*
     * 583. 两个字符串的删除操作
     */
    public int minDistance2(String word1, String word2) {
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
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)){
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

    /*
     * 72. 编辑距离
     */
    public int minDistance3(String word1, String word2) {
        int n = word1.length();
        int m = word2.length();
        // dp[i-1, j-1] 想要得到相等所需要编辑的最少次数
        int[][] dp = new int[n + 1][m + 1];
        // word1为空字符串时，word2要删除i次才能相等
        for (int i = 0; i <= n; i++) {
            dp[i][0] = i;
        }
        // word2为空字符串时，word1要删除j次才能相等
        for (int j = 0; j <= m; j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)){
                    // 相等则不做编辑
                    dp[i][j] = dp[i - 1][j - 1];
                } else{
                    // 情况1：替换[i-1]或[j-1]
                    // 情况2：word1删除
                    // 情况3：word2删除
                    dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
            }
        }
        return dp[n][m];
    }

    /*
     * 343. 整数拆分
     */
    public int integerBreak(int n) {
        int[] dp = new int[n + 1];
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = i - 1; j > 0; j--) {
                dp[i] = Math.max(dp[i], Math.max(dp[j] * (i - j), j * (i - j)));
            }
        }
        return dp[n];
    }

    /*
     * 1636. 按照频率将数组升序排序
     */
    public int[] frequencySort(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
            list.add(nums[i]);
        }

        list.sort((a, b) -> {
            Integer aVal = map.get(a);
            Integer bVal = map.get(b);
            if (aVal == bVal){
                return b - a;
            }
            return aVal - bVal;
        });


        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    /*
     * 1636. 按照频率将数组升序排序
     */
    public int[] frequencySort1(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
            list.add(nums[i]);
        }

        list.sort((a, b) -> {
            Integer aVal = map.get(a);
            Integer bVal = map.get(b);
            if (aVal == bVal){
                return b - a;
            }
            return aVal - bVal;
        });


        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    /*
     * 63. 不同路径 II
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int n = obstacleGrid.length;
        int m = obstacleGrid[0].length;
        int[][] dp = new int[n][m];
        for (int i = 0; i < n; i++) {
            if (dp[i][0] == 1) break; // 一旦遇到障碍，后续都到不了
            dp[i][0] = 1;
        }
        for (int j = 0; j < m; j++) {
            if (dp[0][j] == 1) break; // 一旦遇到障碍，后续都到不了
            dp[0][j] = 1;
        }
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < m; j++) {
                if (obstacleGrid[i][j] == 0) {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }

        return dp[n - 1][m - 1];
    }

    /*
     * 343. 整数拆分 换个思路
     */
    public int integerBreak1(int n) {
        int[] dp = new int[n + 1];
        dp[2] = 1;
        for (int i = 3; i <= n; i++) {
            // 这里的 j 其实最大值为 i-j,再大只不过是重复而已
            for (int j = 1; j <= i - j; j++) {
                // 两个情况：拆分成两个数、拆分成两个及以上个数(dp[i - j])
                // j * (i - j) 是单纯的把整数 i 拆分为两个数 也就是 j,i-j,再相乘
                // 而j * dp[i - j]是将 i 拆分成两个以及两个以上的个数(从dp表格中找它的计算结果),再相乘。
                dp[i] = Math.max(dp[i], Math.max(j * (i - j), j * dp[i - j]));
            }
        }
        return dp[n];
    }

    /*
     * 96. 不同的二叉搜索树
     */
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                // j-1 为j为头结点左子树节点数量，i-j 为以j为头结点右子树节点数量
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }

    /*
     * 背包问题
     */
    public static void main(String[] args) {
        int[] weight = {1, 3, 4};
        int[] value = {15, 20, 30};
        int bagsize = 4;
        testweightbagproblem3(weight, value, bagsize);
    }

    public static void testweightbagproblem3(int[] weight, int[] value, int bagsize){
        int[][] dp = new int[weight.length + 1][bagsize + 1];
        for (int i = 1; i <= weight.length; i++) {
            for (int j = 1; j <= bagsize; j++) {
                if (j < weight[i - 1]){
                    dp[i][j] = dp[i - 1][j];
                }else {
                    // 物品可以重复，可以在装过本物品的情况下继续装本物品
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - weight[i - 1]] + value[i - 1]);
                }
            }
        }
        for (int[] ints : dp) {
            System.out.println(Arrays.toString(ints));
        }
    }

    public static void testweightbagproblem4(int[] weight, int[] value, int bagsize){
        int[] dp = new int[bagsize + 1];
        for (int i = 0; i < weight.length; i++) {
            for (int j = weight[i]; j <= bagsize; j++) {
                dp[j] = Math.max(dp[j], dp[j - weight[i]] + value[i]);
            }
        }
        System.out.println(Arrays.toString(dp));
    }

    public static void testweightbagproblem1(int[] weight, int[] value, int bagsize){
        int[][] dp = new int[weight.length][bagsize + 1];
        for (int j = weight[0]; j <= bagsize; j++) {
            dp[0][j] = value[0];
        }
        for (int i = 1; i < weight.length; i++) {
            for (int j = 1; j <= bagsize; j++) {
                if (j < weight[i]){
                    // 选择不放物品i
                    // 那么从下标为[0-i]的物品里任意取，放进容量为j的背包
                    // 和
                    // 从下标为[0-i-1]的物品里任意取，放进容量为j的背包 是等价的
                    dp[i][j] = dp[i - 1][j];
                }else {
                    // 假设在容量为j的背包里，我一定要把重量为weight[i]的物品i放进去。
                    // 既然我们一定要放物品i，那么
                    // dp[i][j]：从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大。
                    // 就变成了
                    // 物品i的价值 + 从下标为[0-i-1]的物品里任意取，放进容量为 j - weight[i] (已经放了物品i了)的背包
                    // (因为放了物品i之后，容量为j的背包里容量只剩下 j - weight[i] 了)，价值总和最大。
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]);
                }
            }
        }
        System.out.println("weight:" + Arrays.toString(weight));
        System.out.println("value:" + Arrays.toString(value));
        for (int i = 0; i < dp.length; i++) {
            System.out.println(Arrays.toString(dp[i]));
        }
    }


    public static void testweightbagproblem2(int[] weight, int[] value, int bagsize){
        // 容量为j的背包，所背的物品价值可以最大为dp[j]
        int[] dp = new int[bagsize + 1];
        for (int i = 0; i < weight.length; i++) {
            // 倒序遍历背包容量，防止重复使用物品
            for (int j = bagsize; j >= weight[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - weight[i]] + value[i]);
            }
        }

        System.out.println("weight:" + Arrays.toString(weight));
        System.out.println("value:" + Arrays.toString(value));
        System.out.println(Arrays.toString(dp));
    }

    /*
     * 416. 分割等和子集
     */
    public boolean canPartition(int[] nums) {
        int[] dp = new int[10001];
        int target = Arrays.stream(nums).sum();
        if (target % 2 != 0){
            return false;
        }
        target = target / 2;
        for (int i = 0; i <= nums.length; i++) {
            // 每一个元素不可重复放入，所以从大到小遍历
            for (int j = target; j >= nums[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - nums[i]] + nums[i]);
            }
        }
        return target == dp[target];
    }

    public boolean canPartition2(int[] nums) {
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        if (sum % 2 != 0){
            return false;
        }
        int target = sum / 2;
        int[][] dp = new int[nums.length][target + 1];
        for (int j = nums[0]; j <= target; j++) {
            dp[0][j] = nums[0];
        }
        for (int i = 1; i < nums.length; i++) {
            for (int j = 1; j <= target; j++) {
                if (j < nums[i]){
                    dp[i][j] = dp[i - 1][j];
                }else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - nums[i]] + nums[i]);
                }
            }
        }
        return target == dp[nums.length - 1][target];
    }

    /*
     * 1640. 能否连接形成数组
     */
    public boolean canFormArray(int[] arr, int[][] pieces) {
        HashMap<Integer, int[]> map = new HashMap<>();
        for (int[] piece : pieces) {
            map.put(piece[0], piece);
        }
        for (int i = 0; i < arr.length;) {
            if (!map.containsKey(arr[i])){
                return false;
            }
            int[] piece = map.get(arr[i]);
            for (int j = 0; j < piece.length; j++) {
                // 依次遍历子数组，如果顺序不对直接返回
                if (arr[i] != piece[j]){
                    return false;
                }
                i++;
            }
        }
        return true;
    }

    /*
     * 1049. 最后一块石头的重量 II
     */
    public int lastStoneWeightII(int[] stones) {
        int sum = 0;
        for (int i = 0; i < stones.length; i++) {
            sum += stones[i];
        }
        int target = sum / 2;
        int[][] dp = new int[stones.length][target + 1];
        for (int j = stones[0]; j <= target; j++) {
            dp[0][j] = stones[0];
        }
        for (int i = 1; i < stones.length; i++) {
            for (int j = 1; j <= target; j++) {
                if (j < stones[i]){
                    dp[i][j] = dp[i - 1][j];
                }else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - stones[i]] + stones[i]);
                }
            }
        }
        for (int[] ints : dp) {
            System.out.println(Arrays.toString(ints));
        }
        return (sum - dp[stones.length - 1][target]) - dp[stones.length - 1][target];
    }

    /*
     * 1049. 最后一块石头的重量 II
     */
    public int lastStoneWeightII2(int[] stones) {
        int sum = 0;
        for (int i = 0; i < stones.length; i++) {
            sum += stones[i];
        }
        int target = sum / 2;
        int[] dp = new int[target + 1];
        for (int i = 0; i < stones.length; i++) {
            for (int j = target; j >= stones[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - stones[i]] + stones[i]);
            }
        }
        return (sum - dp[target]) -dp[target];
    }

    /*
     * 494. 目标和
     */
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int i = 0; i < nums.length; i++){
            sum += nums[i];
        }
        // target绝对值比sum还大 || 都比target小，凑不出target
        if (Math.abs(target) > sum || (target + sum) % 2 == 1){
            return 0; // 此时没有方案
        }
        int bagSize = (target + sum) / 2;
        if (bagSize < 0) {
            return 0;
        }
        // dp[j]为恰好能凑满容量为j的背包方案数
        int[] dp = new int[bagSize + 1];
        dp[0] = 1;
        for (int i = 0; i < nums.length; i++) {
            // 当不能装下nums[i]时,方案数直接继承之前的dp[j](不进入循环)
            for (int j = bagSize; j >= nums[i]; j--) {
                dp[j] += dp[j - nums[i]];
            }
        }
        return dp[bagSize];
    }

    /*
     * 474. 一和零
     */
    public int findMaxForm(String[] strs, int m, int n) {
        // dp[i][j]表示i个0和j个1时的最大子集
        int[][] dp = new int[m + 1][n + 1];
        int zeroNum, oneNum;
        for (String str : strs) {
            zeroNum = 0;
            oneNum = 0;
            for (char c : str.toCharArray()) {
                if (c == '0'){
                    zeroNum++;
                }else {
                    oneNum++;
                }
            }
            for (int i = m; i >= zeroNum; i--) {
                for (int j = n; j >= oneNum; j--) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1);
                }
            }
        }
        return dp[m][n];
    }

    /*
     * 1652. 拆炸弹
     */
    public int[] decrypt(int[] code, int k) {
        int len = code.length;
        int[] res = new int[len];
        int temp = k >= 0 ? k == 0 ? 0 : 1 : -1;
        for (int i = 0; i < len; i++) {
            int sum = 0;
            // 此处为 != 正数负数都可以计算
            for (int j = temp; j != k + temp; j += temp) {
                sum += code[(i + j + len) % len];
            }
            res[i] = sum;
        }
        return res;
    }

    /*
     * 518. 零钱兑换 II
     */
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        // 总和为0有一种方法
        dp[0] = 1;
        for (int i = 0; i < coins.length; i++) {
            // 当coins = [1,2,5]
            // 总和为1-5,面额为[1]时只有使用硬币1一种方法
            // 总和为2，面额为[1,2]时有dp[2-2] = dp[0]的一种方法 + dp[2]的一种方法 两种方法
            // 总和为3，面额为[1,2]时有dp[3-2] = dp[1]的一种方法 + dp[3]的一种方法 两种方法
            // 总和为5，面额为[1,2]时有dp[5-2] = dp[3]的两种方法 + dp[5]的一种方法 三种方法
            // 总和为5，面额为[1,2,5]时有dp[5-5] = dp[0]的一种方法 + dp[5]的三种方法 四种方法
            // 可以看出，dp[j-coins[i]](总额为j-coins[i]的方法数) + dp[j](总额为j时的方法数)
            // 递推公式为：总和为j时，有dp[j] + dp[j - coins[i]]种方法
            for (int j = coins[i]; j <= amount; j++) {
                dp[j] += dp[j - coins[i]];
            }
        }
        return dp[amount];
    }

    /*
     * 518. 零钱兑换 II
     */
    public int change1(int amount, int[] coins) {
        int[][] dp = new int[coins.length + 1][amount + 1];
        for (int i = 0; i <= coins.length; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= coins.length; i++) {
            for (int j = 1; j <= amount; j++) {
                if (j < coins[i - 1]){
                    dp[i][j] = dp[i - 1][j];
                }else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i - 1]];
                }
            }
        }
        return dp[coins.length][amount];
    }

    /*
     * 377. 组合总和 Ⅳ
     */
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int j = 0; j <= target; j++) {
            for (int i = 0; i < nums.length; i++) {
                if (j >= nums[i]){
                    dp[j] += dp[j - nums[i]];
                }
            }
        }
        return dp[target];
    }

    /*
     * 70. 爬楼梯
     */
    public int climbStairsFunc(int n, int m) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int j = 1; j <= n; j++) {
            for (int i = 1; i < m; i++) {
                if (j - i >= 0){
                    dp[j] += dp[j - i];
                }
            }
        }
        return dp[n];
    }

    /*
     * 322. 零钱兑换
     */
    public int coinChange1(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        // 填充最大值，递推公式需要寻找最小值
        Arrays.fill(dp, amount + 1);
        // 防止计算结果被初始值覆盖
        dp[0] = 0;
        for (int j = 0; j <= amount; j++) {
            for (int i = 0; i < coins.length; i++) {
                if (j >= coins[i]) {
                    dp[j] = Math.min(dp[j], dp[j - coins[i]] + 1);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }

    /*
     * 279. 完全平方数
     */
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        Arrays.fill(dp, n + 1);
        dp[0] = 0;
        for (int j = 0; j <= n; j++) {
            for (int i = 1; i * i <= j; i++) {
                dp[j] = Math.min(dp[j], dp[j - i * i] + 1);
            }
        }
        return dp[n] > n ? 0 : dp[n];
    }

    /*
     * 面试题 17.09. 第 k 个数
     */
    public int getKthMagicNumber(int k) {
        int[] dp = new int[k + 1];
        dp[1] = 1;
        int p3 = 1, p5 = 1, p7 = 1;
        for (int i = 2; i <= k; i++) {
            int num3 = dp[p3] * 3, num5 = dp[p5] * 5, num7 = dp[p7] * 7;
            dp[i] = Math.min(Math.min(num3, num5), num7);
            if (dp[i] == num3) {
                p3++;
            }
            if (dp[i] == num5) {
                p5++;
            }
            if (dp[i] == num7) {
                p7++;
            }
        }
        return dp[k];
    }

    /*
     * 611. 有效三角形的个数
     */
    public int triangleNumber(int[] nums) {
        Arrays.sort(nums);
        int res = 0;
        for (int i = nums.length - 1; i >= 2; i--) {
            int j = i - 1;
            // 优化掉了第三层循环
            int k = 0;
            while (k < j){
                // 从num[k]到num[j]的数都满足要求，结果直接加上j - k
                if (nums[k] + nums[j] > nums[i]){
                    res += j - k;
                    j--;
                }else{
                    // //否则k自增，重新判断
                    k++;
                }
            }
        }
        return res;
    }

    /*
     * 1894. 找到需要补充粉笔的学生编号
     */
    public int chalkReplacer(int[] chalk, int k) {
        long total = 0;
        for (int num : chalk) {
            total += num;
        }
        k %= total;
        int res = -1;
        for (int i = 0; i < chalk.length; i++) {
            if (chalk[i] > k){
                res = i;
                break;
            }
            k -= chalk[i];
        }

        return res;
    }

    /*
     * 93. 复原 IP 地址
     */
    public List<String> restoreIpAddresses(String s) {
        if (s.length() > 12) return new ArrayList<String>();
        List<String> res = new ArrayList<>();
        backtracking2(0, s, res, 0);
        return res;
    }

    private void backtracking2(int startIndex, String s, List<String> res, int pointNum) {
        if (pointNum == 3){
            if (isValid(s, startIndex, s.length() - 1)){
                res.add(s);
            }
            return;
        }
        for (int i = startIndex; i < s.length(); i++) {
            if (isValid(s, startIndex, i)){
                s = s.substring(0, i + 1) + "." + s.substring(i + 1);
                pointNum++;
                // 加上逗号，所以是i+2
                backtracking2(i + 2, s, res, pointNum);
                pointNum--;
                s = s.substring(0, i + 1) + s.substring(i + 2);
            }else{
                break;
            }
        }
    }

    private boolean isValid(String s, int start, int end) {
        if (start > end) {
            return false;
        }
        if (s.charAt(start) == '0' && start != end) { // 0开头的数字不合法
            return false;
        }
        int num = 0;
        for (int i = start; i <= end; i++) {
            if (s.charAt(i) > '9' || s.charAt(i) < '0') { // 遇到非数字字符不合法
                return false;
            }
            num = num * 10 + (s.charAt(i) - '0');
            if (num > 255) { // 如果大于255了不合法
                return false;
            }
        }
        return true;
    }

    /*
     * 131. 分割回文串
     */
    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        backtracking1(0, s, res, new ArrayList<String>());
        return res;
    }

    private void backtracking1(int startIndex, String s, List<List<String>> res, ArrayList<String> path) {
        if (startIndex == s.length()){
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = startIndex; i < s.length(); i++) {
            if (isStrCircle(s, startIndex, i)){
                path.add(s.substring(startIndex, i + 1));
            }else{
                continue;
            }
            backtracking1(i + 1, s, res, path);
            path.remove(path.size() -  1);
        }
    }

    private boolean isStrCircle(String s, int start, int end) {
        for (int i = start, j = end; i < j; i++, j--) {
            if (s.charAt(i)!= s.charAt(j)){
                return false;
            }
        }
        return true;
    }

    /*
     * 面试题 01.08. 零矩阵
     */
    public void setZeroes(int[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;
        boolean[] col = new boolean[n];
        boolean[] row = new boolean[m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (matrix[i][j] == 0){
                    col[i] = true;
                    row[j] = true;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (col[i] || row[j]){
                    matrix[i][j] = 0;
                }
            }
        }
    }

    /*
     * 1760. 袋子里最少数目的球
     */
    public int minimumSize(int[] nums, int maxOperations) {
        int left = 1;
        int right = 0;
        for (int num : nums) {
            right = Math.max(right, num);
        }
        int pos = right;
        while (left <= right){
            int mid = left + (right - left) / 2;
            int operations = 0;
            for (int num : nums) {
                operations += num / mid;
                // 能够整除的话结果为1，例如4/4=1，但是不需要分，要减1
                if (num % mid == 0){
                    operations--;
                }
            }
            // 分的次数大于所给的:表示mid值太小，分的次数过多，需要减少分的次数
            if (operations > maxOperations){
                left = mid + 1;
            }else {
                right = mid - 1;
                pos = mid;
            }
        }
        return pos;
    }

    /*
     * 1694. 重新格式化电话号码
     */
    public String reformatNumber(String number) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < number.length(); i++) {
            char ch = number.charAt(i);
            if (Character.isDigit(ch)){
                sb.append(ch);
            }
        }
        int n = sb.length();
        StringBuilder res = new StringBuilder();
        while (n > 4){
            res.append(sb.substring(0, 3)).append('-');
            sb.delete(0, 3);
            n -= 3;
        }
        if (n <= 3){
            res.append(sb);
        }else{
            res.append(sb.substring(0, 2)).append('-').append(sb.substring(2,4));
        }
        return res.toString();
    }

    /*
     * 777. 在LR字符串中交换相邻字符
     */
    public boolean canTransform(String start, String end) {
        int n = start.length();
        int i = 0, j = 0;
        while (i < n && j < n) {
            while (i < n && start.charAt(i) == 'X') {
                i++;
            }
            while (j < n && end.charAt(j) == 'X') {
                j++;
            }
            if (i < n && j < n) {
                if (start.charAt(i) != end.charAt(j)) {
                    return false;
                }
                char c = start.charAt(i);
                if ((c == 'L' && i < j) || (c == 'R' && i > j)) {
                    return false;
                }
                i++;
                j++;
            }
        }
        while (i < n) {
            if (start.charAt(i) != 'X') {
                return false;
            }
            i++;
        }
        while (j < n) {
            if (end.charAt(j) != 'X') {
                return false;
            }
            j++;
        }
        return true;
    }

    /*
     * 2169. 得到 0 的操作数
     */
    public int countOperations(int num1, int num2) {
        int count = 0;
        while (num1 != 0 && num2 != 0){
            if (num1 > num2){
                num1 -= num2;
            }else{
                num2 -= num1;
            }
            count++;
        }
        return count;
    }

    public long pickGifts(int[] gifts, int k) {
        PriorityQueue<Integer> queue = new PriorityQueue<>((a, b) -> b - a);
        Arrays.stream(gifts).forEach(queue::offer);
        while (k-- > 0) {
            Integer num = queue.poll();
            queue.offer((int) Math.sqrt(num));
        }
        long res = 0;
        while (!queue.isEmpty()) {
            res += queue.poll();
        }
        return res;
    }

    public int[] successfulPairs(int[] spells, int[] potions, long success) {
        int[] res = new int[spells.length];
        Arrays.sort(potions);
        for (int i = 0; i < spells.length; i++) {
            int left = 0;
            int right = potions.length - 1;
            while (left <= right) {
                int mid = (left + right) / 2;
                // 直到找到最小的，能够超过success的那个为止
                if ((long) potions[mid] * spells[i] >= success) {
                    // 因为排序过，只要找到了第一个，那么后面的就都能符合条件
                    res[i] = potions.length - mid;
                    // 当前的数能够符合条件，继续向内寻找更小的
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
        }
        return res;
    }
}
