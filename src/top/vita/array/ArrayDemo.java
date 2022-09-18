package top.vita.array;

import java.util.*;

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



}
