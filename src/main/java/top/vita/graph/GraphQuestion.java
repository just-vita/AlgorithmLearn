package top.vita.graph;

import cn.hutool.core.collection.ListUtil;

import java.util.*;

@SuppressWarnings("all")
public class GraphQuestion {

	int n;
	int[] father;

	// 并查集初始化
	public GraphQuestion() {
		this.n = 1005;
		this.father = new int[n];

		for (int i = 0; i < father.length; i++) {
			father[i] = i;
		}
	}

	// 并查集里寻根的过程
	public int find(int u) {
		if (u == father[u]) {
			return u;
		}
		father[u] = find(father[u]);
		return father[u];
	}

	// 将v->u 这条边加入并查集
	public void union(int u, int v) {
		u = find(u);
		v = find(v);
		if (u == v) {
			return;
		}
		father[u] = v;
	}

	// 判断 u 和 v 是否找到同一个根，根据本题题意去掉同根的边
	private Boolean same(int u, int v) {
		u = find(u);
		v = find(v);
		return u == v;
	}

	public int[] findRedundantConnection(int[][] edges) {
		for (int i = 0; i < edges.length; i++) {
			if (same(edges[i][0], edges[i][1])) {
				return edges[i];
			} else {
				union(edges[i][0], edges[i][1]);
			}
		}
		return null;
	}
	
	public static class EdgeComparator implements Comparator<Edge>{

		@Override
		public int compare(Edge o1, Edge o2) {
			return o1.weight - o2.weight;
		}
		
	}
	
	/*
	public Set<Edge> findRedundantConnection1(Graph graph) {
		// 假装并查集工具类
		GraphQuestion unionFind = new GraphQuestion();
		PriorityQueue<Edge> queue = new PriorityQueue<Edge>(new EdgeComparator());
		for (Edge edge : graph.edges) {
			queue.add(edge);
		}
		HashSet<Edge> result = new HashSet<Edge>();
		while (!queue.isEmpty()) {
			Edge edge = queue.poll();
			if (!unionFind.same(edge.from, edge.to)) {
				result.add(edge);
				unionFind.union(edge.from, edge.to);
			}
		}
		return result;
	}
	*/
	
	/*
	 * 1791. 找出星型图的中心节点
	 */
    public int findCenter(int[][] edges) {
        Map<Integer,Integer> map = new HashMap<>();
        for(int[] edge : edges){
        	if (map.getOrDefault(edge[0], 0) > 1) {
        		return edge[0];
        	}else if (map.getOrDefault(edge[1], 0) > 1) {
        		return edge[1];
        	}
            map.put(edge[0], map.getOrDefault(edge[0], 1) + 1);
            map.put(edge[1], map.getOrDefault(edge[1], 1) + 1);
        }
        return -1;
    }
	
    /*
     * 797. 所有可能的路径
     */
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
    	// 所有路径都是从0开始
    	path.add(0);
    	subIslandsDFS(graph, 0);
    	return res;
    }

	List<List<Integer>> res = new ArrayList<List<Integer>>();
    List<Integer> path = new ArrayList<Integer>();
    
    private void subIslandsDFS(int[][] graph, int x) {
    	// 遍历到底了，当前路径加入结果集
    	if (x == graph.length - 1) {
    		// 注意path要转为ArrayList
    		res.add(new ArrayList<>(path));
    		return;
    	}
    	// 到当前节点的终点为止一直遍历
    	for (int i = 0; i < graph[x].length; i++) {
			path.add(graph[x][i]);
			subIslandsDFS(graph, graph[x][i]);
			path.remove(path.size() - 1);
		}
    }
    
    /*
     * 1557. 可以到达所有点的最少点数目
     */
    public List<Integer> findSmallestSetOfVertices(int n, List<List<Integer>> edges) {
    	// 找入度为0的点
    	int[] arr = new int[n];
    	for (List<Integer> edge: edges) {
    		// 下一个点的位置
			arr[edge.get(1)]++;
		}
    	List<Integer> res = new ArrayList<Integer>();
    	for (int i = 0; i < n; i++) {
    		// 等于0代表没有点指向这个点，也就是入度为0
			if (arr[i] == 0) {
				res.add(i);
			}
		}
    	return res;
    }
	
//    /*
//     * LCP 07. 传递信息
//     */
//    int n,k;
//    int path = 0;
//    List<ArrayList<Integer>> list;
//    
//    public int numWays(int n, int[][] relation, int k) {
//    	this.n = n;
//    	this.k = k;
//    	list = new ArrayList<ArrayList<Integer>>();
//    	// 填充新变量list
//    	for (int i = 0; i < n; i++) {
//    		list.add(new ArrayList<Integer>());
//		}
//    	for (int i = 0; i < relation.length; i++) {
//			list.get(relation[i][0]).add(relation[i][1]);
//		}
//    	numWaysDfs(0,0);
//    	return path;
//    }
//
//	private void numWaysDfs(int start, int step) {
//		// 出口，等于k时没找到也要退出
//		if (step == k) {
//            if (start == n - 1){
//			    path++;
//            }
//            return;
//		}
//		// 得到当前start对应的所有边，再继续递归寻找
//		ArrayList<Integer> temp = list.get(start);
//		for (Integer next : temp) {
//			numWaysDfs(next, step + 1);
//		}
//	}
    
    /*
     * 733. 图像渲染
     */
    public int[][] floodFill(int[][] image, int sr, int sc, int color) {
		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[0].length; j++) {
				if (i == sr && j == sc) {
					infect(image, i, j, image[sr][sc], color);
				}
			}
		}
		return image;
	}

	private void infect(int[][] image, int i, int j, int raw, int color) {
		if (i >= image.length || i < 0 || j >= image[0].length || j < 0 || image[i][j] != raw || (image[i][j] == raw && image[i][j] == color)) {
			return;
		}
		image[i][j] = color;
		infect(image, i + 1, j, raw, color);
		infect(image, i - 1, j, raw, color);
		infect(image, i, j + 1, raw, color);
		infect(image, i, j - 1, raw, color);
	}
    
    /*
     * 1460. 通过翻转子数组使两个数组相等
     */
    public boolean canBeEqual(int[] target, int[] arr) {
    	int[] res = new int[1001];
    	for (int i = 0; i < arr.length; i++) {
    		res[target[i]]++;
    		res[arr[i]]--;
		}
    	for (int i = 0; i < res.length; i++) {
			if (res[i] != 0) {
				return false;
			}
		}
    	return true;
    }
    
    /*
     * 994. 腐烂的橘子 DFS
     */
    public int orangesRotting1(int[][] grid) {
    	for (int i = 0; i < grid.length; i++) {
			for (int j = 0; j < grid[0].length; j++) {
				if (grid[i][j] == 2) {
					rotting(grid, i, j, 2);
				}
			}
		}
    	
    	int max = 0;
    	for (int i = 0; i < grid.length; i++) {
			for (int j = 0; j < grid[0].length; j++) {
				if (grid[i][j] == 1) {
					// 还有橘子没被传染到
					return -1;
				}else {
					max = Math.max(max, grid[i][j]);
				}
			}
		}
    	// level初始值为2，所以减2
    	return max == 0 ? 0 : max - 2;
    }

	private void rotting1(int[][] grid, int i, int j, int level) {
		if (i >= grid.length || i < 0 || j >= grid[0].length || j < 0) {
			return;
		}
		// 不为新鲜橘子并且传播次数比当前少的橘子，截停传染
        if (grid[i][j] != 1 && grid[i][j] < level){
            return;
        }

        // 将当前传播次数记录在路径中
		grid[i][j] = level;
		level++;
		rotting1(grid, i + 1, j, level);
		rotting1(grid, i - 1, j, level);
		rotting1(grid, i, j + 1, level);
		rotting1(grid, i, j - 1, level);
	}
	
    /*
     * 994. 腐烂的橘子 BFS
     */
    public int orangesRotting2(int[][] grid) {
    	int n = grid.length;
    	int m = grid[0].length;
    	// 新鲜橘子的数量
    	int count = 0;
    	// 腐烂橘子的位置
    	Queue<int[]> que = new LinkedList<int[]>();
    	for (int r = 0; r < n; r++) {
			for (int c = 0; c < m; c++) {
				if (grid[r][c] == 1) {
					count++;
				}else if(grid[r][c] == 2) {
					que.offer(new int[]{r,c});
				}
			}
		}
    	
    	// 次数/分钟数/轮数
    	int round = 0;
    	// 方向
		int[][] direction = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
    	while (count > 0 && !que.isEmpty()) {
    		round++;
    		int size = que.size();
    		for (int i = 0; i < size; i++) {
				int[] orange = que.poll();
				for (int j = 0; j < direction.length; j++) {
					int r = orange[0] + direction[j][0];
					int c = orange[1] + direction[j][1];
					if (r < n && r >= 0 && c < m && c >= 0 && grid[r][c] == 1) {
						count--;
						grid[r][c] = 2;
						// 将刚腐烂的橘子加入队列
						que.offer(new int[] { r , c });
					}
				}
				/*
				int r = orange[0];
				int c = orange[1];
				if (r - 1 >= 0 && grid[r - 1][c] == 1) {
					count--;
					grid[r - 1][c] = 2;
					// 将刚腐烂的橘子加入队列
					que.offer(new int[] { r - 1, c });
				}
				if (r + 1 < n && grid[r + 1][c] == 1) {
					count--;
					grid[r + 1][c] = 2;
					que.offer(new int[] { r + 1, c });
				}
				if (c - 1 >= 0 && grid[r][c - 1] == 1) {
					count--;
					grid[r][c - 1] = 2;
					que.offer(new int[] { r, c - 1 });
				}
				if (c + 1 < m && grid[r][c + 1] == 1) {
					count--;
					grid[r][c + 1] = 2;
					que.offer(new int[] { r, c + 1 });
				}
				*/
			}
    	}
    	
    	if (count > 0) {
    		return -1;
    	}
    	
    	return round;
    }
	
	
	
	
	
	
    
	/*
	 * 542. 01 矩阵
	 */
	public int[][] updateMatrix(int[][] mat) {
		int n = mat.length;
		int m = mat[0].length;
		Queue<int[]> que = new LinkedList<>();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (mat[i][j] == 0) {
					// 将所有为0的位置加入队列
					que.offer(new int[] { i, j });
				}else {
					// 默认将其他位置设置为最大距离
					mat[i][j] = n + m;
				}
			}
		}
		int[][] direction = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
		
		while (!que.isEmpty()) {
			int[] temp = que.poll();
			for (int i = 0; i < direction.length; i++) {
				int r = temp[0] + direction[i][0];
				int c = temp[1] + direction[i][1];
				// mat[r][c] > mat[temp[0]][temp[1]] + 1
				// 根据队列其他方向上的结果获取距离，队列中存放的位置代表的信息是与0的最近距离
				// 其他方向中与0的最近距离+1就是当前位置距离0的距离
				if (r < n && r >= 0 && c < m && c >= 0 && mat[r][c] > mat[temp[0]][temp[1]] + 1) {
					mat[r][c] = mat[temp[0]][temp[1]] + 1;
					que.offer(new int[] { r, c });
				}
			}
		}
		return mat;
	}

	/*
	 * 841. 钥匙和房间
	 */
	int open = 0;
	public boolean canVisitAllRooms(List<List<Integer>> rooms) {
		boolean[] visited = new boolean[rooms.size()];
		dfs(rooms, 0, visited);
		return open == rooms.size();
	}

	private void dfs(List<List<Integer>> rooms, Integer index, boolean[] visited) {
		List<Integer> room = rooms.get(index);
		if (room == null || visited[index]) {
			return;
		}
        open++;
		visited[index] = true;
		for (int i = 0; i < room.size(); i++) {
			dfs(rooms, room.get(i), visited);
		}
	}
	
	/*
	 * 215. 数组中的第K个最大元素
	 */
    public int findKthLargest(int[] nums, int k) {
		PriorityQueue<Integer> queue = new PriorityQueue<Integer>((a, b) -> b - a);
		for (int i = 0; i < nums.length; i++) {
			queue.add(nums[i]);
		}
		int res = 0;
		for (int i = 0; i < k; i++) {
			res = queue.poll();
		}
		return res;
    }
	
	/*
	 * 347. 前 K 个高频元素
	 */
    public int[] topKFrequent(int[] nums, int k) {
    	HashMap<Integer, Integer> map = new HashMap<Integer,Integer>();
    	for (int i = 0; i < nums.length; i++) {
    		map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
		}
    	// 获取频率键值对
    	Set<Map.Entry<Integer,Integer>> entrySet = map.entrySet();
    	// 小顶堆
    	PriorityQueue<Map.Entry<Integer,Integer>> queue = new PriorityQueue<Map.Entry<Integer,Integer>>((a,b) -> a.getValue() - b.getValue());
    	for (Map.Entry<Integer, Integer> entry : entrySet) {
			queue.offer(entry);
			if (queue.size() > k) {
				// 弹出频率最小的
				queue.poll();
			}
		}
    	int[] res = new int[k];
    	for (int i = k - 1; i >= 0; i--) {
    		// 逆序放入结果集
			res[i] = queue.poll().getKey();
		}
    	return res;
    }
	
	/*
	 * 1254. 统计封闭岛屿的数目
	 */
	int[][] g;
	int ans;

	public int closedIsland(int[][] grid) {
		g = grid;
		for (int i = 0; i < g.length; i++) {
			for (int j = 0; j < g[0].length; j++) {
				if (g[i][j] == 0 && dfs(i, j))
					ans++;
			}
		}
		return ans;
	}

	private boolean dfs(int i, int j) {
		if (i < 0 || i >= g.length || j < 0 || j >= g[0].length)
			return false; // 终止条件1
		if (g[i][j] == 1)
			return true; // 终止条件2
		g[i][j] = 1;
		return dfs(i + 1, j) & dfs(i - 1, j) & dfs(i, j + 1) & dfs(i, j - 1);
	}
	
	/*
	 * 1020. 飞地的数量
	 */
	// 数量
	int enclave = 0;
    public int numEnclaves(int[][] grid) {
    	int n = grid.length;
    	int m = grid[0].length;
    	// 先遍历四个边界,将边界上的1变成0
    	for (int i = 0; i < n; i++) {
			if (grid[i][0] == 1) {
				solveDFS(grid, i, 0);
			}
			if (grid[i][m - 1] == 1) {
				solveDFS(grid, i, m - 1);
			}
		}
    	for (int i = 0; i < m; i++) {
			if (grid[0][i] == 1) {
				solveDFS(grid, 0, i);
			}
			if (grid[n - 1][i] == 1) {
				solveDFS(grid, n - 1, i);
			}
		}
    	// 重置结果数，因为上面已经修改过
    	enclave = 0;
    	for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (grid[i][j] == 1) {
					solveDFS(grid, i, j);
				}
			}
		}
    
    	return enclave;
    }

	private void solveDFS(int[][] grid, int i, int j) {
		if (i < 0 || i == grid.length || j < 0 || j == grid[0].length || grid[i][j] != 1) {
			return;
		}
		grid[i][j] = 0;
		enclave++;
		solveDFS(grid, i + 1, j);
		solveDFS(grid, i - 1, j);
		solveDFS(grid, i, j + 1);
		solveDFS(grid, i, j - 1);
	}
	
    /*
     * 1905. 统计子岛屿
     */
	int islandCount = 0;
	boolean notSub;
    public int countSubIslands(int[][] grid1, int[][] grid2) {
    	// 只遍历grid2,通过grid2的位置判断grid1的位置是否为岛屿
    	int n = grid2.length;
    	int m = grid2[0].length;
    	for (int i = 0; i < n; i++) {
    		for (int j = 0; j < m; j++) {
				if (grid2[i][j] == 1) {
					notSub = false;
					subIslandsDFS(grid1, grid2, i, j);
					if (!notSub) {
						islandCount++;
					}
				}
			}
		}
    	return islandCount;
    }

	private void subIslandsDFS(int[][] grid1, int[][] grid2, int i, int j) {
		if (i < 0 || i == grid1.length || j < 0 || j == grid1[0].length || grid2[i][j] == 0) {
			return;
		}
		// 在grid1的当前位置上并不是陆地，则直接将整块岛标记为不是子岛
		if (grid1[i][j] != 1) {
			notSub = true;
		}
		
		grid2[i][j] = 0;
        subIslandsDFS(grid1, grid2, i + 1, j);
        subIslandsDFS(grid1, grid2, i - 1, j);
        subIslandsDFS(grid1, grid2, i, j + 1);
        subIslandsDFS(grid1, grid2, i, j - 1);
	}
    
	/*
	 * 130. 被围绕的区域
	 */
    public void solve(char[][] board) {
    	int n = board.length;
    	int m = board[0].length;
    	// 记录能到边界上的O的位置
    	List<int[]> temp = new ArrayList<int[]>();
    	// 遍历四个边界，记录能到边界上的O的位置
    	for (int i = 0; i < n; i++) {
			if (board[i][0] == 'O') {
				solveDFS(board, i, 0, temp);
			}
			if (board[i][m - 1] == 'O') {
				solveDFS(board, i, m - 1, temp);
			}
		}
    	
    	for (int i = 0; i < m; i++) {
			if (board[0][i] == 'O') {
				solveDFS(board, 0, i, temp);
			}
			if (board[n - 1][i] == 'O') {
				solveDFS(board, n - 1, i, temp);
			}
		}

    	// 全部设为X，因为已经有O的位置了
        for (int i = 0; i < board.length; i++) {
    		Arrays.fill(board[i], 'X');
		}
    	// 将存储的O存入
    	for (int i = 0; i < temp.size(); i++) {
    		int[] xy = temp.get(i);
			board[xy[0]][xy[1]] = 'O';
		}
    }

	private void solveDFS(char[][] board, int i, int j, List<int[]> temp) {
		if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] == 'X') {
			return;
		}
		temp.add(new int[]{i, j});
        board[i][j] = 'X';
		solveDFS(board, i + 1, j, temp);
		solveDFS(board, i - 1, j, temp);
		solveDFS(board, i, j + 1, temp);
		solveDFS(board, i, j - 1, temp);
	}
    
	/*
	 * 90. 子集 II
	 */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
    	List<List<Integer>> res = new ArrayList<List<Integer>>();
    	backtracking(0,res,new ArrayList<Integer>(),nums,false);
    	return res;
    }

	private void backtracking(int startIndex, List<List<Integer>> res, ArrayList<Integer> path, int[] nums, boolean choosePre) {
		if (startIndex == nums.length) {
			res.add(new ArrayList<Integer>(path));
			return;
		}

		path.add(nums[startIndex]);
		// 没有使用前一个数
		backtracking(startIndex + 1, res, path, nums, false);
		path.remove(path.size() - 1);
		// 没有使用前一个数，且不与前一个数相等
		if (!choosePre && startIndex > 0 && nums[startIndex - 1] == nums[startIndex]) {
			return;
		}
        // 使用了前一个数
        backtracking(startIndex + 1, res, path, nums, true);
	}
    
    /*
     * 1162. 地图分析
     */
	int[][] dirs = new int[][] { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
    public int maxDistance(int[][] grid) {
        Queue<int[]> queue = new LinkedList<>();
        int m = grid.length, n = grid[0].length;
        // 先把所有的陆地都入队
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    queue.offer(new int[] {i, j});
                }
            }
        }

        // 从各个陆地开始，一圈一圈的遍历海洋，最后遍历到的海洋就是离陆地最远的海洋
        boolean hasOcean = false;
        int[] point = null;
        while (!queue.isEmpty()) {
            point = queue.poll();
            int x = point[0], y = point[1];
            // 取出队列的元素，将其四周的海洋入队
            for (int i = 0; i < dirs.length; i++) {
                int newX = x + dirs[i][0];
                int newY = y + dirs[i][1];
                if (newX < 0 || newX >= m || newY < 0 || newY >= n || grid[newX][newY] != 0) {
                    continue;
                }
                grid[newX][newY] = grid[x][y] + 1; // 这里我直接修改了原数组，因此就不需要额外的数组来标志是否访问
                hasOcean = true;
                queue.offer(new int[] {newX, newY});
            }
        }

        // 没有陆地或者没有海洋，返回-1
        if (point == null || !hasOcean) {
            return -1;
        }

    	// 返回最后一次遍历到的海洋的距离
    	// -1是因为陆地的初始值为1
    	return grid[point[0]][point[1]] - 1;
    }
    
    /*
     * 47. 全排列 II
     */
    boolean[] vis;
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> path = new ArrayList<Integer>();
        vis = new boolean[nums.length];
        Arrays.sort(nums);
        backtrack(nums, res, 0, path);
        return res;
    }

    public void backtrack(int[] nums, List<List<Integer>> res, int idx, List<Integer> path) {
        if (idx == nums.length) {
            res.add(new ArrayList<Integer>(path));
            return;
        }
        for (int i = 0; i < nums.length; ++i) {
        	// !vis[i - 1] 的意思是 nums[i - 1] 刚被回溯掉
            if (vis[i] || (i > 0 && nums[i] == nums[i - 1] && !vis[i - 1])) {
                continue;
            }
            path.add(nums[i]);
            vis[i] = true;
            backtrack(nums, res, idx + 1, path);
            vis[i] = false;
            path.remove(idx);
        }
    }
    
    /*
     * 39. 组合总和
     */
    public List<List<Integer>> combinationSum(int[] nums, int target) {
    	List<List<Integer>> res = new ArrayList<List<Integer>>();
    	List<Integer> path = new ArrayList<Integer>();
		Arrays.sort(nums);	
    	dfs(0, nums, res, path, target);
    	return res;
    }

	private void dfs(int i, int[] nums, List<List<Integer>> res, List<Integer> path, int target) {
		if (target < 0) {
			return;
		}
		if (target == 0) {
			res.add(new ArrayList<Integer>(path));
			return;
		}
		for (int j = i; j < nums.length; j++) {
			if (target < 0) {
				break;
			}
			path.add(nums[j]);
			dfs(j, nums, res, path, target - nums[j]);
			path.remove(path.size() - 1);
		}
	}

	public void setZeroes(int[][] matrix) {
    	// 标记当前行列有没有零存在
		int[] col = new int[matrix.length];
		int[] row = new int[matrix[0].length];
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {
				if (matrix[i][j] == 0) {
					col[i] = 1;
					row[j] = 1;
				}
			}
		}
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {
				if (col[i] == 1) {
					matrix[i][j] = 0;
				}
				if (row[j] == 1) {
					matrix[i][j] = 0;
				}
			}
		}
	}

	public int numIslands(char[][] grid) {
		int n = grid.length;
		int m = grid[0].length;
		int res = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (grid[i][j] == '1') {
					infectIslands(grid, i, j, n, m);
					res++;
				}
			}
		}
		return res;
	}

	private void infectIslands(char[][] grid, int i, int j, int n, int m) {
		if (i < 0 || j < 0 || i >= n || j >= m || grid[i][j] != '1') {
			return;
		}
		grid[i][j] = '2';
		infectIslands(grid, i + 1, j, n, m);
		infectIslands(grid, i - 1, j, n, m);
		infectIslands(grid, i, j + 1, n, m);
		infectIslands(grid, i, j - 1, n, m);
	}

	public int orangesRotting(int[][] grid) {
		int n = grid.length;
		int m = grid[0].length;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (grid[i][j] == 2) {
					rotting(grid, i, j, 2);
				}
			}
		}
		int max = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (grid[i][j] == 1) {
					return -1;
				}
				max = Math.max(max, grid[i][j]);
			}
		}
		return max == 0 ? 0 : max - 2;
	}

	private void rotting(int[][] grid, int i, int j, int level) {
		if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length) {
			return;
		}
		if (grid[i][j] != 1 && grid[i][j] < level) {
			return;
		}
		grid[i][j] = level;
		level++;
		rotting(grid, i + 1, j, level);
		rotting(grid, i - 1, j, level);
		rotting(grid, i, j + 1, level);
		rotting(grid, i, j - 1, level);
	}

	public int orangesRotting21(int[][] grid) {
		int count = 0;
		Queue<int[]> queue = new LinkedList<>();
		int n = grid.length;
		int m = grid[0].length;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (grid[i][j] == 1) {
					count++;
				} else if(grid[i][j] == 2) {
					queue.add(new int[]{i, j});
				}
			}
		}
		int[][] direction = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
		int round = 0;
		while (count > 0 && !queue.isEmpty()) {
			round++;
			int size = queue.size();
			while (size > 0) {
				size--;
				int[] oranges = queue.poll();
				for (int i = 0; i < direction.length; i++) {
					int row = oranges[0] + direction[i][0];
					int col = oranges[1] + direction[i][1];
					if (row >= 0 && row < n && col >= 0 && col < m && grid[row][col] == 1) {
						count--;
						grid[row][col] = 2;
						queue.add(new int[]{row, col});
					}
				}
			}

		}
		if (count > 0) {
			return -1;
		}
		return round;
	}

	public List<Boolean> checkIfPrerequisite(int numCourses, int[][] prerequisites, int[][] queries) {
		int[] inDegree = new int[numCourses];
		HashSet<Integer>[] adj = new HashSet[numCourses];
		for (int[] arr : prerequisites) {
			int cur = arr[0];
			int next = arr[1];
			inDegree[next]++;
			if (adj[cur] == null) {
				adj[cur] = new HashSet<>();
			}
			adj[cur].add(next);
		}
		Queue<Integer> queue = new LinkedList<>();
		for (int i = 0; i < numCourses; i++) {
			if (inDegree[i] == 0) {
				queue.add(i);
			}
		}
		boolean[][] isPre = new boolean[numCourses][numCourses];
		while (!queue.isEmpty()) {
			Integer cur = queue.poll();
			if (adj[cur] == null) {
				continue;
			}
			HashSet<Integer> list = adj[cur];
			for (Integer next : list) {
				// 将这个数可到达的数都设置为true
				isPre[cur][next] = true;
				// cur已经确定是next的先决条件，若已经能够到达cur处，则next处也必定能到达
				// 反之，如果cur还未到达，但next处已经为true，则保留原样
				for (int i = 0; i < numCourses; i++) {
					isPre[i][next] = isPre[i][cur] | isPre[i][next];
				}
				inDegree[next]--;
				if (inDegree[next] == 0) {
					queue.add(next);
				}
			}
		}
		ArrayList<Boolean> res = new ArrayList<>();
		for (int[] query : queries) {
			res.add(isPre[query[0]][query[1]]);
		}
		return res;
	}

	public boolean checkValidGrid(int[][] grid) {
		if (grid[0][0] != 0) {
			return false;
		}
    	// 八个方向
		int[] dx = {-2, -2, -1, -1, 1, 1, 2, 2};
		int[] dy = {-1, 1, -2, 2, -2, 2, -1, 1};
		Queue<int[]> queue = new LinkedList<>();
		queue.offer(new int[]{0, 0});
		int index = 0;
		while (queue.size() > 0) {
			index++;
			int[] arr = queue.poll();
			// 判断八个方向中是否能够有一个是正确的
			for (int i = 0; i < 8; i++) {
				int x = arr[0] + dx[i];
				int y = arr[1] + dy[i];
				if (x < grid.length && y < grid[0].length && x >= 0 && y >= 0 && grid[x][y] == index) {
					if (index == grid.length * grid.length - 1) {
						// 已找到最后一个
						return true;
					}
					queue.add(new int[]{x, y});
					// 已经找到一个落点，直接跳出循环
					break;
				}
			}
		}
		return false;
	}

	public List<List<Integer>> queensAttacktheKing(int[][] queens, int[] king) {
		// 八个方向
		int[] dx = {-1, 1, 1, 0, -1, 0, 1, -1};
		int[] dy = {-1, 1, 0, 1, 0, -1, -1, 1};
		boolean[][] isQueen = new boolean[8][8];
		for (int[] queen : queens) {
			isQueen[queen[0]][queen[1]] = true;
		}
		ArrayList<List<Integer>> res = new ArrayList<>();
		for (int i = 0; i < dx.length; i++) {
			int x = dx[i] + king[0];
			int y = dy[i] + king[1];
			while (x < 8 && y < 8 && x >= 0 && y >= 0) {
				if (isQueen[x][y]) {
					// 找到第一个能够到达的皇后
					res.add(ListUtil.of(x, y));
					break;
				}
				// 继续往后寻找
				x += dx[i];
				y += dy[i];
			}
		}
		return res;
	}
}
