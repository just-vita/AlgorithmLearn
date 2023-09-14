package top.vita.graph;

import cn.hutool.core.collection.ListUtil;

import java.util.*;

@SuppressWarnings("all")
public class GraphQuestion {

	int n;
	int[] father;

	// ���鼯��ʼ��
	public GraphQuestion() {
		this.n = 1005;
		this.father = new int[n];

		for (int i = 0; i < father.length; i++) {
			father[i] = i;
		}
	}

	// ���鼯��Ѱ���Ĺ���
	public int find(int u) {
		if (u == father[u]) {
			return u;
		}
		father[u] = find(father[u]);
		return father[u];
	}

	// ��v->u �����߼��벢�鼯
	public void union(int u, int v) {
		u = find(u);
		v = find(v);
		if (u == v) {
			return;
		}
		father[u] = v;
	}

	// �ж� u �� v �Ƿ��ҵ�ͬһ���������ݱ�������ȥ��ͬ���ı�
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
		// ��װ���鼯������
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
	 * 1791. �ҳ�����ͼ�����Ľڵ�
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
     * 797. ���п��ܵ�·��
     */
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
    	// ����·�����Ǵ�0��ʼ
    	path.add(0);
    	subIslandsDFS(graph, 0);
    	return res;
    }

	List<List<Integer>> res = new ArrayList<List<Integer>>();
    List<Integer> path = new ArrayList<Integer>();
    
    private void subIslandsDFS(int[][] graph, int x) {
    	// ���������ˣ���ǰ·����������
    	if (x == graph.length - 1) {
    		// ע��pathҪתΪArrayList
    		res.add(new ArrayList<>(path));
    		return;
    	}
    	// ����ǰ�ڵ���յ�Ϊֹһֱ����
    	for (int i = 0; i < graph[x].length; i++) {
			path.add(graph[x][i]);
			subIslandsDFS(graph, graph[x][i]);
			path.remove(path.size() - 1);
		}
    }
    
    /*
     * 1557. ���Ե������е�����ٵ���Ŀ
     */
    public List<Integer> findSmallestSetOfVertices(int n, List<List<Integer>> edges) {
    	// �����Ϊ0�ĵ�
    	int[] arr = new int[n];
    	for (List<Integer> edge: edges) {
    		// ��һ�����λ��
			arr[edge.get(1)]++;
		}
    	List<Integer> res = new ArrayList<Integer>();
    	for (int i = 0; i < n; i++) {
    		// ����0����û�е�ָ������㣬Ҳ�������Ϊ0
			if (arr[i] == 0) {
				res.add(i);
			}
		}
    	return res;
    }
	
//    /*
//     * LCP 07. ������Ϣ
//     */
//    int n,k;
//    int path = 0;
//    List<ArrayList<Integer>> list;
//    
//    public int numWays(int n, int[][] relation, int k) {
//    	this.n = n;
//    	this.k = k;
//    	list = new ArrayList<ArrayList<Integer>>();
//    	// ����±���list
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
//		// ���ڣ�����kʱû�ҵ�ҲҪ�˳�
//		if (step == k) {
//            if (start == n - 1){
//			    path++;
//            }
//            return;
//		}
//		// �õ���ǰstart��Ӧ�����бߣ��ټ����ݹ�Ѱ��
//		ArrayList<Integer> temp = list.get(start);
//		for (Integer next : temp) {
//			numWaysDfs(next, step + 1);
//		}
//	}
    
    /*
     * 733. ͼ����Ⱦ
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
     * 1460. ͨ����ת������ʹ�����������
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
     * 994. ���õ����� DFS
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
					// ��������û����Ⱦ��
					return -1;
				}else {
					max = Math.max(max, grid[i][j]);
				}
			}
		}
    	// level��ʼֵΪ2�����Լ�2
    	return max == 0 ? 0 : max - 2;
    }

	private void rotting1(int[][] grid, int i, int j, int level) {
		if (i >= grid.length || i < 0 || j >= grid[0].length || j < 0) {
			return;
		}
		// ��Ϊ�������Ӳ��Ҵ��������ȵ�ǰ�ٵ����ӣ���ͣ��Ⱦ
        if (grid[i][j] != 1 && grid[i][j] < level){
            return;
        }

        // ����ǰ����������¼��·����
		grid[i][j] = level;
		level++;
		rotting1(grid, i + 1, j, level);
		rotting1(grid, i - 1, j, level);
		rotting1(grid, i, j + 1, level);
		rotting1(grid, i, j - 1, level);
	}
	
    /*
     * 994. ���õ����� BFS
     */
    public int orangesRotting2(int[][] grid) {
    	int n = grid.length;
    	int m = grid[0].length;
    	// �������ӵ�����
    	int count = 0;
    	// �������ӵ�λ��
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
    	
    	// ����/������/����
    	int round = 0;
    	// ����
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
						// ���ո��õ����Ӽ������
						que.offer(new int[] { r , c });
					}
				}
				/*
				int r = orange[0];
				int c = orange[1];
				if (r - 1 >= 0 && grid[r - 1][c] == 1) {
					count--;
					grid[r - 1][c] = 2;
					// ���ո��õ����Ӽ������
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
	 * 542. 01 ����
	 */
	public int[][] updateMatrix(int[][] mat) {
		int n = mat.length;
		int m = mat[0].length;
		Queue<int[]> que = new LinkedList<>();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (mat[i][j] == 0) {
					// ������Ϊ0��λ�ü������
					que.offer(new int[] { i, j });
				}else {
					// Ĭ�Ͻ�����λ������Ϊ������
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
				// ���ݶ������������ϵĽ����ȡ���룬�����д�ŵ�λ�ô������Ϣ����0���������
				// ������������0���������+1���ǵ�ǰλ�þ���0�ľ���
				if (r < n && r >= 0 && c < m && c >= 0 && mat[r][c] > mat[temp[0]][temp[1]] + 1) {
					mat[r][c] = mat[temp[0]][temp[1]] + 1;
					que.offer(new int[] { r, c });
				}
			}
		}
		return mat;
	}

	/*
	 * 841. Կ�׺ͷ���
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
	 * 215. �����еĵ�K�����Ԫ��
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
	 * 347. ǰ K ����ƵԪ��
	 */
    public int[] topKFrequent(int[] nums, int k) {
    	HashMap<Integer, Integer> map = new HashMap<Integer,Integer>();
    	for (int i = 0; i < nums.length; i++) {
    		map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
		}
    	// ��ȡƵ�ʼ�ֵ��
    	Set<Map.Entry<Integer,Integer>> entrySet = map.entrySet();
    	// С����
    	PriorityQueue<Map.Entry<Integer,Integer>> queue = new PriorityQueue<Map.Entry<Integer,Integer>>((a,b) -> a.getValue() - b.getValue());
    	for (Map.Entry<Integer, Integer> entry : entrySet) {
			queue.offer(entry);
			if (queue.size() > k) {
				// ����Ƶ����С��
				queue.poll();
			}
		}
    	int[] res = new int[k];
    	for (int i = k - 1; i >= 0; i--) {
    		// �����������
			res[i] = queue.poll().getKey();
		}
    	return res;
    }
	
	/*
	 * 1254. ͳ�Ʒ�յ������Ŀ
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
			return false; // ��ֹ����1
		if (g[i][j] == 1)
			return true; // ��ֹ����2
		g[i][j] = 1;
		return dfs(i + 1, j) & dfs(i - 1, j) & dfs(i, j + 1) & dfs(i, j - 1);
	}
	
	/*
	 * 1020. �ɵص�����
	 */
	// ����
	int enclave = 0;
    public int numEnclaves(int[][] grid) {
    	int n = grid.length;
    	int m = grid[0].length;
    	// �ȱ����ĸ��߽�,���߽��ϵ�1���0
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
    	// ���ý��������Ϊ�����Ѿ��޸Ĺ�
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
     * 1905. ͳ���ӵ���
     */
	int islandCount = 0;
	boolean notSub;
    public int countSubIslands(int[][] grid1, int[][] grid2) {
    	// ֻ����grid2,ͨ��grid2��λ���ж�grid1��λ���Ƿ�Ϊ����
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
		// ��grid1�ĵ�ǰλ���ϲ�����½�أ���ֱ�ӽ����鵺���Ϊ�����ӵ�
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
	 * 130. ��Χ�Ƶ�����
	 */
    public void solve(char[][] board) {
    	int n = board.length;
    	int m = board[0].length;
    	// ��¼�ܵ��߽��ϵ�O��λ��
    	List<int[]> temp = new ArrayList<int[]>();
    	// �����ĸ��߽磬��¼�ܵ��߽��ϵ�O��λ��
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

    	// ȫ����ΪX����Ϊ�Ѿ���O��λ����
        for (int i = 0; i < board.length; i++) {
    		Arrays.fill(board[i], 'X');
		}
    	// ���洢��O����
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
	 * 90. �Ӽ� II
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
		// û��ʹ��ǰһ����
		backtracking(startIndex + 1, res, path, nums, false);
		path.remove(path.size() - 1);
		// û��ʹ��ǰһ�������Ҳ���ǰһ�������
		if (!choosePre && startIndex > 0 && nums[startIndex - 1] == nums[startIndex]) {
			return;
		}
        // ʹ����ǰһ����
        backtracking(startIndex + 1, res, path, nums, true);
	}
    
    /*
     * 1162. ��ͼ����
     */
	int[][] dirs = new int[][] { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
    public int maxDistance(int[][] grid) {
        Queue<int[]> queue = new LinkedList<>();
        int m = grid.length, n = grid[0].length;
        // �Ȱ����е�½�ض����
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    queue.offer(new int[] {i, j});
                }
            }
        }

        // �Ӹ���½�ؿ�ʼ��һȦһȦ�ı����������������ĺ��������½����Զ�ĺ���
        boolean hasOcean = false;
        int[] point = null;
        while (!queue.isEmpty()) {
            point = queue.poll();
            int x = point[0], y = point[1];
            // ȡ�����е�Ԫ�أ��������ܵĺ������
            for (int i = 0; i < dirs.length; i++) {
                int newX = x + dirs[i][0];
                int newY = y + dirs[i][1];
                if (newX < 0 || newX >= m || newY < 0 || newY >= n || grid[newX][newY] != 0) {
                    continue;
                }
                grid[newX][newY] = grid[x][y] + 1; // ������ֱ���޸���ԭ���飬��˾Ͳ���Ҫ�������������־�Ƿ����
                hasOcean = true;
                queue.offer(new int[] {newX, newY});
            }
        }

        // û��½�ػ���û�к��󣬷���-1
        if (point == null || !hasOcean) {
            return -1;
        }

    	// �������һ�α������ĺ���ľ���
    	// -1����Ϊ½�صĳ�ʼֵΪ1
    	return grid[point[0]][point[1]] - 1;
    }
    
    /*
     * 47. ȫ���� II
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
        	// !vis[i - 1] ����˼�� nums[i - 1] �ձ����ݵ�
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
     * 39. ����ܺ�
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
    	// ��ǵ�ǰ������û�������
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
				// ��������ɵ������������Ϊtrue
				isPre[cur][next] = true;
				// cur�Ѿ�ȷ����next���Ⱦ����������Ѿ��ܹ�����cur������next��Ҳ�ض��ܵ���
				// ��֮�����cur��δ�����next���Ѿ�Ϊtrue������ԭ��
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
    	// �˸�����
		int[] dx = {-2, -2, -1, -1, 1, 1, 2, 2};
		int[] dy = {-1, 1, -2, 2, -2, 2, -1, 1};
		Queue<int[]> queue = new LinkedList<>();
		queue.offer(new int[]{0, 0});
		int index = 0;
		while (queue.size() > 0) {
			index++;
			int[] arr = queue.poll();
			// �жϰ˸��������Ƿ��ܹ���һ������ȷ��
			for (int i = 0; i < 8; i++) {
				int x = arr[0] + dx[i];
				int y = arr[1] + dy[i];
				if (x < grid.length && y < grid[0].length && x >= 0 && y >= 0 && grid[x][y] == index) {
					if (index == grid.length * grid.length - 1) {
						// ���ҵ����һ��
						return true;
					}
					queue.add(new int[]{x, y});
					// �Ѿ��ҵ�һ����㣬ֱ������ѭ��
					break;
				}
			}
		}
		return false;
	}

	public List<List<Integer>> queensAttacktheKing(int[][] queens, int[] king) {
		// �˸�����
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
					// �ҵ���һ���ܹ�����Ļʺ�
					res.add(ListUtil.of(x, y));
					break;
				}
				// ��������Ѱ��
				x += dx[i];
				y += dy[i];
			}
		}
		return res;
	}
}
