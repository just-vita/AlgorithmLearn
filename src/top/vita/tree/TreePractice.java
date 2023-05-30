package top.vita.tree;

import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Stack;



public class TreePractice {

	public static void main(String[] args) {

	}
	
	 public List<Integer> preorderTraversal(TreeNode root) {
		 List<Integer> res = new ArrayList<Integer>();
		 if (root == null) {
			 return res;
		 }
		 Stack<TreeNode> stack = new Stack<TreeNode>();
		 stack.push(root);
		 while (!stack.isEmpty()) {
			 TreeNode node = stack.pop();
			 res.add(node.val);
			 if (node.right != null) {
				 stack.push(node.right);
			 }
			 if (node.left != null) {
				 stack.push(node.left);
			 }
		 }
		 return res;
	 }
	 
	public List<Integer> inorderTraversal(TreeNode root) {
		 List<Integer> res = new ArrayList<Integer>();
		 if (root == null) {
			 return res;
		 }
		 Stack<TreeNode> stack = new Stack<TreeNode>();
		 TreeNode cur = root;
		 while (cur != null || !stack.isEmpty()) {
			 if (cur != null) {
				 stack.push(cur);
				 cur = cur.left;
			 }else {
				 cur = stack.pop();
				 res.add(cur.val);
				 cur = cur.right;
			 }
		 }
		 return res;
	}
	
	 public List<List<Integer>> levelOrder(TreeNode root) {
		if (root == null)
			return new ArrayList();
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		// 将第一层加入
		que.offer(root);
		while (!que.isEmpty()) {
			// 创建保存一层数的容器
			List<Integer> item = new ArrayList<Integer>();
			int size = que.size();
			while (size > 0) {
				// 取出一个数
				TreeNode tempNode = que.poll();
				item.add(tempNode.val);
				if (tempNode.left != null)
					// 将左节点加入队列
					que.offer(tempNode.left);
				if (tempNode.right != null)
					// 将右节点加入队列
					que.offer(tempNode.right);
				// 循环一次，队列长度减少一次
				size--;
			}
			// 加入这一层的数
			res.add(item);
		}
		return res;
	 }
	 
	public List<Integer> rightSideView(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		Deque<TreeNode> que = new LinkedList<>();
		if (root == null) return res;
		que.addFirst(root);
		while (!que.isEmpty()) {
			int size = que.size();
			for (int i = 0; i < size; i++) {
				TreeNode node = que.pollFirst();
				if (node.left != null) que.offer(node.left);
				if (node.right != null) que.offer(node.right);
				// 队列最后的数，即这一层树最右边的数
				if (i == size - 1) res.add(node.val);
			}
		}
		return res;
	}

    public List<Double> averageOfLevels(TreeNode root) {
		List<Double> res = new ArrayList<Double>();
		Queue<TreeNode> que = new LinkedList<>();
    	if (root == null) return res;
		que.offer(root);
		while (!que.isEmpty()) {
			int size = que.size();
			// 总和
			double leverSum = 0.0;
			for (int i = 0; i < size; i++) {
				TreeNode node = que.poll();
				// 累加这层的和
				leverSum += node.val;
				if (node.left != null) que.offer(node.left);
				if (node.right != null) que.offer(node.right);
			}
			// 计算平均值加入结果集
			res.add(leverSum / size);
    	}
    	return res;
    }
    
    public List<List<Integer>> levelOrder(Node root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        Queue<Node> que = new LinkedList<Node>();
        if (root == null) return res;
        
		que.offer(root);
		while (!que.isEmpty()) {
			int size = que.size();
			List<Integer> levelList = new ArrayList<Integer>();
			for (int i = 0; i < size; i++) {
				Node node = que.poll();
				levelList.add(node.val);
				// 处理子节点
				List<Node> children = node.children;
				if (children == null || children.size() == 0) continue;
				for (Node child : children) {
					if (child != null) que.offer(child);	
				}
			}
        	res.add(levelList);
        }
        return res;
    }
    
    public List<Integer> largestValues(TreeNode root) {
    	List<Integer> res = new ArrayList<Integer>();
    	Queue<TreeNode> que = new LinkedList<TreeNode>();
    	if (root == null) return res;
		que.offer(root);
		while (!que.isEmpty()) {
			int size = que.size();
			// 初始化为最小值
			int max = Integer.MIN_VALUE;
			for (int i = 0; i < size; i++) {
				TreeNode node = que.poll();
				max = Math.max(max, node.val);
				if (node.left != null) que.offer(node.left);
				if (node.right != null) que.offer(node.right);
			}
			res.add(max);
		}
		return res;
    }
    
    public int maxDepth(TreeNode root) {
    	if (root == null) return 0;
    	Queue<TreeNode> que = new LinkedList<>();
    	que.offer(root);
    	int depth = 0;
    	while (!que.isEmpty()) {
    		int size = que.size();
    		while (size > 0) {
    			TreeNode node = que.poll();
				if (node.left != null) que.offer(node.left);
				if (node.right != null) que.offer(node.right);
				size --;
    		}
    		depth++;
    	}
    	return depth;
    }
    
    public int minDepth(TreeNode root) {
    	if (root == null) return 0;
    	Queue<TreeNode> que = new LinkedList<TreeNode>();
    	que.offer(root);
    	int depth = 0;
    	while (!que.isEmpty()) {
    		int size = que.size();
    		depth++;
    		TreeNode cur = null;
    		for (int i = 0; i < size; i++) {
				cur = que.poll();
				// 左右节点都为空则为叶子节点，直接返回
				if (cur.left == null && cur.right == null) return depth;
				if (cur.left != null) que.offer(cur.left);
				if (cur.right != null) que.offer(cur.right);
    		}
    	}
    	return depth;
    }
    
    public boolean isSymmetric(TreeNode root) {
    	return compare(root.left, root.right);
    }
    
    public boolean compare(TreeNode left,TreeNode right) {
    	if (left == null && right == null) return true;
    	if (left == null || right == null || left.val != right.val) return false;
    	// 比较外侧
    	boolean outside = compare(left.left, right.right);
    	// 比较内侧
    	boolean inside = compare(left.right, right.left);
    	return outside && inside;
    }
    
    /*
	 * 222. 完全二叉树的节点个数 
	 * 给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。
	 */
    // 迭代法
    public int countNodes(TreeNode root) {
    	if (root == null) return 0;
    	Queue<TreeNode> que = new LinkedList<TreeNode>();
    	que.offer(root);
    	int count = 0;
    	while (!que.isEmpty()) {
    		int size = que.size();
    		for (int i = 0; i < size; i++) {
    			TreeNode node = que.poll();
    			if (node != null) count++;
				if (node.left != null) que.offer(node.left);
				if (node.right != null) que.offer(node.right);
			}
    	}
    	return count;
    }
    
//    // 递归法
//    public int countNodes(TreeNode root) {
//    	if (root == null) return 0;
//    	int leftNum = countNodes(root.left); // 左边
//    	int rightNum = countNodes(root.right); // 右边
//    	int treeNum = leftNum + rightNum + 1; // 中间，加一是因为算上中间节点
//    	return treeNum;
//    }
    
    /*
	 * 110. 平衡二叉树
	 * 给定一个二叉树，判断它是否是高度平衡的二叉树。
	 */
    public boolean isBalanced(TreeNode root) {
    	return getHeight(root) != -1; 
    }

	private int getHeight(TreeNode root) {
		if (root == null) return 0;
		int leftHeight = getHeight(root.left);
		if (leftHeight == -1) return -1;
		int rightHeight = getHeight(root.right);
		if (rightHeight == -1) return -1;
		// 左右子树高度差大于1，return -1表示已经不是平衡树了
		if (Math.abs(leftHeight - rightHeight) > 1) return -1;
		return Math.max(leftHeight, rightHeight) + 1;
	}
	
	/*
	 * 257. 二叉树的所有路径 
	 * 给你一个二叉树的根节点 root ，按 任意顺序 ，
	 * 返回所有从根节点到叶子节点的路径。
	 */
    public List<String> binaryTreePaths(TreeNode root) {
    	List<String> res = new ArrayList<>();
    	if (root == null) return res;
    	List<Integer> paths = new ArrayList<>();
    	traversal(root,paths,res);
    	return res;
    }

	private void traversal(TreeNode root, List<Integer> paths, List<String> res) {
		// 前序遍历，先将当前的根节点加入
		paths.add(root.val);
		// 是叶子节点
		if (root.left == null && root.right == null) {
			StringBuilder sb = new StringBuilder();
			// 加入除最后一个的，因为要带上符号
			for (int i = 0; i < paths.size() - 1; i++) {
				sb.append(paths.get(i)).append("->");
			}
			// 将最后的数值加入
			sb.append(paths.get(paths.size() - 1));
			res.add(sb.toString());
			return;
		}
		if (root.left != null) {
			traversal(root.left, paths, res);
			paths.remove(paths.size() - 1);
		}
		if (root.right != null) {
			traversal(root.right, paths, res);
			paths.remove(paths.size() - 1);
		}
	}
	
	/*
	 * 404. 左叶子之和 
	 * 给定二叉树的根节点 root ，返回所有左叶子之和。
	 */
    public int sumOfLeftLeaves(TreeNode root) {
    	if(root == null) return 0;
    	int res = 0;
    	// 当前左节点是叶子节点
    	if (root.left != null && root.left.left == null && root.left.right == null) {
    		res += root.left.val;
    	}
    	return sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right) + res;
    }
    
    /*
	 * 513. 找树左下角的值 
	 * 给定一个二叉树的 根节点 root，
	 * 请找出该二叉树的 最底层 最左边 节点的值。
	 * 假设二叉树中至少有一个节点。
	 */
    private int Deep = -1;
    private int value = 0;
    public int findBottomLeftValue(TreeNode root) {
    	value = root.val;
    	findLeftValue(root, 0);
    	return value;
    }

	private void findLeftValue(TreeNode root, int deep) {
		if (root == null) return;
		// 使用前序遍历，先遍历左子树
		// 为叶子节点
		if (root.left == null && root.right == null) {
			if (deep > Deep) {
				// 更新最大深度和左节点值
				Deep = deep;
				value = root.val;
			}
		}
		if (root.left != null) {
			findLeftValue(root.left, deep + 1);
		}
		if (root.right != null) {
			findLeftValue(root.right, deep + 1);
		}
	}
	
	// 使用迭代法
    public int findBottomLeftValue2(TreeNode root) {
    	if (root == null) return 0;
    	int res = 0;
    	Queue<TreeNode> que = new LinkedList<>();
    	que.offer(root);
    	while (!que.isEmpty()) {
    		int size = que.size();
    		for (int i = 0; i < size; i++) {
    			TreeNode node = que.poll();
    			// 将每层的第一个节点作为结果
				if (i == 0) {
					res = node.val;
				}
				if (node.left != null) que.offer(node.left);
				if (node.right != null) que.offer(node.right);
			}
    	}
    	return res;
    }
    
    /*
	 *  112. 路径总和
	 *  给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。
	 *  判断该树中是否存在 根节点到叶子节点
	 *  的路径，这条路径上所有节点值相加等于目标和 targetSum 。
	 *  如果存在，返回 true ；否则，返回 false 。
	 */
    public boolean hasPathSum(TreeNode root, int targetSum) {
    	if (root == null) return false;
    	return traversal(root, targetSum - root.val);
    }
    
    private boolean traversal(TreeNode cur, int count) {
    	// 是叶子节点，且和为targetSum（count是逐渐减少的）
    	if (cur.left == null && cur.right == null && count == 0) return true;
    	// 是叶子节点，但和部位targetSum，直接返回false
    	if (cur.left == null && cur.right == null) return false;
    	
		if (cur.left != null) {
			// count - cur.left.val 有回溯作用，因为执行后的count的值并没有变
			if (traversal(cur.left, count - cur.left.val))
				return true;
		}
		if (cur.right != null) {
			if (traversal(cur.right, count - cur.right.val))
				return true;
		}

    	return false;
    }
    
    /*
	 * 113. 路径总和 II 
	 * 给你二叉树的根节点 root 和一个整数目标和 targetSum ，
	 * 找出所有 从根节点到叶子节点
	 * 路径总和等于给定目标和的路径。
	 */
    List<List<Integer>> res;
    LinkedList<Integer> path;
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
		res = new ArrayList<List<Integer>>();
		path = new LinkedList<Integer>();
		traversal2(root, targetSum);
		return res;
    }

	private void traversal2(TreeNode node, int count) {
		if (node == null) return;
		// 将节点加入路径列表中
		path.offer(node.val);
		count -= node.val;
		if (node.left == null & node.right == null && count == 0) {
			res.add(new LinkedList<Integer>(path));
		}
		traversal2(node.left, count);
		traversal2(node.right, count);
		// 回溯，如果和不等于目标和，则将最后加入的节点删除
		path.removeLast();
	}
    
	/*
	 * 106. 从中序与后序遍历序列构造二叉树 
	 * 给定两个整数数组 inorder 和 postorder ，其中 inorder 是二叉树的中序遍历，
	 * postorder 是同一棵树的后序遍历，请你构造并返回这颗 二叉树 。
	 */
    public TreeNode buildTree2(int[] inorder, int[] postorder) {
    	return buildTree1(inorder, 0 , inorder.length,postorder,0,postorder.length);
    }
    
    private TreeNode buildTree1(int[] inorder, int inLeft, int inRight, 
    		int[] postorder, int postLeft, int postRight) {
    	
    	if (inRight - inLeft < 1) {
    		return null;
    	}
    	
    	if (inRight - inLeft == 1) {
    		return new TreeNode(inorder[inLeft]);
    	}
    	
    	int rootVal = postorder[postRight - 1];
    	TreeNode root = new TreeNode(rootVal);
    	int rootIndex = 0;
    	for (int i = inLeft; i < inRight; i++) {
			if (inorder[i] == rootVal) {
				rootIndex = i;
				break;
			}
		}
    	
		root.left = buildTree1(inorder, inLeft, rootIndex, 
				postorder, postLeft, postLeft + (rootIndex - inLeft));
		root.right = buildTree1(inorder, rootIndex + 1, inRight,
				postorder, postLeft + (rootIndex - inLeft),postRight - 1);
    	
		return root;
	}

    /*
	 * 108. 将有序数组转换为二叉搜索树 
	 * 给你一个整数数组 nums ，其中元素已经按 升序 排列，
	 * 请你将其转换为一棵 高度平衡 二叉搜索树。
	 */
    public TreeNode sortedArrayToBST(int[] nums) {
    	return dfs(nums, 0, nums.length);
    }

	private TreeNode dfs(int[] nums, int left ,int right) {
		if (left >= right) return null;
    	int mid = (left + right) / 2;
    	TreeNode root = new TreeNode(nums[mid]);
    	root.left = dfs(nums,left, mid);
    	root.right = dfs(nums, mid + 1, right);
    	return root;
	}
	
	/*
	 * 105. 从前序与中序遍历序列构造二叉树 
	 * 给定两个整数数组 preorder 和 inorder ，
	 * 其中 preorder 是二叉树的先序遍历，
	 * inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。
	 */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return pre_order(0, inorder.length - 1, 0, inorder.length - 1, preorder, inorder);
    }
    
    TreeNode pre_order(int leftpre, int rightpre, int leftin, int rightin, int[] pre, int[] in) {
        if (leftpre > rightpre || leftin > rightin) return null;
        TreeNode root = new TreeNode(pre[leftpre]);
        int rootin = leftin;
        // 找到root节点在inorder中的位置
        while (rootin <= rightin && in[rootin] != pre[leftpre]) rootin++;
        // 节点个数
        int left = rootin - leftin;
        // 从 左边 + 1 到 左边 + 节点个数，从 左边 到 root - 1（左子树）
        root.left = pre_order(leftpre + 1, leftpre + left, leftin, rootin - 1, pre, in);
        // 从 左边 + 1 + 节点个数 到 前序右边，从 root + 1 到 中序右边（右子树）
        root.right = pre_order(leftpre + left + 1, rightpre, rootin + 1, rightin, pre, in);
        return root;
    }
    
	/*
	 * 103. 二叉树的锯齿形层序遍历 
	 * 给你二叉树的根节点 root ，返回其节点值的 锯齿形层序遍历
	 */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
    	ArrayList<List<Integer>> res = new ArrayList<List<Integer>>();
    	traversal(root, res, 0);
    	return res;
    }

	private void traversal(TreeNode root, ArrayList<List<Integer>> res, int level) {
		if (root == null) return;
		
		// 当level等于数组长度时，添加一个空数组，防止下面get不到
		if (res.size() == level) res.add(new ArrayList<>());
		
		if (level % 2 == 1) {
			// 从头插入
			res.get(level).add(0,root.val);
		}else {
			res.get(level).add(root.val);
		}
		
		traversal(root.left,res,level+1);
		traversal(root.right,res,level+1);
	}
	
    public List<Integer> inorderTraversal1(TreeNode root) {
    	if(root != null){
    		List<Integer> res = new ArrayList<Integer>();
    		Stack<TreeNode> stack = new Stack<TreeNode>();
    		while (!stack.isEmpty() || root != null) {
    			if (root != null) {
    				stack.push(root);
    				root = root.left;
    			}else {
    				root = stack.pop();
    				res.add(root.val);
    				root = root.right;
    			}
    		}
    	}
    	return null;
    }

//    int max = 0;
//    public int maxDepth1(TreeNode root) {
//    	if (root == null) return 0;
//    	dfs(root, 0);
//    	return max;
//    }
//    private void dfs(TreeNode node, int deep) {
//    	if(node == null) return;
//    	deep++;
//    	if (node.left == null && node.right == null) {
//    		if (deep > max) {
//    			max = deep;
//    		}
//    	}
//    	dfs(node.left, deep);
//    	dfs(node.right, deep);
//    }
    
    /*
	 * 剑指 Offer II 045. 二叉树最底层最左边的值 
	 * 给定一个二叉树的 根节点 root，请找出该二叉树的 最底层 最左边 节点的值。
	 */
//    int max = 0;
//    int val = 0;
//    public int findBottomLeftValue1(TreeNode root) {
//    	dfs(root, 0);
//    	return val;
//    }
//
//	private void dfs(TreeNode node, int deep) {
//		if (node == null) return;
//		deep++;
//		if (node.left == null && node.right == null) {
//			if (deep > max) {
//				max = deep;
//				val = node.val;
//			}
//		}
//		dfs(node.left, deep);
//		dfs(node.right, deep);
//	}
    /*
	 * 1302. 层数最深叶子节点的和 
	 * 给你一棵二叉树的根节点 root ，请你返回 层数最深的叶子节点的和
	 */
    int max = 0;
    int sum = 0;
    public int deepestLeavesSum(TreeNode root) {
    	dfs(root, 0);
    	return sum;
    }
	private void dfs(TreeNode node, int deep) {
		if(node == null) return;
		deep++;
		if (node.left == null && node.right == null) {
			if (deep > max) {
				max = deep;
				sum = node.val;
			}else if (deep == max) {
				sum += node.val;
			}
		}
		dfs(node.left, deep);
		dfs(node.right, deep);
	}
	
	/*
	 * 617. 合并二叉树
	 */
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
    	if (root1 == null) {
    		return root2;
    	}
    	if (root2 == null) {
    		return root1;
    	}
    	root1.val += root2.val;
    	root1.left = mergeTrees(root1.left, root2.left);
    	root1.right = mergeTrees(root1.right, root2.right);
    	return root1;
    }
    
    /*
	 * 98. 验证二叉搜索树 
	 * 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
	 */
    Long preVal = Long.MIN_VALUE;
    public boolean isValidBST(TreeNode root) {
		if (root == null) {
			return true;
		}
		// 如果左树无效则直接返回
		boolean isLeftBST = isValidBST(root.left);
		if (!isLeftBST) {
			return false;
		}
		// 如果不是升序则直接返回
		if (root.val <= preVal) {
			return false;
		} else {
			preVal = (long) root.val;
		}
		return isValidBST(root.right);
    }
    
    /*
	 * 958. 二叉树的完全性检验
	 *  给定一个二叉树的 root ，确定它是否是一个 完全二叉树 。
	 */
    public boolean isCompleteTree(TreeNode root) {
    	LinkedList<TreeNode> que = new LinkedList<TreeNode>();
    	// 发现叶子节点的标记
    	boolean leaf = false;
    	TreeNode l = null;
    	TreeNode r = null;
    	que.push(root);
    	while (!que.isEmpty()) {
    		TreeNode head = que.poll();
    		l = head.left;
    		r = head.right;
    		// 标记触发后，只要不是叶子节点就返回false
    		// 且左子节点为空时右子节点不为空则代表不是CBT
    		if (
    				(leaf && (l != null || r != null)) 
    				
    				|| 
    				
    				(l == null && r != null)
				) {
    			return false;
    		}
    		
    		if (l != null) que.add(l);
    		if (r != null) que.add(r);
    		
    		// 发现第一个null时触发标记，代表之后的子树如果不是叶子节点则不是CBT
    		if (l == null || r == null) {
    			leaf = true;
    		}
    	}
    	return true;
    }
    
	/*
	 * 剑指 Offer 55 - II. 平衡二叉树
	 */
    public boolean isBalanced1(TreeNode root) {
    	return process(root).isBalance;
    }
    
    class Return{
    	boolean isBalance;
    	int height;
		public Return(boolean isBalance, int height) {
			this.isBalance = isBalance;
			this.height = height;
		}
    }

	private Return process(TreeNode root) {
		if (root == null) return new Return(true, 0);
		
		Return leftR = process(root.left);
		Return rightR = process(root.right);
		
		int height = Math.max(leftR.height, rightR.height) + 1;
		boolean isBalance = 
				(leftR.isBalance && rightR.isBalance) 
				&& 
				(leftR.height - rightR.height < 2);
		
		return new Return(isBalance,height);
	}
    
	public boolean isValidBST1(TreeNode root) {
		return checkBST(root).isBST;
	}

	class ReturnData{
		boolean isBST;
		int max;
		int min;
		public ReturnData(boolean isBST, int max, int min) {
			this.isBST = isBST;
			this.max = max;
			this.min = min;
		}
	}
    
	private ReturnData checkBST(TreeNode root) {
		if (root == null) return null;
		
		ReturnData left = checkBST(root.left);
		ReturnData right = checkBST(root.right);

		int min = root.val;
		int max = root.val;
		if (left != null) {
			min = Math.min(left.min, min);
			max = Math.max(left.max, max);
		}
		if (right != null) {
			min = Math.min(right.min, min);
			max = Math.max(right.max, max);
		}
		
		boolean isBST = false;
		
		if (
				(left != null ? (left.isBST && left.max < root.val) : true)
				&&
				(right != null ? (right.isBST && right.min > root.val) : true)
			) {
			isBST = true;
		}

		return new ReturnData(isBST, max, min);
	}
    
    public boolean isFullBinaryTree(TreeNode root) {
    	if (root == null) return true;
    	TreeInfo info = isF(root);
    	// 2^l-1 --> 1 << height - 1
    	return info.nodes == (1 << info.height - 1); 
    }
    
    class TreeInfo{
    	int height;
    	int nodes;
		public TreeInfo(int height, int nodes) {
			super();
			this.height = height;
			this.nodes = nodes;
		}
    }
    
    public TreeInfo isF(TreeNode root) {
    	if (root == null) return new TreeInfo(0, 0);
    	
    	TreeInfo leftInfo = isF(root.left);
    	TreeInfo rightInfo = isF(root.right);
    	
    	int height = Math.max(leftInfo.height,rightInfo.height) + 1;
    	int nodes = Math.max(leftInfo.nodes,rightInfo.nodes) + 1;
    	
    	return new TreeInfo(height,nodes);
    }
    
    /*
     * 剑指 Offer 68 - II. 二叉树的最近公共祖先
     * 236. 二叉树的最近公共祖先
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
        	return root;
        }
        
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        
        if (left != null && right != null) {
        	return root;
        }
        
        return left != null ? left : right;
    }
    
    /*
	 * 剑指 Offer II 053. 二叉搜索树中的中序后继 
	 * 给定一棵二叉搜索树和其中的一个节点 p
	 * 找到该节点在树中的中序后继。如果节点没有中序后继，请返回 null
	 */
//    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
//        if (root == null) return null;
//        if (p.right != null) {
//        	return getLeftMost(p);
//        }else {
//        	
//        }
//    }
//
//	private TreeNode getLeftMost(TreeNode node) {
//		if (node == null) return null;
//		while (node.left != null) {
//			node = node.left;
//		}
//		return node;
//	}
    
    /*
     * 226. 翻转二叉树
     */
    public TreeNode invertTree1(TreeNode root) {
    	if (root == null) return null;
    	TreeNode temp = root.left;
    	root.left = root.right;
    	root.right = temp;
    	
    	invertTree1(root.left);
    	invertTree1(root.right);
    	
    	return root;
    }    
    
    /*
     * 701. 二叉搜索树中的插入操作
     */
    public TreeNode insertIntoBST(TreeNode root, int val) {
    	if (root == null){
            return new TreeNode(val);
        }
    	if (root.val > val) {
			root.left = insertIntoBST(root.left, val);
    	}else{
			root.right = insertIntoBST(root.right, val);
    	}
    	return root;
    }
    
    /*
     * 剑指 Offer II 048. 序列化与反序列化二叉树
     */
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) return "#_";
        String str = root.val + "_";
        str += serialize(root.left);
        str += serialize(root.right);
        return str;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        LinkedList<String> que = new LinkedList<String>();
        String[] strs = data.split("_");
        for (int i = 0; i < strs.length; i++) {
			que.add(strs[i]);
		}
        return decode(que);
    }

	private TreeNode decode(LinkedList<String> que) {
		String str = que.poll();
		if(str.equals("#")) return null;
		TreeNode head = new TreeNode(Integer.valueOf(str));
		head.left = decode(que);
		head.right = decode(que);
		return head;
	}
    
	/*
	 * 235. 二叉搜索树的最近公共祖先
	 */
    public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
    	if (p.val < root.val && q.val < root.val) {
    		return lowestCommonAncestor(root.left, p, q);
        }
    	if (p.val > root.val && q.val > root.val) {
    		return lowestCommonAncestor(root.right, p, q);
    	}
    	return root;
    }
    
    /*
     * 654. 最大二叉树
     */
    public TreeNode constructMaximumBinaryTree(int[] nums) {
		return createMaxTree(nums, 0, nums.length - 1);
    }

	private TreeNode createMaxTree(int[] nums, int l, int r) {
		if (l > r) {
			return null;
		}
		int max = findMax(nums,l,r);
		TreeNode root = new TreeNode(nums[max]);
		root.left = createMaxTree(nums, l, max - 1);
		root.right = createMaxTree(nums, max + 1, r);
		return root;
	}

	private int findMax(int[] nums, int l, int r) {
		int max = -1;
		int index = l;
		for (int i = l; i <= r; i++) {
			if (max < nums[i]) {
				max = nums[i];
				index = i;
			}
		}
		return index;
	}
    
    /*
     * 655. 输出二叉树
     */
    public List<List<String>> printTree(TreeNode root) {
        int height = calDepth(root);
        // 矩阵的行数 m 应该等于 height + 1
        int m = height + 1;
        // 2的次方形式
        // 矩阵的列数 n 应该等于 2 ^ height+1 - 1 。
        int n = (1 << (height + 1)) - 1;
        // 任意空单元格都应该包含空字符串 "" 
        List<List<String>> res = new ArrayList<List<String>>();
        for (int i = 0; i < m; i++) {
            List<String> row = new ArrayList<String>();
            for (int j = 0; j < n; j++) {
                row.add("");
            }
            res.add(row);
        }
        // 根节点 需要放置在 顶行 的 正中间 ，对应位置为 res[0][(n-1)/2] 。
        dfs(res, root, 0, (n - 1) / 2, height);
        return res;
    }

    public int calDepth(TreeNode root) {
        int h = 0;
        if (root.left != null) {
            h = Math.max(h, calDepth(root.left) + 1);
        }
        if (root.right != null) {
            h = Math.max(h, calDepth(root.right) + 1);
        }
        return h;
    }

    public void dfs(List<List<String>> res, TreeNode root, int r, int c, int height) {
        res.get(r).set(c, Integer.toString(root.val));
        if (root.left != null) {
        	// 将其左子节点放置在 res[r+1][c-2^height-r-1]
            dfs(res, root.left, r + 1, c - (1 << (height - r - 1)), height);
        }
        if (root.right != null) {
        	// 右子节点放置在 res[r+1][c+2^height-r-1] 。
            dfs(res, root.right, r + 1, c + (1 << (height - r - 1)), height);
        }
    }

    /*
     * 230. 二叉搜索树中第K小的元素
     */
    int ret = 0;int cur = 0;
    public int kthSmallest(TreeNode root, int k) {
    	/*
    	
    	Stack<TreeNode> stack = new Stack<>();
    	while (root != null || !stack.isEmpty()) {
    		while (root != null) {
    			stack.push(root);
    			root = root.left;
    		}
    		
    		root = stack.pop();
    		
    		k--;
    		if (k == 0) {
    			break;
    		}
    		root = root.right;
    	}
    	return root.val;
    	
    	*/
    	
    	inOrder(root,k);
    	return ret;
    }

	private void inOrder(TreeNode root, int k) {
		if (root == null) {
			return;
		}
		inOrder(root.left, k);
		cur++;
		if (cur == k) {
			ret = root.val;
			return;
		}
		inOrder(root.right, k);
	}
    
    /*
     * 100. 相同的树
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
		if ((p == null && q != null) || (p != null && q == null)) {
			return false;
		}
		
		if (p == null && q == null) {
			return true;
		}
		
    	if (p.val != q.val) {
    		return false;
    	}
    	
		return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
    
    /*
     * 997. 找到小镇的法官
     */
    public int findJudge(int n, int[][] trust) {
    	// 入度出度
    	int[] inDegrees = new int[n];
    	int[] outDegrees = new int[n];
    	
    	for (int[] graph : trust) {
			outDegrees[graph[0] - 1]++;
			inDegrees[graph[1] - 1]++;
		}
    	for (int i = 0; i < n; i++) {
    		// 出度为0且入度最大
			if (inDegrees[i] == n - 1 && outDegrees[i] == 0) {
				return i + 1;
			}
		}
    	return -1;
    }
    
    /*
     * 1557. 可以到达所有点的最少点数目
     */
    public List<Integer> findSmallestSetOfVertices(int n, List<List<Integer>> edges) {
    	int[] inDegrees = new int[n];
    	for (List<Integer> edge : edges) {
    		// 修改入度，代表有节点指向它
			inDegrees[edge.get(1)]++;
		}
    	List<Integer> res = new ArrayList<Integer>();
    	for (int i = 0; i < n; i++) {
    		// 入度为0就代表没有节点指向它，要加入结果集
			if (inDegrees[i] == 0) {
				res.add(i);
			}
		}
    	return res;
    }
    
    /*
     * 543. 二叉树的直径
     */
    public int diameterOfBinaryTree(TreeNode root) {
    	return getMaxDiameter(root).maxDistance - 1;
    }
    
    private Info getMaxDiameter(TreeNode root) {
    	if (root == null) {
    		return new Info(0, 0);
    	}
    	
    	// 向左右要信息
    	Info left = getMaxDiameter(root.left);
    	Info right = getMaxDiameter(root.right);
    	
    	// 有三种为最大直径的可能：
    	// 不经过根节点 -> 左树里有最大直径 和 右树里有最大直径
    	// 经过根节点   -> 左树最大高度 + 右树最大高度 + 本身高度
    	int maxDiameter = Math.max(left.maxDistance, Math.max(right.maxDistance, left.height + right.height + 1));
    	// 加上本身高度
    	int height = Math.max(left.height, right.height) + 1;
		return new Info(maxDiameter, height);
	}

	public class Info{
    	int maxDistance;
    	int height;
    	
		public Info(int maxDistance, int height) {
			this.maxDistance = maxDistance;
			this.height = height;
		}
    }
    
    /*
     * 124. 二叉树中的最大路径和
     */
	int max1 = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
    	dfs(root);
        return max;
    }

	private int dfs(TreeNode root) {
		if (root == null) {
			return 0;
		}
		
		// 舍弃负数
		int left = Math.max(0, dfs(root.left));
		int right = Math.max(0, dfs(root.right));
		max = Math.max(max, left + right + root.val);
		
		return Math.max(left, right) + root.val;
	}
    
	/*
	 * 687. 最长同值路径
	 */
	int path1 = 0;
    public int longestUnivaluePath(TreeNode root) {
    	if (root == null) {
    		return 0;
    	}
    	getLongestPath(root);
    	return path1;
    }

	private int getLongestPath(TreeNode root) {
		if (root.left == null && root.right == null) {
			return 0;
		}
		
		int left = root.left != null ? getLongestPath(root.left) : 0;
		int right = root.right != null ? getLongestPath(root.right) : 0;
		
		if (left > 0 && root.left.val != root.val) {
			left = 0;
		}
		if (right > 0 && root.right.val != root.val) {
			right = 0;
		}
		
		path1 = Math.max(left + right, path1);
		return Math.max(left, right) + 1;
	}
    
    /*
     * 337. 打家劫舍 III
     */
    public int rob(TreeNode root) {
    	if (root == null) {
    		return 0;
    	}
    	RobMoney rob = robMoney(root);
    	return Math.max(rob.kuruMaxMoney, rob.konaiMaxMoney);
    }
    
    private RobMoney robMoney(TreeNode root) {
    	if (root == null) {
    		return new RobMoney(0, 0);
    	}
    	if (root.left == null && root.right == null) {
    		return new RobMoney(root.val, 0);
    	}
    	
    	// 选择抢的收益
    	int kuru = root.val;
    	// 选择不抢的最大收益
    	int konai = 0;
    	
    	RobMoney left = robMoney(root.left);
    	// 选择抢，则获取不抢左子树、右子树房屋的收益
    	kuru += left.konaiMaxMoney;
    	// 选择不抢，则获取抢和不抢左子树、右子树房屋的最大收益
    	konai += Math.max(left.konaiMaxMoney, left.kuruMaxMoney);
    	
    	RobMoney right = robMoney(root.right);
    	kuru += right.konaiMaxMoney;
    	konai += Math.max(right.konaiMaxMoney, right.kuruMaxMoney);
    	
		return new RobMoney(kuru, konai);
	}

    public class RobMoney{
    	int kuruMaxMoney;
    	int konaiMaxMoney;
    	
		public RobMoney(int kuruMaxMoney, int konaiMaxMoney) {
			this.kuruMaxMoney = kuruMaxMoney;
			this.konaiMaxMoney = konaiMaxMoney;
		}
    }
    
    /*
     * 129. 求根节点到叶节点数字之和
     */
    int res123 = 0;
    public int sumNumbers(TreeNode root) {
    	List<String> strs = new ArrayList<>();
    	dfs(root, strs, 0);
    	return res123;
    }
	private void dfs(TreeNode root, List<String> strs, int val) {
		if (root == null) {
			return;
		}
		int sum = val * 10 + root.val;
		if (root.left == null && root.right == null) {
			res123 += sum;
		}
		dfs(root.left, strs, sum);
		dfs(root.right, strs, sum);
    }
    
    /*
     * 652. 寻找重复的子树
     */
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
    	HashMap<String,Integer> map = new HashMap<>();
    	List<TreeNode> res = new ArrayList<TreeNode>();
    	serialize(root, map, res);
    	return res;
   }

	private String serialize(TreeNode root, HashMap<String, Integer> map, List<TreeNode> res) {
		String str = "";
		if (root == null) {
			return "#";
		}
		// 获取左右子树信息，序列化成字符串
		str = root.val + "_" + serialize(root.left, map, res) + serialize(root.right, map, res); 
		// 只在第一次遇到相同序列化时加入结果集
		if (map.get(str) == 1) {
			res.add(root);
		}
		map.put(str, map.getOrDefault(str, 0) + 1);
		return str;
	}
    
    /*
     * 572. 另一棵树的子树
     */
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
    	if (root == null && subRoot == null) {
    		return true;
    	}
    	if (root == null && subRoot != null) {
    		return false;
    	}
		return isSameTree1(root, subRoot) || isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);

    }

	private boolean isSameTree1(TreeNode root, TreeNode subRoot) {
		if (root == null && subRoot == null) return true;
    	return root != null && subRoot != null 
    			&& root.val == subRoot.val 
    			&& isSameTree1(root.left, subRoot.left) 
    			&& isSameTree1(root.right, subRoot.right);
	}
	
	/*
	 * 1091. 二进制矩阵中的最短路径
	 */
    public int shortestPathBinaryMatrix(int[][] grid) {
    	 if (grid == null || grid.length == 0 || grid[0].length == 0 || grid[0][0] == 1) {
             return -1;
         }
    	// 定义八个方向
		int[][] dirs = { { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, -1 }, { 0, 1 }, { -1, -1 }, { -1, 0 }, { -1, 1 } };
    	int n = grid.length;
    	int m = grid[0].length;
    	Queue<int[]> que = new LinkedList<>();
    	// 将起点加入队列
		que.add(new int[] { 0, 0 });
    	grid[0][0] = 1;
    	int path = 1;
    	while (!que.isEmpty()) {
    		int size = que.size();
    		while (size-- > 0) {
    			int[] cur = que.poll();
    			int x = cur[0];
    			int y = cur[1];
    			// 到达终点
    			if (x == m - 1 && y == n - 1) {
    				return path;
    			}
    			for (int[] dir : dirs) {
					int x1 = x + dir[0];
					int y1 = y + dir[1];
					if (x1 < 0 || x1 >= m || y1 < 0 || y1 >= n || grid[x1][y1] == 1) {
						continue;
					}
					que.add(new int[] {x1, y1});
					grid[x1][y1] = 1;
				}
    		}
    		path++;
    	}
    	return -1;
    }

	
	
	
	
	
	
	
    
}

class Node {
	public int val;
	public List<Node> children;
	public Node next;
	public Node left;
	public Node right;

	public Node() {
	}

	public Node(int _val) {
		val = _val;
	}

	public Node(int _val, List<Node> _children) {
		val = _val;
		children = _children;
	}
};
