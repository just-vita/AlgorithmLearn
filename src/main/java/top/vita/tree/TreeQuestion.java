package top.vita.tree;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @Author vita
 * @Date 2023/4/26 12:30
 */
@SuppressWarnings("all")
public class TreeQuestion {
    public List<Integer> preorderTraversal(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        dfs1(root, result);
        return result;
    }

    public void dfs1(TreeNode root, ArrayList<Integer> result) {
        if (root == null) {
            return;
        }
        result.add(root.val);
        dfs(root.left, result);
        dfs(root.right, result);
    }

    public List<Integer> preorderTraversal2(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            result.add(node.val);

            if (node.right != null) {
                stack.push(node.right);
            }
            if (node.left != null) {
                stack.push(node.left);
            }
        }
        return result;
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        dfs2(root, result);
        return result;
    }

    public void dfs2(TreeNode root, ArrayList<Integer> result) {
        if (root == null) {
            return;
        }
        dfs(root.left, result);
        result.add(root.val);
        dfs(root.right, result);
    }

    public List<Integer> inorderTraversal1(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()) {
            if (cur != null) {
                stack.push(cur);
                cur = cur.left;
            } else {
                cur = stack.pop();
                result.add(cur.val);
                cur = cur.right;
            }
        }
        return result;
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        dfs(root, result);
        return result;
    }

    public void dfs(TreeNode root, ArrayList<Integer> result) {
        if (root == null) {
            return;
        }
        dfs(root.left, result);
        dfs(root.right, result);
        result.add(root.val);
    }

    public List<Integer> postorderTraversal1(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            result.add(node.val);
            if (node.left != null) {
                stack.push(node.left);
            }
            if (node.right != null) {
                stack.push(node.right);
            }
        }
        Collections.reverse(result);
        return result;
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            // 创建保存一层数的容器
            ArrayList<Integer> path = new ArrayList<>();
            // 在外面先定义一个，防止在进行添加子节点的时候影响到循环次数
            int size = queue.size();
            while (size > 0) {
                TreeNode node = queue.poll();
                path.add(node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
                // 减少这一层的长度
                size--;
            }
            result.add(path);
        }
        return result;
    }

    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            ArrayList<Integer> path = new ArrayList<>();
            int size = queue.size();
            while (size > 0) {
                TreeNode node = queue.poll();
                path.add(node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
                size--;
            }
            result.add(path);
        }
        Collections.reverse(result);
        return result;
    }

    public List<Integer> rightSideView(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (i == size - 1) {
                    result.add(node.val);
                }
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        return result;
    }

    public List<Double> averageOfLevels(TreeNode root) {
        ArrayList<Double> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            double sum = 0;
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                sum += node.val;
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            // 直接使用队列的长度求平均值
            result.add(sum / size);
        }
        return result;
    }

    public List<List<Integer>> levelOrder(Node root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            ArrayList<Integer> path = new ArrayList<>();
            while (size > 0) {
                size--;
                Node node = queue.poll();
                path.add(node.val);
                List<Node> children = node.children;
                if (children == null || children.size() == 0) {
                    continue;
                }
                for (Node cur : children) {
                    if (cur != null) {
                        queue.offer(cur);
                    }
                }
            }
            result.add(path);
        }
        return result;
    }

    public List<Integer> largestValues(TreeNode root) {
        List<Integer> result = new ArrayList();
        if (root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            int max = Integer.MIN_VALUE;
            while (size > 0) {
                size--;
                TreeNode node = queue.poll();
                max = max > node.val ? max : node.val;
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            result.add(max);
        }
        return result;
    }

    public Node connect(Node root) {
        if (root == null) {
            return null;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            Node pre = null;
            while (size > 0) {
                size--;
                Node node = queue.poll();
                if (pre != null) {
                    pre.next = node;
                }
                pre = node;
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        return root;
    }

    public Node connect1(Node root) {
        if (root == null) {
            return null;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            Node pre = null;
            while (size > 0) {
                size--;
                Node node = queue.poll();
                if (pre != null) {
                    pre.next = node;
                }
                pre = node;
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        return root;
    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }

    public boolean isSymmetric3(TreeNode root) {
        return checkIsSymmetric2(root.left, root.right);
    }

    private boolean checkIsSymmetric2(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        } else if (left == null || right == null || left.val != right.val) {
            return false;
        }
        boolean outside = checkIsSymmetric2(left.left, right.right);
        boolean inside = checkIsSymmetric2(left.right, right.left);
        return outside && inside;
    }

    int max = 0;

    public int maxDepth(TreeNode root) {
        getMaxDepth(root, 0);
        return max;
    }

    public void getMaxDepth(TreeNode node, int depth) {
        if (node == null) {
            return;
        }
        depth++;
        max = Math.max(max, depth);
        if (node.left != null) {
            getMaxDepth(node.left, depth);
        }
        if (node.right != null) {
            getMaxDepth(node.right, depth);
        }
    }

    public int maxDepth(Node root) {
        getMaxDepth(root, 0);
        return max;
    }

    public void getMaxDepth(Node node, int depth) {
        if (node == null) {
            return;
        }
        depth++;
        max = Math.max(max, depth);
        for (Node child : node.children) {
            getMaxDepth(child, depth);
        }
    }

    public int maxDepth1(Node root) {
        if (root == null) {
            return 0;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        int depth = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            depth++;
            while (size > 0) {
                size--;
                Node node = queue.poll();
                for (Node child : node.children) {
                    queue.offer(child);
                }
            }
        }
        return depth;
    }

    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int depth = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            depth++;
            while (size > 0) {
                size--;
                TreeNode node = queue.poll();
                if (node.left == null && node.right == null) {
                    return depth;
                }
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        return depth;
    }


    public int countNodes(TreeNode root) {
        if (root == null) return 0;
        int leftNum = countNodes(root.left); // 左边
        int rightNum = countNodes(root.right); // 右边
        int treeNum = leftNum + rightNum + 1; // 中间，加一是因为算上中间节点
        return treeNum;
    }

    public boolean isBalanced(TreeNode root) {
        return isBalancedDfs(root, 0) != -1;
    }

    // 返回-1代表不是平衡树
    public int isBalancedDfs(TreeNode node, int depth) {
        if (node == null) {
            return 0;
        }
        // 左子树高度
        int left = isBalancedDfs(node.left, depth + 1);
        if (left == -1) {
            return -1;
        }
        // 右子树高度
        int right = isBalancedDfs(node.right, depth + 1);
        if (right == -1) {
            return -1;
        }
        if (Math.abs(left - right) > 1) {
            return -1;
        } else {
            return 1 + Math.max(left, right);
        }
    }

    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        List<Integer> path = new ArrayList<>();
        binaryTreePathsDfs(root, res, path);
        return res;
    }

    public void binaryTreePathsDfs(TreeNode node, List<String> res, List<Integer> path) {
        path.add(node.val);
        if (node.left == null && node.right == null) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < path.size() - 1; i++) {
                sb.append(path.get(i));
                sb.append("->");
            }
            sb.append(path.get(path.size() - 1));
            res.add(sb.toString());
        }
        if (node.left != null) {
            binaryTreePathsDfs(node.left, res, path);
            path.remove(path.size() - 1);
        }
        if (node.right != null) {
            binaryTreePathsDfs(node.right, res, path);
            path.remove(path.size() - 1);
        }
    }

    public List<String> binaryTreePaths2(TreeNode root) {
        List<String> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Stack<Object> stack = new Stack<>();
        stack.push(root);
        stack.push(root.val + "");
        while (!stack.isEmpty()) {
            String path = (String) stack.pop();
            TreeNode node = (TreeNode) stack.pop();
            if (node.left == null && node.right == null) {
                res.add(path);
            }
            if (node.left != null) {
                stack.push(node.left);
                stack.push(path + "->" + node.left.val);
            }
            if (node.right != null) {
                stack.push(node.right);
                stack.push(path + "->" + node.right.val);
            }
        }
        return res;
    }

    public List<String> binaryTreePaths3(TreeNode root) {
        List<String> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        String path = "";
        binaryTreePathsDfs(root, res, path);
        return res;
    }

    public void binaryTreePathsDfs(TreeNode node, List<String> res, String path) {
        path += node.val;
        if (node.left == null && node.right == null) {
            res.add(path);
            return;
        }
        if (node.left != null) {
            binaryTreePathsDfs(node.left, res, path + "->");
        }
        if (node.right != null) {
            binaryTreePathsDfs(node.right, res, path + "->");
        }
    }

    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        int sum = 0;
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            if (node.left != null && node.left.left == null && node.left.right == null) {
                sum += node.left.val;
            }
            if (node.left != null) {
                stack.push(node.left);
            }
            if (node.right != null) {
                stack.push(node.right);
            }
        }
        return sum;
    }

    public int findBottomLeftValue(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int res = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            int i = 0;
            while (size > 0) {
                size--;
                TreeNode node = queue.poll();
                if (node.left == null && node.right == null && i == 0) {
                    res = node.val;
                }
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
                i++;
            }
        }
        return res;
    }

    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }

        return dfs(root, targetSum, root.val);
    }

    public boolean dfs(TreeNode node, int targetSum, int sum) {
        if (node == null) {
            return false;
        }
        if (node.left == null && node.right == null) {
            if (targetSum == sum) {
                return true;
            }
        }
        boolean left = false;
        if (node.left != null) {
            left = dfs(node.left, targetSum, sum + node.left.val);
        }
        boolean right = false;
        if (node.right != null) {
            right = dfs(node.right, targetSum, sum + node.right.val);
        }
        return left || right;
    }

    public boolean hasPathSum1(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        // 用两个栈来实现
        Stack<TreeNode> stack = new Stack<>();
        Stack<Integer> intStack = new Stack<>();
        stack.push(root);
        intStack.push(root.val);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            Integer sum = intStack.pop();
            if (node.left == null && node.right == null && targetSum == sum) {
                return true;
            }
            if (node.left != null) {
                stack.push(node.left);
                intStack.push(sum + node.left.val);
            }
            if (node.right != null) {
                stack.push(node.right);
                intStack.push(sum + node.right.val);
            }
        }

        return false;
    }

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        dfs(root, targetSum, root.val, res, new ArrayList<Integer>());
        return res;
    }

    public void dfs(TreeNode node, int targetSum, int sum, List<List<Integer>> res, List<Integer> path) {
        if (node == null) {
            return;
        }
        path.add(node.val);
        if (node.left == null && node.right == null && targetSum == sum) {
            res.add(new ArrayList<>(path));
            return;
        }
        if (node.left != null) {
            dfs(node.left, targetSum, sum + node.left.val, res, path);
            path.remove(path.size() - 1);
        }
        if (node.right != null) {
            dfs(node.right, targetSum, sum + node.right.val, res, path);
            path.remove(path.size() - 1);
        }
    }

    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return createMaxTree(nums, 0, nums.length - 1);
    }

    private TreeNode createMaxTree(int[] nums, int l, int r) {
        if (l > r) {
            return null;
        }
        int max = findMax(nums, l, r);
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

    public TreeNode searchBST(TreeNode root, int val) {
        if (root == null || root.val == val) {
            return root;
        }
        if (val < root.val) {
            return searchBST(root.left, val);
        }
        if (val > root.val) {
            return searchBST(root.right, val);
        }
        return null;
    }

    public TreeNode searchBST2(TreeNode root, int val) {
        while (root != null) {
            if (val < root.val) {
                root = root.left;
            } else if (val > root.val) {
                root = root.right;
            } else {
                return root;
            }
        }
        return null;
    }

    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        // 一方为空就用另一方，为null也无所谓
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

    public TreeNode mergeTrees1(TreeNode root1, TreeNode root2) {
        if (root1 == null) {
            return root2;
        }
        if (root2 == null) {
            return root1;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root1);
        queue.offer(root2);
        while (!queue.isEmpty()) {
            TreeNode node1 = queue.poll();
            TreeNode node2 = queue.poll();
            node1.val += node2.val;
            if (node1.left != null && node2.left != null) {
                queue.offer(node1.left);
                queue.offer(node2.left);
            }
            if (node1.right != null && node2.right != null) {
                queue.offer(node1.right);
                queue.offer(node2.right);
            }
            // 直接操作这棵树
            if (node1.left == null) {
                node1.left = node2.left;
            }
            if (node1.right == null) {
                node1.right = node2.right;
            }
        }
        return root1;
    }

    public int[] findMode(TreeNode root) {
        if (root == null) {
            return new int[0];
        }
        Map<Integer, Integer> map = new HashMap<>();
        List<Integer> list = new ArrayList<>();
        // 获得频率 Map
        searchBST(root, map);
        List<Map.Entry<Integer, Integer>> mapList = map.entrySet().stream()
                .sorted((c1, c2) -> c2.getValue().compareTo(c1.getValue()))
                .collect(Collectors.toList());
        // 把频率最高的加入 list
        list.add(mapList.get(0).getKey());
        // 将频率相同的也加入
        for (int i = 1; i < mapList.size(); i++) {
            if (mapList.get(i).getValue() == mapList.get(i - 1).getValue()) {
                list.add(mapList.get(i).getKey());
            } else {
                break;
            }
        }
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    void searchBST(TreeNode cur, Map<Integer, Integer> map) {
        if (cur == null) {
            return;
        }
        map.put(cur.val, map.getOrDefault(cur.val, 0) + 1);
        searchBST(cur.left, map);
        searchBST(cur.right, map);
    }

    TreeNode pre = null;
    List<Integer> result = new ArrayList<>();
    int count = 0;
    int maxCount = 0;

    public int[] findMode2(TreeNode root) {
        dfs(root);
        return result.stream().mapToInt(Integer::intValue).toArray();
    }

    public void dfs(TreeNode cur) {
        if (cur == null) {
            return;
        }
        dfs(cur.left);
        if (pre == null || pre.val != cur.val) {
            count = 1;
        } else {
            count++;
        }
        // 发现了频率更高的数字，重置结果列表
        if (count > maxCount) {
            result.clear();
            result.add(cur.val);
            maxCount = count;
        } else if (count == maxCount) {
            result.add(cur.val);
        }
        pre = cur;
        dfs(cur.right);
    }

    public int getMinimumDifference(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        // 中序遍历BST树取出有序列表
        getMinimumDifferenceDfs(root, list);
        if (list.size() < 2) {
            return 0;
        }
        int result = Integer.MAX_VALUE;
        // 直接循环判断
        for (int i = 1; i < list.size(); i++) {
            result = Math.min(result, list.get(i) - list.get(i - 1));
        }
        return result;
    }

    public void getMinimumDifferenceDfs(TreeNode root, ArrayList<Integer> list) {
        if (root == null) {
            return;
        }
        getMinimumDifferenceDfs(root.left, list);
        list.add(root.val);
        getMinimumDifferenceDfs(root.right, list);
    }

    int result1 = Integer.MAX_VALUE;

    //    TreeNode pre = null;
    public int getMinimumDifference1(TreeNode root) {
        getMinimumDifferenceDfs(root);
        return result1;
    }

    public void getMinimumDifferenceDfs(TreeNode cur) {
        if (cur == null) {
            return;
        }
        getMinimumDifferenceDfs(cur.left);
        if (pre != null) {
            result1 = Math.min(result1, cur.val - pre.val);
        }
        pre = cur;
        getMinimumDifferenceDfs(cur.right);
    }

    public int getMinimumDifference2(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int result = Integer.MAX_VALUE;
        TreeNode pre = null;
        TreeNode cur = root;
        Stack<TreeNode> stack = new Stack<>();
        while (cur != null || !stack.isEmpty()) {
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();

            if (pre != null) {
                result = Math.min(result, cur.val - pre.val);
            }
            pre = cur;

            cur = cur.right;
        }
        return result;
    }

    //    TreeNode pre = null;
    public boolean isValidBST123(TreeNode root) {
        if (root == null) {
            return true;
        }
        // 中序遍历
        boolean left = isValidBST123(root.left);

        if (pre != null && pre.val >= root.val) {
            return false;
        }
        pre = root;

        boolean right = isValidBST123(root.right);

        return left && right;
    }

    public boolean isValidBST2(TreeNode root) {
        if (root == null) {
            return true;
        }
        TreeNode pre = null;
        TreeNode cur = root;
        Stack<TreeNode> stack = new Stack<>();
        while (cur != null || !stack.isEmpty()) {
            if (cur != null) {
                stack.push(cur);
                // 左
                cur = cur.left;
            } else {
                cur = stack.pop();

                // 中
                if (pre != null && pre.val >= cur.val) {
                    return false;
                }
                pre = cur;

                // 右
                cur = cur.right;
            }
        }
        return true;
    }

    public boolean isValidBST3(TreeNode root) {
        if (root == null) {
            return true;
        }
        TreeNode pre = null;
        TreeNode cur = root;
        Stack<TreeNode> stack = new Stack<>();
        while (cur != null || !stack.isEmpty()) {
            while (cur != null) {
                stack.push(cur);
                // 左
                cur = cur.left;
            }
            // 中
            cur = stack.pop();
            if (pre != null && pre.val >= cur.val) {
                return false;
            }
            pre = cur;

            // 右
            cur = cur.right;
        }
        return true;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // 找到了就返回节点，没找到就返回null
        if (root == null || root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        // 两个子孙都找到了，那么当前节点就是最近的公共祖先
        if (left != null && right != null) {
            return root;
        }
        // 左边没找到，但是右边找到了，代表右边已经找到了最近的公共祖先，往上返回
        if (left == null && right != null) {
            return right;
        }
        // 左边找到了，但是右边没找到，代表左边已经找到了最近的公共祖先，往上返回
        return left;
    }

    public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
        // 比两个子孙都大，搜索左边
        if (root.val > p.val && root.val > q.val) {
            return lowestCommonAncestor1(root.left, p, q);
        }
        // 比两个子孙都小，搜索右边
        if (root.val < p.val && root.val < q.val) {
            return lowestCommonAncestor1(root.right, p, q);
        }
        // 找到第一个在（p, q）区间中的节点，这个就是最近公共祖先（BST特性）
        return root;
    }

    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode cur = stack.pop();
            if (cur.val > p.val && cur.val > q.val) {
                stack.push(cur.left);
                continue;
            }
            if (cur.val < p.val && cur.val < q.val) {
                stack.push(cur.right);
                continue;
            }
            return cur;
        }
        return null;
    }

    public TreeNode lowestCommonAncestor3(TreeNode root, TreeNode p, TreeNode q) {
        while (root != null){
            // 比两个子孙都大，搜索左边
            if (root.val > p.val && root.val > q.val) {
                root = root.left;
            }
            // 比两个子孙都小，搜索右边
            else if (root.val < p.val && root.val < q.val) {
                root = root.right;
            }
            // 找到第一个在（p, q）区间中的节点，这个就是最近公共祖先（BST特性）
            else {
                return root;
            }
        }
        return null;
    }

    public TreeNode insertIntoBST(TreeNode root, int val) {
        if (root == null) {
            return new TreeNode(val);
        }
        if (root.val > val){
            root.left = insertIntoBST(root.left, val);
        } else if (root.val < val){
            root.right = insertIntoBST(root.right, val);
        }
        return root;
    }

    public TreeNode insertIntoBST2(TreeNode root, int val) {
        if (root == null){
            return new TreeNode(val);
        }
        TreeNode cur = root;
        TreeNode parent = root;
        while (cur != null){
            parent = cur;
            if (cur.val > val) {
                cur = cur.left;
            } else {
                cur = cur.right;
            }
        }
        if (parent.val > val){
            parent.left = new TreeNode(val);
        } else {
            parent.right = new TreeNode(val);
        }
        return root;
    }

    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) {
            return null;
        }
        if (root.val > key) {
            root.left = deleteNode(root.left, key);
        } else if (root.val < key) {
            root.right = deleteNode(root.right, key);
        } else {
            if (root.left == null) {
                return root.right;
            } else if (root.right == null) {
                return root.left;
            } else if (root.left != null && root.right != null) {
                // 把左节点移动到「右节点的最左节点」的左节点上
                // 找到「右节点的最左节点」
                TreeNode cur = root.right;
                while (cur.left != null){
                    cur = cur.left;
                }
                // 将待删除节点的左节点移动「右节点的最左节点」上
                cur.left = root.left;
                // 将待删除节点用右节点覆盖
                root = root.right;
            }
        }
        return root;
    }

    public TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null){
            return null;
        }

        // 根节点不在范围内的情况
        if (root.val < low){
            // low比根节点大，希望右子树可以比low大
            return trimBST(root.right, low, high);
        }
        if (root.val > high){
            // high比根节点小，希望左子树可以比high小
            return trimBST(root.left, low, high);
        }

        // 根节点在范围内的情况
        root.left = trimBST(root.left, low, high);
        root.right = trimBST(root.right, low, high);
        // 保留根节点
        return root;
    }

    public TreeNode sortedArrayToBST1(int[] nums) {
        return sortedArrayToBSTDfs(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBSTDfs(int[] nums, int left, int right) {
        if (left > right){
            return null;
        }
        // 取中间值作为当前的根节点
        int mid = left + ((right - left) / 2);
        TreeNode cur = new TreeNode(nums[mid]);
        // 左节点用比中间节点小的
        cur.left = sortedArrayToBSTDfs(nums, left, mid - 1);
        // 右节点用比中间节点大的
        cur.right = sortedArrayToBSTDfs(nums, mid + 1, right);
        return cur;
    }

    int preNum = 0;
    public TreeNode convertBST(TreeNode root) {
        convertBSTDfs(root);
        return root;
    }

    private void convertBSTDfs(TreeNode cur) {
        if (cur == null){
            return;
        }
        // 先遍历右节点，从后往前遍历，拿到累加和
        convertBSTDfs(cur.right);
        cur.val += preNum;
        preNum = cur.val;
        convertBSTDfs(cur.left);
    }

    public TreeNode convertBST2(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        int preNum = 0;
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()){
            while (cur != null){
                // 先遍历到最右节点，从后往前遍历
                stack.push(cur);
                cur = cur.right;
            }
            cur = stack.pop();
            cur.val += preNum;
            preNum = cur.val;
            cur = cur.left;
        }
        return root;
    }

    public int[] levelOrder1(TreeNode root) {
        if (root == null) {
            return new int[0];
        }
        List<Integer> list = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            while (size > 0) {
                size--;
                TreeNode cur = queue.poll();
                list.add(cur.val);
                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
        }
        int[] res = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return res;
    }

    public List<List<Integer>> levelOrder2(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            while (size > 0) {
                size--;
                TreeNode cur = queue.poll();
                list.add(cur.val);
                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
            res.add(list);
        }
        return res;
    }

    public List<List<Integer>> levelOrder3(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        // 用栈实现从右往左存入
        Stack<TreeNode> stack = new Stack<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        // 用来判断层数奇偶
        int level = 1;
        while (!queue.isEmpty()){
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            while (size > 0) {
                size--;
                TreeNode cur = queue.poll();
                list.add(cur.val);
                if (level % 2 == 1) {
                    // 现在是寄数，那么下一层就是偶数，需要的是逆序结果
                    // 因为栈的出栈顺序是逆序，需要反过来先将左子树加入栈中
                    if (cur.left != null) {
                        stack.push(cur.left);
                    }
                    if (cur.right != null) {
                        stack.push(cur.right);
                    }
                } else {
                    // 现在是偶数，那么下一层就是奇数，需要的是正序结果
                    // 因为栈的出栈顺序是逆序，需要反过来先将右子树加入栈中
                    if (cur.right != null) {
                        stack.push(cur.right);
                    }
                    if (cur.left != null) {
                        stack.push(cur.left);
                    }
                }
            }
            // 栈的长度会在循环中减少，所以这个地方得要先保存栈的长度
            // 否则会导致结果出错，草了，我居然和上面那个循环里的size一样卡了半天
            int stackSize = stack.size();
            for (int i = 0; i < stackSize; i++) {
                queue.offer(stack.pop());
            }
            res.add(list);
            level++;
        }
        return res;
    }

    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if (A == null || B == null) {
            return false;
        }
        // 把当前节点作为起点 || 把左子树作为起点 || 把右子树作为起点
        return findSub(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B);
    }

    public boolean findSub(TreeNode A, TreeNode B) {
        // B树遍历到底了，也就是说在A树中找到了一条完全重合的路径
        if (B == null) {
            return true;
        }
        // A树遍历到底了，也就是说在A树中没有找到一条完全重合的路径
        if (A == null) {
            return false;
        }
        // 当节点值相等且左右子树都能证明路径重合的话才会返回true
        return A.val == B.val && findSub(A.left, B.left) && findSub(A.right, B.right);
    }

    /*Node pre, head;
    public Node treeToDoublyList(Node root) {
        if (root == null) {
            return root;
        }
        dfs(root);
        // 左子树就是前驱（比当前数小），右子树就是后继（比当前数大），对着图例走一遍就理解了
        head.left = pre;
        pre.right = head;
        return head;
    }

    void dfs(Node cur) {
        if (cur == null) {
            return;
        }
        // 前
        dfs(cur.left);
        // 中
        // pre为空，代表到达树的最左下角的叶子节点
        if (pre == null) {
            // 将它作为双向链表的头节点
            head = cur;
        } else {
            // 不是头节点，存储前驱节点，形成双向链表
            cur.left = pre;
            // 存储上一个节点的后继节点（当前节点）
            pre.right = cur;
        }
        // 存储前驱节点
        pre = cur;
        // 后
        dfs(cur.right);
    }*/

    int res = 0;
    int rank = 0;
    public int kthLargest(TreeNode root, int k) {
        kthLargestDfs(root, k);
        return res;
    }

    public void kthLargestDfs(TreeNode cur, int k) {
        if (res != 0) {
            return;
        }
        // 直接先找到最大的数
        if (cur.right != null) {
            kthLargestDfs(cur.right, k);
        }

        // 找到了第k大的数
        if (++rank == k) {
            res = cur.val;
            return;
        }

        if (cur.left != null) {
            kthLargestDfs(cur.left, k);
        }
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        // 分治思想
        // 前序遍历数组的第一个永远是树（或子树）的根节点
        // 中序遍历时找出当前的根节点，分出左右子树
        // 也就是：找出root，分出左右子树，之后递归到的子树都是这个流程，直到不能分
        return preOrder(0, preorder.length - 1, 0, inorder.length - 1, preorder, inorder);
    }

    // 参数是前序和中序遍历的节点的索引位置
    TreeNode preOrder(int preLeft, int preRight, int inLeft, int inRight, int[] pre, int[] in) {
        // 左右子树索引相撞，分不了了，直接返回null
        if (preLeft > preRight || inLeft > inRight) {
            return null;
        }
        // 创建当前的根节点
        TreeNode root = new TreeNode(pre[preLeft]);
        // 从中序的最左元素位置开始
        int InRoot = inLeft;
        // 在中序中查找根节点位置，判断是否是根节点（前序的第一个节点）
        while (InRoot <= inRight && pre[preLeft] != in[InRoot]) {
            InRoot++;
        }
        // 在中序中找到了根节点位置
        // 中序的根节点的左边都是左子树，右边都是右子树
        // 得到中序的左子树的长度，剩下的都是右子树
        int length = InRoot - inLeft;
        // 从左至右依次为
        // 前序的左子树里的下一个根节点的位置
        // 前序的最后一个左子树的位置
        // 中序的第一个左子树的位置
        // 中序的最后一个左子树的位置
        root.left = preOrder(preLeft + 1, preLeft + length, inLeft, InRoot - 1, pre, in);
        //  前序的右子树里的下一个根节点的位置
        //  前序的最后一个右子树的位置
        //  中序的第一个右子树的位置
        //  中序的最后一个右子树的位置
        root.right = preOrder(preLeft + length + 1, preRight, InRoot + 1, inRight, pre, in);
        return root;
    }

    public boolean verifyPostorder(int[] postorder) {
        return verifyPostorder(0, postorder.length - 1, postorder);
    }

    boolean verifyPostorder(int left, int right, int[] postorder) {
        // 当前节点为叶子节点，当然是了，直接返回true
        if (left >= right) {
            return true;
        }
        // 记录根节点的值
        int rootValue = postorder[right];
        // 记录一个遍历指针，最后会指向根节点
        int rootPosition = left;
        // 找到第一个大于根节点的节点，也就是第一个右子树（后序遍历的根节点在最后）
        while (rootValue > postorder[rootPosition]) {
            rootPosition++;
        }
        // 记录第一个右子树的位置
        int firstRight = rootPosition;
        // 从右子树开始找，找到最后一个右子树的位置之后的位置（根节点）
        while (rootValue < postorder[rootPosition]) {
            rootPosition++;
        }
        // 判断是否到达根节点 判断左子树是否正确 判断右子树是否正确
        return rootPosition == right && verifyPostorder(left, firstRight - 1, postorder) && verifyPostorder(firstRight, right - 1, postorder);
    }

    public boolean checkTree(TreeNode root) {
        return root.left.val + root.right.val == root.val;
    }

    public void inorderTraversal23(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        inorderTraversal23(root.left, res);
        res.add(root.val);
        inorderTraversal23(root.right, res);
    }

    public int maxDepth3(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int depth = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            while (size != 0) {
                size--;
                TreeNode cur = queue.poll();
                if (cur.left != null) {
                    queue.add(cur.left);
                }
                if (cur.right != null) {
                    queue.add(cur.right);
                }
            }
            depth++;
        }
        return depth;
    }

    public TreeNode invertTree2(TreeNode root) {
        if (root == null) {
            return null;
        }
        root.left = invertTree2(root.left);
        root.right = invertTree2(root.right);
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        return root;
    }

    public int goodNodes(TreeNode root) {
        // 加上当前节点的1个
        return 1 + goodNodesDfs(root.left, root.val) + goodNodesDfs(root.right, root.val);
    }

    private int goodNodesDfs(TreeNode cur, int max) {
        if (cur == null) {
            return 0;
        }
        if (cur.val >= max) {
            // 如果大的话，就把比较大的那个值往下传
            return 1 + goodNodesDfs(cur.left, cur.val) + goodNodesDfs(cur.right, cur.val);
        } else {
            return goodNodesDfs(cur.left, max) + goodNodesDfs(cur.right, max);
        }
    }

    public boolean isSymmetric(TreeNode root) {
        return checkIsSymmetric(root.left, root.right);
    }

    private boolean checkIsSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        } else if (left == null || right == null || left.val != right.val) {
            return false;
        }
        boolean out = checkIsSymmetric(left.left, right.right);
        boolean in = checkIsSymmetric(left.right, right.left);
        return out && in;
    }

    int maxDiameter = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        getDiameterOfBinaryTree(root);
        return maxDiameter;
    }
    private int getDiameterOfBinaryTree(TreeNode cur) {
        // 到达叶子节点时开始返回
        if (cur.left == null && cur.right == null) {
            return 0;
        }
        // 向左右节点索取节点个数信息，加上当前节点本身的1个
        int left = cur.left == null ? 0 : getDiameterOfBinaryTree(cur.left) + 1;
        int right = cur.right == null ? 0 : getDiameterOfBinaryTree(cur.right) + 1;
        // 记录最高直径，其实就是不算根节点，计算左右子节点的最大个数的和
        maxDiameter = Math.max(maxDiameter, left + right);
        return Math.max(left, right);
    }

    int maxPathSum = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        getMaxPathSum(root);
        return maxPathSum;
    }

    private int getMaxPathSum(TreeNode cur) {
        if (cur == null) {
            return 0;
        }
        // 舍弃负数
        int leftSum = Math.max(0, getMaxPathSum(cur.left));
        int rightSum = Math.max(0, getMaxPathSum(cur.right));
        maxPathSum = Math.max(maxPathSum, leftSum + rightSum + cur.val);
        return Math.max(leftSum, rightSum) + cur.val;
    }

    int longestUnivaluePath = 0;
    public int longestUnivaluePath(TreeNode root) {
        if (root == null) {
            return 0;
        }
        getLongestUnivaluePath(root);
        return longestUnivaluePath;
    }

    private int getLongestUnivaluePath(TreeNode cur) {
        if (cur.left == null && cur.right == null) {
            return 0;
        }
        int left = cur.left == null ? 0 : getLongestUnivaluePath(cur.left) + 1;
        int right = cur.right == null ? 0 : getLongestUnivaluePath(cur.right) + 1;
        if (left >= 0 && cur.val != cur.left.val) {
            left = 0;
        }
        if (right >= 0 && cur.val != cur.right.val) {
            right = 0;
        }
        longestUnivaluePath = Math.max(longestUnivaluePath, left + right);
        return Math.max(left, right);
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return getSortedArrayToBST(nums, 0, nums.length);
    }

    private TreeNode getSortedArrayToBST(int[] nums, int left, int right) {
        if (left == right) {
            return new TreeNode(nums[left]);
        }
        int mid = (left + right) / 2;
        TreeNode cur = new TreeNode(nums[mid]);
        cur.left = getSortedArrayToBST(nums, left, mid);
        cur.right = getSortedArrayToBST(nums, mid + 1, right);
        return cur;
    }

    public boolean isValidBST(TreeNode cur) {
        if (cur == null) {
            return true;
        }
        boolean left = isValidBST(cur.left);

        if (pre != null && pre.val >= cur.val) {
            return false;
        }
        pre = cur;

        boolean right = isValidBST(cur.right);
        return left && right;
    }

    int curCount = 0;
    public int kthSmallest(TreeNode root, int k) {
        getKthSmallest(root, k);
        return res;
    }

    public void getKthSmallest(TreeNode cur, int k) {
        if (cur == null) {
            return;
        }
        getKthSmallest(cur.left, k);
        curCount++;
        if (k == curCount) {
            res = cur.val;
            return;
        }
        getKthSmallest(cur.right, k);
    }
}