package top.vita.tree;

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
        if (root == null){
            return;
        }
        result.add(root.val);
        dfs(root.left, result);
        dfs(root.right, result);
    }

    public List<Integer> preorderTraversal2(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null){
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            TreeNode node = stack.pop();
            result.add(node.val);

            if (node.right != null){
                stack.push(node.right);
            }
            if (node.left != null){
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
        if (root == null){
            return;
        }
        dfs(root.left, result);
        result.add(root.val);
        dfs(root.right, result);
    }

    public List<Integer> inorderTraversal1(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null){
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()){
            if (cur != null){
                stack.push(cur);
                cur = cur.left;
            } else{
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
        if (root == null){
            return;
        }
        dfs(root.left, result);
        dfs(root.right, result);
        result.add(root.val);
    }

    public List<Integer> postorderTraversal1(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null){
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            TreeNode node = stack.pop();
            result.add(node.val);
            if (node.left != null){
                stack.push(node.left);
            }
            if (node.right != null){
                stack.push(node.right);
            }
        }
        Collections.reverse(result);
        return result;
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (root == null){
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            // ��������һ����������
            ArrayList<Integer> path = new ArrayList<>();
            // �������ȶ���һ������ֹ�ڽ�������ӽڵ��ʱ��Ӱ�쵽ѭ������
            int size = queue.size();
            while (size > 0){
                TreeNode node = queue.poll();
                path.add(node.val);
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
                    queue.offer(node.right);
                }
                // ������һ��ĳ���
                size--;
            }
            result.add(path);
        }
        return result;
    }

    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null){
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            ArrayList<Integer> path = new ArrayList<>();
            int size = queue.size();
            while (size > 0){
                TreeNode node = queue.poll();
                path.add(node.val);
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
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
        if (root == null){
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int size = queue.size();
            for (int i = 0; i < size; i++){
                TreeNode node = queue.poll();
                if (i == size - 1){
                    result.add(node.val);
                }
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
                    queue.offer(node.right);
                }
            }
        }
        return result;
    }

    public List<Double> averageOfLevels(TreeNode root) {
        ArrayList<Double> result = new ArrayList<>();
        if (root == null){
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int size = queue.size();
            double sum = 0;
            for (int i = 0; i < size; i++){
                TreeNode node = queue.poll();
                sum += node.val;
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
                    queue.offer(node.right);
                }
            }
            // ֱ��ʹ�ö��еĳ�����ƽ��ֵ
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
        while (!queue.isEmpty()){
            int size = queue.size();
            ArrayList<Integer> path = new ArrayList<>();
            while (size > 0){
                size--;
                Node node = queue.poll();
                path.add(node.val);
                List<Node> children = node.children;
                if (children == null || children.size() == 0){
                    continue;
                }
                for (Node cur : children){
                    if (cur != null){
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
        if(root == null){
            return result;
        }
        Queue<TreeNode> queue = new LinkedList();
        queue.offer(root);
        while (!queue.isEmpty()){
            int size = queue.size();
            int max = Integer.MIN_VALUE;
            while (size > 0){
                size--;
                TreeNode node = queue.poll();
                max = max > node.val ? max : node.val;
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
                    queue.offer(node.right);
                }
            }
            result.add(max);
        }
        return result;
    }

    public Node connect(Node root) {
        if (root == null){
            return null;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int size = queue.size();
            Node pre = null;
            while (size > 0){
                size--;
                Node node = queue.poll();
                if (pre != null){
                    pre.next = node;
                }
                pre = node;
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
                    queue.offer(node.right);
                }
            }
        }
        return root;
    }

    public Node connect1(Node root) {
        if (root == null){
            return null;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int size = queue.size();
            Node pre = null;
            while (size > 0){
                size--;
                Node node = queue.poll();
                if (pre != null){
                    pre.next = node;
                }
                pre = node;
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
                    queue.offer(node.right);
                }
            }
        }
        return root;
    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null){
            return null;
        }
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }

    public boolean isSymmetric(TreeNode root) {
        return checkIsSymmetric(root.left, root.right);
    }

    private boolean checkIsSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null){
            return true;
        } else if (left == null || right == null || left.val != right.val){
            return false;
        }
        boolean outside = checkIsSymmetric(left.left, right.right);
        boolean inside = checkIsSymmetric(left.right, right.left);
        return outside && inside;
    }

    int max = 0;
    public int maxDepth(TreeNode root) {
        getMaxDepth(root, 0);
        return max;
    }

    public void getMaxDepth(TreeNode node, int depth){
        if (node == null){
            return;
        }
        depth++;
        max = Math.max(max, depth);
        if (node.left != null){
            getMaxDepth(node.left, depth);
        }
        if (node.right != null){
            getMaxDepth(node.right, depth);
        }
    }

    public int maxDepth(Node root) {
        getMaxDepth(root, 0);
        return max;
    }

    public void getMaxDepth(Node node, int depth){
        if (node == null){
            return;
        }
        depth++;
        max = Math.max(max, depth);
        for (Node child : node.children){
            getMaxDepth(child, depth);
        }
    }

    public int maxDepth1(Node root) {
        if (root == null){
            return 0;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        int depth = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            depth++;
            while (size > 0){
                size--;
                Node node = queue.poll();
                for (Node child : node.children){
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
            while (size > 0){
                size--;
                TreeNode node = queue.poll();
                if (node.left == null && node.right == null) {
                    return depth;
                }
                if (node.left != null){
                    queue.offer(node.left);
                }
                if (node.right != null){
                    queue.offer(node.right);
                }
            }
        }
        return depth;
    }


    public int countNodes(TreeNode root) {
        if (root == null) return 0;
        int leftNum = countNodes(root.left); // ���
        int rightNum = countNodes(root.right); // �ұ�
        int treeNum = leftNum + rightNum + 1; // �м䣬��һ����Ϊ�����м�ڵ�
        return treeNum;
    }

    public boolean isBalanced(TreeNode root) {
        return isBalancedDfs(root, 0) != -1;
    }

    // ����-1������ƽ����
    public int isBalancedDfs(TreeNode node, int depth){
        if (node == null){
            return 0;
        }
        // �������߶�
        int left = isBalancedDfs(node.left, depth + 1);
        if (left == -1){
            return -1;
        }
        // �������߶�
        int right = isBalancedDfs(node.right, depth + 1);
        if (right == -1){
            return -1;
        }
        if (Math.abs(left - right) > 1){
            return -1;
        } else {
            return 1 + Math.max(left, right);
        }
    }

    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        if (root == null){
            return res;
        }
        List<Integer> path = new ArrayList<>();
        binaryTreePathsDfs(root, res, path);
        return res;
    }

    public void binaryTreePathsDfs(TreeNode node, List<String> res, List<Integer> path){
        path.add(node.val);
        if (node.left == null && node.right == null){
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < path.size() - 1; i++){
                sb.append(path.get(i));
                sb.append("->");
            }
            sb.append(path.get(path.size() - 1));
            res.add(sb.toString());
        }
        if (node.left != null){
            binaryTreePathsDfs(node.left, res, path);
            path.remove(path.size() - 1);
        }
        if (node.right != null){
            binaryTreePathsDfs(node.right, res, path);
            path.remove(path.size() - 1);
        }
    }

    public List<String> binaryTreePaths2(TreeNode root) {
        List<String> res = new ArrayList<>();
        if (root == null){
            return res;
        }
        Stack<Object> stack = new Stack<>();
        stack.push(root);
        stack.push(root.val + "");
        while (!stack.isEmpty()){
            String path = (String) stack.pop();
            TreeNode node = (TreeNode) stack.pop();
            if (node.left == null && node.right == null){
                res.add(path);
            }
            if (node.left != null){
                stack.push(node.left);
                stack.push(path + "->" + node.left.val);
            }
            if (node.right != null){
                stack.push(node.right);
                stack.push(path + "->" + node.right.val);
            }
        }
        return res;
    }

    public List<String> binaryTreePaths3(TreeNode root) {
        List<String> res = new ArrayList<>();
        if (root == null){
            return res;
        }
        String path = "";
        binaryTreePathsDfs(root, res, path);
        return res;
    }

    public void binaryTreePathsDfs(TreeNode node, List<String> res, String path){
        path += node.val;
        if (node.left == null && node.right == null){
            res.add(path);
            return;
        }
        if (node.left != null){
            binaryTreePathsDfs(node.left, res, path + "->");
        }
        if (node.right != null){
            binaryTreePathsDfs(node.right, res, path + "->");
        }
    }

    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null){
            return 0;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        int sum = 0;
        while (!stack.isEmpty()){
            TreeNode node = stack.pop();
            if (node.left != null && node.left.left == null && node.left.right == null){
                sum += node.left.val;
            }
            if (node.left != null){
                stack.push(node.left);
            }
            if (node.right != null){
                stack.push(node.right);
            }
        }
        return sum;
    }

    public int findBottomLeftValue(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int res = 0;
        while (!queue.isEmpty()){
            int size = queue.size();
            int i = 0;
            while (size > 0){
                size--;
                TreeNode node = queue.poll();
                if (node.left == null && node.right == null && i == 0){
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
        if (root == null){
            return false;
        }

        return dfs(root, targetSum, root.val);
    }

    public boolean dfs(TreeNode node, int targetSum, int sum){
        if (node == null){
            return false;
        }
        if (node.left == null && node.right == null){
            if (targetSum == sum){
                return true;
            }
        }
        boolean left = false;
        if (node.left != null){
            left = dfs(node.left, targetSum, sum + node.left.val);
        }
        boolean right = false;
        if (node.right != null){
            right = dfs(node.right, targetSum, sum + node.right.val);
        }
        return left || right;
    }

    public boolean hasPathSum1(TreeNode root, int targetSum) {
        if (root == null){
            return false;
        }
        // ������ջ��ʵ��
        Stack<TreeNode> stack = new Stack<>();
        Stack<Integer> intStack = new Stack<>();
        stack.push(root);
        intStack.push(root.val);
        while (!stack.isEmpty()){
            TreeNode node = stack.pop();
            Integer sum = intStack.pop();
            if (node.left == null && node.right == null && targetSum == sum){
                return true;
            }
            if (node.left != null){
                stack.push(node.left);
                intStack.push(sum + node.left.val);
            }
            if (node.right != null){
                stack.push(node.right);
                intStack.push(sum + node.right.val);
            }
        }

        return false;
    }

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null){
            return res;
        }
        dfs(root, targetSum, root.val, res, new ArrayList<Integer>());
        return res;
    }

    public void dfs(TreeNode node, int targetSum, int sum, List<List<Integer>> res, List<Integer> path){
        if (node == null){
            return;
        }
        path.add(node.val);
        if (node.left == null && node.right == null && targetSum == sum){
            res.add(new ArrayList<>(path));
            return;
        }
        if (node.left != null){
            dfs(node.left, targetSum, sum + node.left.val, res, path);
            path.remove(path.size() - 1);
        }
        if (node.right != null){
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
        while (root != null){
            if (val < root.val){
                root = root.left;
            } else if (val > root.val){
                root = root.right;
            } else{
                return root;
            }
        }
        return null;
    }

    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        // һ��Ϊ�վ�����һ����ΪnullҲ����ν
        if (root1 == null){
            return root2;
        }
        if (root2 == null){
            return root1;
        }
        root1.val += root2.val;
        root1.left = mergeTrees(root1.left, root2.left);
        root1.right = mergeTrees(root1.right, root2.right);
        return root1;
    }

    public TreeNode mergeTrees1(TreeNode root1, TreeNode root2) {
        if (root1 == null){
            return root2;
        }
        if (root2 == null){
            return root1;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root1);
        queue.offer(root2);
        while (!queue.isEmpty()){
            TreeNode node1 = queue.poll();
            TreeNode node2 = queue.poll();
            node1.val += node2.val;
            if (node1.left != null && node2.left != null){
                queue.offer(node1.left);
                queue.offer(node2.left);
            }
            if (node1.right != null && node2.right != null){
                queue.offer(node1.right);
                queue.offer(node2.right);
            }
            // ֱ�Ӳ��������
            if (node1.left == null){
                node1.left = node2.left;
            }
            if (node1.right == null){
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
        // ���Ƶ�� Map
        searchBST(root, map);
        List<Map.Entry<Integer, Integer>> mapList = map.entrySet().stream()
                .sorted((c1, c2) -> c2.getValue().compareTo(c1.getValue()))
                .collect(Collectors.toList());
        // ��Ƶ����ߵļ��� list
        list.add(mapList.get(0).getKey());
        // ��Ƶ����ͬ��Ҳ����
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
        if (cur == null){
            return;
        }
        dfs(cur.left);
        if (pre == null || pre.val != cur.val){
            count = 1;
        } else {
            count++;
        }
        // ������Ƶ�ʸ��ߵ����֣����ý���б�
        if (count > maxCount){
            result.clear();
            result.add(cur.val);
            maxCount = count;
        } else if (count == maxCount){
            result.add(cur.val);
        }
        pre = cur;
        dfs(cur.right);
    }

    public int getMinimumDifference(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        // �������BST��ȡ�������б�
        getMinimumDifferenceDfs(root, list);
        if (list.size() < 2){
            return 0;
        }
        int result = Integer.MAX_VALUE;
        // ֱ��ѭ���ж�
        for (int i = 1; i < list.size(); i++){
            result = Math.min(result, list.get(i) - list.get(i - 1));
        }
        return result;
    }

    public void getMinimumDifferenceDfs(TreeNode root, ArrayList<Integer> list){
        if (root == null){
            return;
        }
        getMinimumDifferenceDfs(root.left, list);
        list.add(root.val);
        getMinimumDifferenceDfs(root.right, list);
    }


}
