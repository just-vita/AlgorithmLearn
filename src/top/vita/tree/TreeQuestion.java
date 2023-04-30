package top.vita.tree;

import java.util.*;

/**
 * @Author vita
 * @Date 2023/4/26 12:30
 */
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
            // 创建保存一层数的容器
            ArrayList<Integer> path = new ArrayList<>();
            // 在外面先定义一个，防止在进行添加子节点的时候影响到循环次数
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
                // 减少这一层的长度
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

}
