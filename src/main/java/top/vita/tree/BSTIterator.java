package top.vita.tree;

import java.util.Deque;
import java.util.LinkedList;

class BSTIterator {
    Deque<TreeNode> stack = new LinkedList<>();

    public BSTIterator(TreeNode root) {
        while (root != null) {
            stack.push(root);
            root = root.left;
        }
    }

    public int next() {
        TreeNode cur = stack.pop();
        int val = cur.val;
        // 中序遍历左中右的左中已经走完，切换到右节点
        cur = cur.right;
        // 模拟中序遍历找当前节点最左
        while (cur != null) {
            stack.push(cur);
            cur = cur.left;
        }
        return val;
    }

    public boolean hasNext() {
        return !stack.isEmpty();
    }
}