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
        // ������������ҵ������Ѿ����꣬�л����ҽڵ�
        cur = cur.right;
        // ģ����������ҵ�ǰ�ڵ�����
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