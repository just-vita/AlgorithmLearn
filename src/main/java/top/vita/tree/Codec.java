package top.vita.tree;

import java.util.LinkedList;
import java.util.Queue;

public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) {
            return "#_";
        }
        String str = root.val + "_";
        str += serialize(root.left);
        str += serialize(root.right);
        return str;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Queue<String> queue = new LinkedList<>();
        String[] s = data.split("_");
        for (int i = 0; i < s.length; i++) {
            queue.add(s[i]);
        }
        return decode(queue);
    }

    private TreeNode decode(Queue<String> queue) {
        String cur = queue.poll();
        if ("#".equals(cur)) {
            return null;
        }
        TreeNode head = new TreeNode(Integer.parseInt(cur));
        head.left = decode(queue);
        head.right = decode(queue);
        return head;
    }
}