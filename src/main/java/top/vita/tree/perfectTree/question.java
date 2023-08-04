package top.vita.tree.perfectTree;


public class question {
    public Node connect(Node root) {
		if (root == null || root.left == null)
			return root;
		// 左子树的next指向右子树
		root.left.next = root.right;
		if (root.next != null) {
			// 右子树的next指向root的右边的树的左子树
			root.right.next = root.next.left;
		}
		connect(root.left);
		connect(root.right);
		return root;
    }
    

}

class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {}
    
    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
};