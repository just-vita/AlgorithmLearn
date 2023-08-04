package top.vita.tree;

public class TreeDemo {

    /*
     669. 修剪二叉搜索树
     */
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null) {
            return null;
        }

        if (root.val < low) {
            // BST右子树比当前节点大，希望右子树可以比low大
            return trimBST(root.right, low, high);
        }
        if (root.val > high) {
            // BST左子树比当前节点小，希望左子树可以比high小
            return trimBST(root.left, low, high);
        }

        root.left = trimBST(root.left, low, high);
        root.right = trimBST(root.right, low, high);
        return root;
    }
}
