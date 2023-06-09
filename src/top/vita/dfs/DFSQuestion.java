package top.vita.dfs;

/**
 * @Author vita
 * @Date 2023/6/6 11:19
 */
public class DFSQuestion {
    public boolean exist(char[][] board, String word) {
        boolean[][] visited = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == word.charAt(0)) {
                    if (backtracking(0, i, j, board, word)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private boolean backtracking(int startIndex, int i, int j, char[][] board, String word) {
        if (startIndex >= word.length()) {
            return true;
        }
        if (i >= board.length || j >= board[0].length || i < 0 || j < 0 || board[i][j] != word.charAt(startIndex)) {
            return false;
        }
        board[i][j] += 256;
        boolean top = backtracking(startIndex + 1, i - 1, j, board, word);
        boolean bottom = backtracking(startIndex + 1, i + 1, j, board, word);
        boolean left = backtracking(startIndex + 1, i, j - 1, board, word);
        boolean right = backtracking(startIndex + 1, i, j + 1, board, word);
        board[i][j] -= 256;
        return top || bottom || left || right;
    }

    public int sumNums(int n) {
        // 当不满足 n > 1 时，因为短路与的原因，会跳出递归
        // 后面的 == 0 只是用来让后面的表达式返回的是boolean，改成别的也没问题
        boolean x = n > 1 && (n += sumNums(n - 1)) == 0;
        return n;
    }
}
