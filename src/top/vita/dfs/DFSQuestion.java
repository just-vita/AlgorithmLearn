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
}
