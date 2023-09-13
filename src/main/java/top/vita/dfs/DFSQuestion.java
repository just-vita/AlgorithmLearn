package top.vita.dfs;

import java.util.*;

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

    public String[] permutation(String s) {
        Set<String> res = new HashSet<>();
        permutationDfs(0, s, new StringBuilder(), new boolean[s.length()], res);
        return res.toArray(new String[0]);
    }

    void permutationDfs(int startIndex, String s, StringBuilder sb, boolean[] visited,Set<String> res) {
        if (sb.length() >= s.length()) {
            res.add(sb.toString());
            return;
        }
        for (int i = 0; i < s.length(); i++) {
            if (visited[i]) {
                continue;
            }
            if (i > 0 && s.charAt(i) == s.charAt(i - 1) && !visited[i - 1]) {
                continue;
            }
            visited[i] = true;
            sb.append(s.charAt(i));
            permutationDfs(i, s, sb, visited, res);
            sb.deleteCharAt(sb.length() - 1);
            visited[i] = false;
        }
    }

    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        getAllPermute(nums, new ArrayList<>(), new boolean[nums.length]);
        return res;
    }

    private void getAllPermute(int[] nums, ArrayList<Integer> path, boolean[] visited) {
        if (path.size() == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) {
                continue;
            }
            visited[i] = true;
            path.add(nums[i]);
            getAllPermute(nums, path, visited);
            path.remove(path.size() - 1);
            visited[i] = false;
        }
    }

    public List<List<Integer>> subsets(int[] nums) {
        ArrayList<List<Integer>> res = new ArrayList<>();
        getSubsets(0, nums, new ArrayList<>(), res);
        return res;
    }

    private void getSubsets(int startIndex, int[] nums, List<Integer> path, ArrayList<List<Integer>> res) {
        res.add(new ArrayList<>(path));
        for (int i = startIndex; i < nums.length; i++) {
            path.add(nums[i]);
            getSubsets(i + 1, nums, path, res);
            path.remove(path.size() - 1);
        }
    }

    public List<String> letterCombinations(String digits) {
        if (digits.length() == 0) {
            return new ArrayList<>();
        }
        HashMap<Character, String> phoneMap = new HashMap<Character, String>() {{
            put('2', "abc");
            put('3', "def");
            put('4', "ghi");
            put('5', "jkl");
            put('6', "mno");
            put('7', "pqrs");
            put('8', "tuv");
            put('9', "wxyz");
        }};
        ArrayList<String> res = new ArrayList<>();
        getLetterCombinations(digits, res, phoneMap, new StringBuilder());
        return res;

    }

    private void getLetterCombinations(String digits, ArrayList<String> res, HashMap<Character, String> phoneMap, StringBuilder sb) {
        if (sb.length() == digits.length()) {
            res.add(sb.toString());
            return;
        }
        // 得到数字
        char num = digits.charAt(sb.length());
        // 得到字符
        String chs = phoneMap.get(num);
        for (int i = 0; i < chs.length(); i++) {
            sb.append(chs.charAt(i));
            getLetterCombinations(digits, res, phoneMap, sb);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        ArrayList<List<Integer>> res = new ArrayList<>();
        getCombinationSum(0, 0, new ArrayList<>(), res, candidates, target);
        return res;
    }

    private void getCombinationSum(int startIndex, int sum, ArrayList<Integer> path, ArrayList<List<Integer>> res, int[] candidates, int target) {
        if (sum == target) {
            res.add(new ArrayList<>(path));
            return;
        }
        if (sum > target) {
            return;
        }
        for (int i = startIndex; i < candidates.length; i++) {
            // 剪枝
            if (sum + candidates[i] > target) {
                break;
            }
            path.add(candidates[i]);
            getCombinationSum(i, sum + candidates[i], path, res, candidates, target);
            path.remove(path.size() - 1);
        }
    }

    public List<String> generateParenthesis(int n) {
        ArrayList<String> res = new ArrayList<>();
        getParenthesis(n, n, "", res);
        return res;
    }

    private void getParenthesis(int left, int right, String s, List<String> res) {
        // 记录剩余可用的括号个数
        if (left == 0 && right == 0) {
            res.add(s);
        }
        if (left > 0) {
            getParenthesis(left - 1, right, s + "(", res);
        }
        // 如果前面加过一个左括号，则也可以加一个右括号
        if (right > left) {
            getParenthesis(left, right - 1, s + ")", res);
        }
    }

}
