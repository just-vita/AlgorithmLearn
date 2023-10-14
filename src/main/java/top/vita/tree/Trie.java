package top.vita.tree;

class Trie {

    public static class TrieNode {
        public int end;
        public TrieNode[] nexts;

        public TrieNode() {
            // 作为结尾的次数
            end = 0;
            // 26个小写字母
            nexts = new TrieNode[26];
        }
    }

    private TrieNode root;

    public Trie() {
        root = new TrieNode();
    }

    public void insert(String word) {
        if (word == null) {
            return;
        }
        char[] chs = word.toCharArray();
        TrieNode node = root;
        int index = 0;
        // 给字符串的每个字符创建出前缀树节点，并找到最后一个字符
        for (int i = 0; i < chs.length; i++) {
            index = chs[i] - 'a';
            if (node.nexts[index] == null) {
                node.nexts[index] = new TrieNode();
            }
            node = node.nexts[index];
        }
        // 当前字符作为字符串结尾的次数加一
        node.end++;
    }

    public boolean search(String word) {
        if (word == null) {
            return false;
        }
        char[] chs = word.toCharArray();
        TrieNode node = root;
        int index = 0;
        for (int i = 0; i < chs.length; i++) {
            index = chs[i] - 'a';
            // 当前字符的前缀树构建不完全，代表字符串不存在
            if (node.nexts[index] == null) {
                return false;
            }
            node = node.nexts[index];
        }
        // 成功到达字符串的结尾字符，但字符并没有成为过结尾字符，代表这个字符串只是前缀
        if (node.end == 0) {
            return false;
        }
        return true;
    }

    public boolean startsWith(String prefix) {
        if (prefix == null) {
            return false;
        }
        char[] chs = prefix.toCharArray();
        TrieNode node = root;
        int index = 0;
        for (int i = 0; i < chs.length; i++) {
            index = chs[i] - 'a';
            if (node.nexts[index] == null) {
                return false;
            }
            node = node.nexts[index];
        }
        // 只要前缀树构建过，就代表它是某个字符串的前缀
        return true;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */