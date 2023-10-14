package top.vita.tree;

class Trie {

    public static class TrieNode {
        public int end;
        public TrieNode[] nexts;

        public TrieNode() {
            // ��Ϊ��β�Ĵ���
            end = 0;
            // 26��Сд��ĸ
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
        // ���ַ�����ÿ���ַ�������ǰ׺���ڵ㣬���ҵ����һ���ַ�
        for (int i = 0; i < chs.length; i++) {
            index = chs[i] - 'a';
            if (node.nexts[index] == null) {
                node.nexts[index] = new TrieNode();
            }
            node = node.nexts[index];
        }
        // ��ǰ�ַ���Ϊ�ַ�����β�Ĵ�����һ
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
            // ��ǰ�ַ���ǰ׺����������ȫ�������ַ���������
            if (node.nexts[index] == null) {
                return false;
            }
            node = node.nexts[index];
        }
        // �ɹ������ַ����Ľ�β�ַ������ַ���û�г�Ϊ����β�ַ�����������ַ���ֻ��ǰ׺
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
        // ֻҪǰ׺�����������ʹ�������ĳ���ַ�����ǰ׺
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