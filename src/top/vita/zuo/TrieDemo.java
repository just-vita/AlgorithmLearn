package top.vita.zuo;

public class TrieDemo {

	public static class TrieNode {
		public int path;
		public int end;
		public TrieNode[] nexts;

		public TrieNode() {
			path = 0;
			end = 0;
			// 26¸öÐ¡Ð´×ÖÄ¸
			nexts = new TrieNode[26];
		}
	}

	public static class Trie {

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
			for (int i = 0; i < chs.length; i++) {
				index = chs[i] - 'a';
				if (node.nexts[index] == null) {
					node.nexts[index] = new TrieNode();
				}
				node = node.nexts[index];
				node.path++;
			}
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
				if (node.nexts[index] == null) {
					return false;
				}
				node = node.nexts[index];
			}
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
			if (node.path == 0) {
				return false;
			}
			return true;
		}
	}
}
