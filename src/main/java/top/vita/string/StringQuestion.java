package top.vita.string;

import cn.hutool.core.collection.CollectionUtil;

import java.util.*;

public class StringQuestion {

	public static void main(String[] args) {
//		String  pattern = "aaa";
//		String s = "aa aa aa aa";
//		boolean wordPattern = wordPattern(pattern, s);
//		System.out.println(wordPattern);

//		String s = "ababcbacadefegdehijhklij";
//		List<Integer> partitionLabels = partitionLabels(s);
//		System.out.println(partitionLabels);

//		String[] strs = {"eat", "tea", "tan", "ate", "nat", "bat"};
//		List<List<String>> groupAnagrams = groupAnagrams(strs);
//		System.out.println(groupAnagrams);
//		System.out.println(lengthOfLongestSubstring("abcabcbb"));
//		multiply("123", "456");
//		lengthOfLongestSubstring12(" ");
		
		int manacher = manacher("babad");
		System.out.println(manacher);
	}

	/*
	 * 415. 字符串相加 给定两个字符串形式的非负整数 num1 和num2 ， 计算它们的和并同样以字符串形式返回。
	 */
	public String addStrings(String num1, String num2) {
		StringBuilder res = new StringBuilder();
		int carry = 0;
		int l1 = num1.length() - 1;
		int l2 = num2.length() - 1;
		while (l1 >= 0 || l2 >= 0) {
			int x = l1 < 0 ? 0 : num1.charAt(l1) - '0';
			int y = l2 < 0 ? 0 : num2.charAt(l2) - '0';

			int sum = x + y + carry;
			res.append(sum % 10);
			carry = sum / 10;

			l1--;
			l2--;
		}
		if (carry != 0) {
			res.append(carry);
		}
		return res.reverse().toString();
	}

	/*
	 * 66. 加一 给定一个由 整数 组成的 非空 数组所表示的非负整数， 在该数的基础上加一。
	 */
	public int[] plusOne(int[] digits) {
		for (int i = digits.length - 1; i >= 0; i--) {
			if (digits[i] != 9) {
				digits[i]++;
				return digits;
			}
			digits[i] = 0;
		}
		int[] res = new int[digits.length + 1];
		res[0] = 1;
		return res;
	}

	/*
	 * 1. 两数之和 给定一个整数数组 nums 和一个整数目标值 target， 请你在该数组中找出 和为目标值 target 的那 两个
	 * 整数，并返回它们的数组下标。
	 */
	public int[] twoSum(int[] nums, int target) {
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < nums.length; i++) {
			int temp = target - nums[i];
			if (map.containsKey(temp)) {
				return new int[] { map.get(temp), i };
			}
			map.put(nums[i], i);
		}
		return new int[] {};
	}

	/*
	 * 454. 四数相加 II 给你四个整数数组 nums1、nums2、nums3 和 nums4 ， 数组长度都是 n
	 */
	public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>(nums1.length);
		int temp;
		int res = 0;
		// 将前两个数组的 总和(key) 的 出现次数(val) 放入map
		for (int i : nums1) {
			for (int j : nums2) {
				temp = i + j;
				if (map.containsKey(temp)) {
					map.put(temp, map.get(temp) + 1);
				} else {
					map.put(temp, 1);
				}
			}
		}
		for (int i : nums3) {
			for (int j : nums4) {
				temp = i + j;
				// 用 0 - ( i + j )判断和这个数相加是否等于0
				if (map.containsKey(0 - temp)) {
					res += map.get(0 - temp);
				}
			}
		}

		return res;
	}

	/*
	 * 344. 反转字符串 编写一个函数，其作用是将输入的字符串反转过来。 输入字符串以字符数组 s 的形式给出。
	 */
	public void reverseString(char[] s) {
		int left = 0;
		int right = s.length - 1;
		char temp = 0;
		for (int i = 0; i < s.length / 2; i++) {
			temp = s[left];
			s[left] = s[right];
			s[right] = temp;
			left++;
			right--;
		}
	}

	/*
	 * 541. 反转字符串 II 给定一个字符串 s 和一个整数 k， 从字符串开头算起，每计数至 2k 个字符， 就反转这 2k 字符中的前 k 个字符。
	 */
	public String reverseStr(String s, int k) {
		char[] ch = s.toCharArray();
		int start = 0;
		int end = 0;
		for (int i = 0; i < ch.length; i += 2 * k) {
			// 此次反转的起点
			start = i;
			// 此次反转的终点，如果尾数不够k个则全部反转
			end = Math.min(ch.length - 1, start + k - 1);

			// 用异或运算反转交换
			while (start < end) {
				ch[start] ^= ch[end];
				ch[end] ^= ch[start];
				ch[start] ^= ch[end];
				start++;
				end--;
			}
		}
		return new String(ch);
	}

	/*
	 * 剑指 Offer 05. 替换空格 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
	 */
	public String replaceSpace(String s) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == ' ') {
				sb.append("%20");
			} else {
				sb.append(s.charAt(i));
			}
		}
		return sb.toString();
	}

	/*
	 * 290. 单词规律 给定一种规律 pattern 和一个字符串 s ，判断 s 是否遵循相同的规律。
	 */
	public static boolean wordPattern(String pattern, String s) {
		String[] str = s.split(" ");
		if (pattern.length() != str.length) {
			return false;
		}
		HashMap<Character, String> map = new HashMap<Character, String>();

		for (int i = 0; i < pattern.length(); i++) {
			char temp = pattern.charAt(i);
			// 如果map里有这个key
			if (map.containsKey(temp)) {
				// 如果 value 的值不等于 put 时对应的值
				if (!map.get(temp).equals(str[i])) {
					return false;
				}
			} else {
				// 如果map里没有这个key，但是已经有 str[i] 的值了
				if (map.containsValue(str[i])) {
					return false;
				} else {
					// 字符key， 单词value
					// 一个字符对应一个单词
					map.put(temp, str[i]);
				}
			}
		}
		return true;
	}

	/*
	 * 763. 划分字母区间 字符串 S由小写字母组成。 我们要把这个字符串划分为尽可能多的片段， 同一字母最多出现在一个片段中。
	 * 返回一个表示每个字符串片段的长度的列表。
	 */
	public static List<Integer> partitionLabels(String s) {
		int[] last = new int[26];
		// 获取每个字符最后出现的位置
		for (int i = 0; i < s.length(); i++) {
			last[s.charAt(i) - 'a'] = i;
		}
		int start = 0, end = 0;
		List<Integer> partition = new ArrayList<>();
		for (int i = 0; i < s.length(); i++) {
			// 取最大值，因为距离不能比最后出现的位置小
			end = Math.max(end, last[s.charAt(i) - 'a']);
			// 如果 i 到达了 end 则加入结果集
			if (i == end) {
				// 将距离存入结果集
				partition.add(end - start + 1);
				// end 已经加入结果集，从end + 1开始
				start = end + 1;
			}
		}
		return partition;
	}

	/*
	 * 49. 字母异位词分组 给你一个字符串数组，请你将 字母异位词 组合在一起。 可以按任意顺序返回结果列表。
	 */
	public static List<List<String>> groupAnagrams(String[] strs) {
		HashMap<String, ArrayList<String>> map = new HashMap<String, ArrayList<String>>();
		for (String s : strs) {
			char[] ch = s.toCharArray();
			// 拥有相同的字符的话，排序后都是一样的
			Arrays.sort(ch);
			String key = String.valueOf(ch);
			if (!map.containsKey(key)) {
				map.put(key, new ArrayList());
			}
			map.get(key).add(s);
		}
		return new ArrayList(map.values());
	}

	/*
	 * 3. 无重复字符的最长子串 给定一个字符串 s ， 请你找出其中不含有重复字符的 最长子串 的长度。
	 */
	public static int lengthOfLongestSubstring(String s) {
		return 0;
	}

	/*
	 * 387. 字符串中的第一个唯一字符 给定一个字符串 s ，找到 它的第一个不重复的字符，并返回它的索引 。 如果不存在，则返回 -1 。
	 */
	public int firstUniqChar(String s) {
		for (int i = 0; i < s.length(); i++) {
			char ch = s.charAt(i);
			if (s.indexOf(ch) == s.lastIndexOf(ch)) {
				return i;
			}
		}
		return -1;
	}

	/*
	 * 557. 反转字符串中的单词 III 给定一个字符串 s ，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。
	 */
	public String reverseWords(String s) {
		StringBuilder sb = new StringBuilder();
		String[] split = s.split(" ");
		for (int i = 0; i < split.length; i++) {
			String string = split[i];
			String reverse = reverse(string, 0, string.length() - 1);
			if (i != split.length - 1) {
				sb.append(reverse).append(" ");
			} else {
				sb.append(reverse);
			}
		}
		return sb.toString();
	}

	private String reverse(String s, int begin, int end) {
		char[] cs = s.toCharArray();

		while (begin < end) {
			char temp = cs[begin];
			cs[begin] = cs[end];
			cs[end] = temp;
			begin++;
			end--;
		}

		return new String(cs);
	}

//    /*
//	 * 1374. 生成每种字符都是奇数个的字符串
//	 *  给你一个整数 n，请你返回一个含 n 个字符的字符串，
//	 *  其中每种字符在该字符串中都恰好出现 奇数次 。
//	 */
//    public String generateTheString(int n) {
//    	StringBuilder sb = new StringBuilder();
//    	while (n-- > 0) {
//    		
//    	}
//    }

	/*
	 * 3. 无重复字符的最长子串 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
	 */
	public int lengthOfLongestSubstring1(String s) {
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		int maxSize = 0;
		for (int i = 0; i < s.length(); i++) {
			char ch = s.charAt(i);
			if (map.containsKey(ch)) {
				map.remove(ch);
//				map.clear();
			}
			map.put(ch, 1);
			maxSize = Math.max(maxSize, map.size());
		}
		return maxSize;
	}

	/*
	 * 43. 字符串相乘
	 */
	public static String multiply(String num1, String num2) {
		if ("0".equals(num1) || "0".equals(num2)) {
			return "0";
		}
		int[] res = new int[num1.length() + num2.length()];
		for (int i = num1.length() - 1; i >= 0; i--) {
			int value1 = num1.charAt(i) - '0';
			for (int j = num2.length() - 1; j >= 0; j--) {
				int value2 = num2.charAt(j) - '0';
				// 原本位置的结果加上相乘后的结果
				int sum = res[i + j + 1] + value1 * value2;
				res[i + j + 1] = sum % 10;
				// 保存进位
				res[i + j] += sum / 10;
			}
		}
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < res.length; i++) {
			// 如果运算结果的第一位为0，则不加入
			if (i == 0 && res[i] == 0) {
				continue;
			}
			sb.append(res[i]);
		}
		return sb.toString();
	}

	/*
	 * 3. 无重复字符的最长子串
	 */
	public static int lengthOfLongestSubstring12(String s) {
		int[] last = new int[128];
		for (int i = 0; i < last.length; i++) {
			last[i] = -1;
		}
		int max = 0;
		int start = 0;
		for (int i = 0; i < s.length(); i++) {
			char ch = s.charAt(i);
			start = Math.max(start, last[ch] + 1);
			max = Math.max(i - start + 1, max);
			last[ch] = i;
		}
		return max;
	}

	/*
	 * 1455. 检查单词是否为句中其他单词的前缀
	 */
	public int isPrefixOfWord(String sentence, String searchWord) {
		String[] strs = sentence.split(" ");
		for (int i = 0; i < strs.length; i++) {
			if (strs[i].startsWith(searchWord)) {
				return i + 1;
			}
		}
		return -1;
	}

	/*
	 * 2114. 句子中的最多单词数
	 */
	public int mostWordsFound(String[] sentences) {
		int max = 0;
		for (int i = 0; i < sentences.length; i++) {
			max = Math.max(max, sentences[i].split(" ").length);
		}
		return max;
	}

	/*
	 * 1684. 统计一致字符串的数目
	 */
	public int countConsistentStrings(String allowed, String[] words) {
		int[] str = new int[26];
		for (int i = 0; i < allowed.length(); i++) {
			str[allowed.charAt(i) - 'a'] = 1;
		}
		int count = 0;
		// outer:for
		outer: for (int i = 0; i < words.length; i++) {
			for (int j = 0; j < words[i].length(); j++) {
				if (str[words[i].charAt(j)] != 1) {
					// 直接跳过整个大循环 同理还有break outer
					continue outer;
				}
			}
			count++;
		}
		return count;
	}

	/*
	 * 459. 重复的子字符串
	 */
	public boolean repeatedSubstringPattern(String s) {
		char[] chs = s.toCharArray();
		if (chs.length == 1) {
			return false;
		}
		int[] next = new int[chs.length];
		int j = 0;
		// java中可省略
		next[0] = 0;
		// 位置的信息为 从0到本身(i)的相同前后缀数，而不是0到本身的前一位(i - 1)的相同前后缀数
		for (int i = 1; i < chs.length; i++) {
			while (j > 0 && chs[j] != chs[i]) {
				j = next[j - 1];
			}
			if (chs[i] == chs[j]) {
				j++;
			}
			next[i] = j;
		}
		int n = s.length();
		if (next[n - 1] != 0 && n % (n - next[n - 1]) == 0) {
			return true;
		}
		return false;
	}

	public int[] getNext(char[] chs) {
		if (chs.length == 1) {
			return new int[] { 1 };
		}
		int[] next = new int[chs.length];
		next[0] = -1;
		// 可省略
		next[1] = 0;
		int i = 2;
		int cn = 0;
		while (i < chs.length) {
			if (chs[i - 1] == chs[cn]) {
				// 将当前位置的信息记为前一位的信息加1
				next[i++] = ++cn;
			} else if (cn > 0) {
				// 当前位置的字符和i-1匹配不上，将cn移动到当前cn对应的位置（当前位置的前缀的后一位）
				cn = next[cn];
			} else {
				// 没有前缀，将当前信息设置为0
				next[i++] = 0;
			}
		}
		return next;
	}

	/*
	 * 13. 罗马数字转整数
	 */
	public int romanToInt(String s) {
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		map.put('I', 1);
		map.put('V', 5);
		map.put('X', 10);
		map.put('L', 50);
		map.put('C', 100);
		map.put('D', 500);
		map.put('M', 1000);
		int res = 0;
		char[] chs = s.toCharArray();
		for (int i = 0; i < chs.length - 1; i++) {
			// 字符代表的值不小于其右边
			if (map.get(chs[i]) >= map.get(chs[i + 1])) {
				res += map.get(chs[i]);
			} else { // 小于右边，减去当前
				res -= map.get(chs[i]);
			}
		}
		// 加上最后一位
		res += map.get(chs[chs.length - 1]);
		return res;
	}

	/*
	 * 12. 整数转罗马数字
	 */
	public String intToRoman(int num) {
		int values[] = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
		String strs[] = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };
		StringBuilder res = new StringBuilder();
		for (int i = 0; i < values.length; i++) {
			int value = values[i];
			String str = strs[i];
			while (num >= value) {
				num -= value;
				res.append(str);
			}
			if (num == 0) {
				break;
			}
		}
		return res.toString();
	}

	/*
	 * 16. 最接近的三数之和
	 */
	public int threeSumClosest(int[] nums, int target) {
		int res = Integer.MAX_VALUE;
		Arrays.sort(nums);
		// 减2是为了内部循环
		for (int i = 0; i < nums.length - 2; i++) {
			int left = i + 1;
			int right = nums.length - 1;
			while (left < right) {
				int sum = nums[left] + nums[right] + nums[i];
				res = Math.abs(target - res) > Math.abs(target - sum) ? sum : res;
				if (sum > target) {
					right--;
				} else if (sum < target) {
					left++;
				} else {
					return res;
				}
			}
		}
		return res;
	}

	/*
	 * 17. 电话号码的字母组合
	 */
	List<String> res = new ArrayList<String>();
	HashMap<Character, String> phoneMap = new HashMap<Character, String>() {
		{
			put('2', "abc");
			put('3', "def");
			put('4', "ghi");
			put('5', "jkl");
			put('6', "mno");
			put('7', "pqrs");
			put('8', "tuv");
			put('9', "wxyz");
		}
	};

	public List<String> letterCombinations(String digits) {
		if (digits.length() == 0) {
			return res;
		}
		combine(digits, 0, new StringBuilder());
		return res;
	}

	private void combine(String digits, int i, StringBuilder temp) {
		if (i == digits.length()) {
			res.add(temp.toString());
		} else {
			char ch = digits.charAt(i);
			String phone = phoneMap.get(ch);
			for (int j = 0; j < phone.length(); j++) {
				temp.append(phone.charAt(j));
				combine(digits, i + 1, temp);
				temp.deleteCharAt(i);
			}
		}
	}

	/*
	 * 18. 四数之和
	 */
	public List<List<Integer>> fourSum(int[] nums, int target) {
		if (nums.length < 4) {
			return new ArrayList<List<Integer>>();
		}
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		Arrays.sort(nums);
		for (int i = 0; i < nums.length - 3; i++) {
			// 跳过重复
			if (i > 0 && nums[i] == nums[i - 1]) {
				continue;
			}
			for (int j = i + 1; j < nums.length - 2; j++) {
				// 跳过重复
				if (j > i + 1 && nums[j] == nums[j - 1]) {
					continue;
				}
				int left = j + 1;
				int right = nums.length - 1;
				while (left < right) {
					long sum = (long) nums[i] + nums[j] + nums[left] + nums[right];
					if (sum > target) {
						right--;
					} else if (sum < target) {
						left++;
					} else {
						List<Integer> list = new ArrayList<Integer>();
						list.add(nums[i]);
						list.add(nums[j]);
						list.add(nums[left]);
						list.add(nums[right]);
						res.add(list);
						left++;
						right--;
						while (left < right && nums[left] == nums[left - 1])
							left++;
						while (left < right && nums[right] == nums[right + 1])
							right--;
					}
				}
			}
		}
		return res;
	}

	/*
	 * 22. 括号生成
	 */
	public List<String> generateParenthesis(int n) {
		List<String> res = new ArrayList<String>();
		generate(n, n, "", res);
		return res;
	}

	private void generate(int left, int right, String str, List<String> res) {
		if (left == 0 && right == 0) {
			res.add(str);
			return;
		}
		if (left > 0) {
			generate(left - 1, right, str + "(", res);
		}
		if (right > 0) {
			// 如果右括号剩余多于左括号剩余的话，可以拼接右括号
			generate(left, right - 1, str + ")", res);
		}
	}

	/*
	 * 28. 实现 strStr()
	 */
	public int strStr(String haystack, String needle) {
		char[] chs1 = haystack.toCharArray();
		char[] chs2 = needle.toCharArray();

		// 获取next
		int[] next = new int[needle.length()];
		int j = 0;
		for (int i = 1; i < next.length; i++) {
			while (j > 0 && next[i] != next[j]) {
				j = next[j - 1];
			}
			if (chs2[i] == chs2[j]) {
				next[i] = ++j;
			}
		}

		// KMP
		int i = 0;
		j = 0;
		while (i < chs1.length) {
			if (chs1[i] == chs2[j]) {
				i++;
				j++;
			} else if (j > 0) {
				j = next[j - 1];
			} else {
				i++;
			}
			if (j == chs2.length) {
				return i - j;
			}
		}

		return -1;
	}

	/*
	 * 5. 最长回文子串
	 */
	public String longestPalindrome(String s) {
		String res = "";
		for (int i = 0; i < s.length(); i++) {
			// 奇数时，由一个中心点向两边扩散
			String s1 = palindrome(s, i, i);
			// 偶数时，由中间的两个中心点向两边扩散
			String s2 = palindrome(s, i, i + 1);
			
			res = res.length() > s1.length() ? res : s1;
			res = res.length() > s2.length() ? res : s2;
		}
		return res;
	}

	private String palindrome(String s, int left, int right) {
		while (left >= 0 && right < s.length()) {
			if (s.charAt(left) == s.charAt(right)) {
				// 向左右扩散
				left--;
				right++;
			}else {
				break;
			}
		}
		return s.substring(left + 1, right);
	}

    public static int manacher(String s) {
    	char[] str = manacherString(s);
    	int[] pArr = new int[str.length];
    	// 圆心
    	int C = -1;
    	// 右半径
    	int R = -1;
    	int max = Integer.MIN_VALUE;
    	for (int i = 0; i < str.length; i++) {
			pArr[i] = R > i ? Math.min(pArr[2 * C - i], R - i) : 1;
			
			while(i + pArr[i] < str.length && i - pArr[i] > -1) {
				// i + pArr[i] 就是 i'，是 i 的对称
				// i - pArr[i] 即为 i
				if (str[i + pArr[i]] == str[i - pArr[i]]) {
					pArr[i]++;
				}else {
					break;
				}
			}
			if (i + pArr[i] > R) {
				R = i + pArr[i];
				C = i;
			}
			max = Math.max(max, pArr[i]);
		}
    	
        int maxR = 0;
        int maxC = 0;
        for(int i = 0; i < pArr.length; i++){
            if(pArr[i]>maxR){
                maxR = pArr[i];
                maxC = i;
            }
        }
    	return max - 1;
    }

	private static char[] manacherString(String s) {
		char[] charArr = s.toCharArray();
		char[] res = new char[s.length() * 2 + 1];
		int index = 0;
		for (int i = 0; i < res.length; i++) {
			res[i] = (i & 1) == 0 ? '#' : charArr[index++];
		}
		return res;
	}

	/**
	 * 344. 反转字符串
	 */
	public void reverseString1(char[] s) {
		int left = 0;
		int right = s.length - 1;
		while (left < right){
			char tmp = s[left];
			s[left] = s[right];
			s[right] = tmp;
			left++;
			right--;
		}
	}

	/**
	 * 541. 反转字符串 II
	 */
	public String reverseStr1(String s, int k) {
		char[] chs = s.toCharArray();
		for (int i = 0; i < chs.length; i += 2 * k) {
			int left = i;
			int right = Math.min(s.length(), left + k - 1);
			while (left < right) {
				char tmp = chs[left];
				chs[left] = chs[right];
				chs[right] = tmp;
				left++;
				right--;
			}
		}
		return new String(chs);
	}

	/**
	 * 剑指 Offer 05. 替换空格
	 */
	public String replaceSpace2(String s) {
		if(s == null || s.length() == 0){
			return s;
		}
		StringBuilder space = new StringBuilder();
		for (char ch : s.toCharArray()) {
			if (ch == ' '){
				// 加上两个空格，用来放%20
				space.append("  ");
			}
		}
		if (space.length() == 0){
			return s;
		}
		// 指向加入空格前的最后一位字符
		int left = s.length() - 1;
		// 将放置%20所需要的空间加入字符串，底层使用的是StringBuilder的append方法然后toString()
		s += space;
		// 指向最后一个空格字符
		int right = s.length() - 1;
		char[] chs = s.toCharArray();
		while (left >= 0){
			if (chs[left] == ' '){
				// 填充字符串
				chs[right--] = '0';
				chs[right--] = '2';
				chs[right] = '%';
			} else{
				chs[right] = chs[left];
			}
			left--;
			right--;
		}
		return new String(chs);
	}

	/**
	 * 151. 反转字符串中的单词
	 */
	public String reverseWords1(String s) {
		StringBuilder sb = removeSpace(s);
		reverseStrings(sb, 0, sb.length() - 1);
		reverseWord(sb);
		return sb.toString();
	}

	private StringBuilder removeSpace(String s) {
		int left = 0;
		int right = s.length() - 1;
		while (s.charAt(left) == ' '){
			left++;
		}
		while (s.charAt(right) == ' '){
			right--;
		}
		StringBuilder sb = new StringBuilder();
		while (left <= right){
			char ch = s.charAt(left);
			if (ch != ' ' || sb.charAt(sb.length() - 1) != ' '){
				sb.append(ch);
			}
			left++;
		}
		return sb;
	}

	private void reverseStrings(StringBuilder sb, int left, int right) {
		while (left < right) {
			char tmp = sb.charAt(left);
			sb.setCharAt(left, sb.charAt(right));
			sb.setCharAt(right, tmp);
			left++;
			right--;
		}
	}

	private void reverseWord(StringBuilder sb) {
		int left = 0;
		int right = 1;
		while (left < sb.length()){
			// 找到单词中最后一个字符的位置
			while (right < sb.length() && sb.charAt(right) != ' '){
				right++;
			}
			reverseStrings(sb, left, right - 1);
			// 指向后一个单词
			left = right + 1;
			right = left + 1;
		}
	}

	/**
	 * 剑指 Offer 58 - II. 左旋转字符串
	 */
	public String reverseLeftWords(String s, int n) {
		int len = s.length();
		StringBuilder sb = new StringBuilder(s);
		reverseString(sb, 0, n - 1);
		reverseString(sb, n, len - 1);
		return sb.reverse().toString();
	}

	public void reverseString(StringBuilder sb, int start, int end) {
		while (start < end) {
			char temp = sb.charAt(start);
			sb.setCharAt(start, sb.charAt(end));
			sb.setCharAt(end, temp);
			start++;
			end--;
		}
	}
	public String reverseWords12(String s) {
		// 将传进来的字符串以空格拆分
		String[] strings = s.trim().split(" ");
		StringBuilder sb = new StringBuilder();
		// 从后往前遍历
		for (int i = strings.length - 1; i >= 0; i--) {
			// 去除多出来的空格
			if (strings[i].equals("")) {
				continue;
			}
			// 到头了，append然后去空格
			if (i == 0) {
				sb.append(strings[i].trim());
			} else {
				// 怕有多余的空格，去掉，再加上去
				sb.append(strings[i].trim()).append(" ");
			}
		}
		// 输出String
		return sb.toString();
	}

	public char firstUniqChar1(String s) {
		Map<Character, Integer> map = new LinkedHashMap<>();
		for (char ch : s.toCharArray()){
			if (map.containsKey(ch)){
				map.put(ch, map.get(ch) + 1);
			} else {
				map.put(ch, 1);
			}
		}
		for (Map.Entry<Character, Integer> entry : map.entrySet()){
			if (entry.getValue() == 1) {
				return entry.getKey();
			}
		}
		return ' ';
	}

	public String minWindow(String s, String t) {
		if (s.length() < t.length()) {
			return "";
		}
		int[] hash = new int['z' - 'A' + 1];
		for (char c : t.toCharArray()) {
			hash[c - 'A']++;
		}
		int left = 0;
		int right = 0;
		// 需要的字符的数量，也就是t字符串的字符的个数
		int count = t.length();
		// 结果字符串的起始位置
		int begin = -1;
		// 符合需求的最小子串的长度
		int size = s.length() + 1;
		char[] chs = s.toCharArray();
		while (right < s.length()) {
			// 找到一个t的字符
			// 循环减少需要的个数，不是需要的字符的话会被减到-1甚至更小
			if (hash[chs[right] - 'A']-- > 0) {
				// 减少需要的字符个数
				count--;
			}
			// 直接在这里加，省的在后面计算结果的时候还要加一
			right++;
			// 已经找齐t字符串的字符
			if (count == 0) {
				// 开始从左窗口开始尽量缩小窗口
				// 只要左窗口找到一个t的字符，缩小窗口就停止
				while (hash[chs[left] - 'A']++ < 0) {
					// 恢复之前减去的字符个数
					left++;
				}
				// 记录长度和起始位置
				if (right - left < size) {
					size = right - left;
					begin = left;
				}
				// 因为移动了已经有需要字符的左窗口，所以需要的字符加一
				count++;
				left++;
			}
		}
		return begin == -1 ? "" : new String(chs, begin, size);
	}

	public int maxPower(String s) {
		char[] chs = s.toCharArray();
		int i = 0;
		int max = 0;
		while (i < chs.length) {
			int start = i;
			while (i < chs.length - 1 && chs[i] == chs[i + 1]) {
				i++;
			}
			max = Math.max(max, i - start + 1);
			i++;
		}
		return max;
	}

	public boolean checkZeroOnes(String s) {
		char[] chs = s.toCharArray();
		int i = 0;
		int max0 = 0;
		int max1 = 0;
		while (i < chs.length) {
			int start = i;
			while (i < chs.length - 1 && chs[i] == chs[i + 1]) {
				i++;
			}
			if (chs[start] == '0') {
				max0 = Math.max(max0, i - start + + 1);
			} else {
				max1 = Math.max(max1, i - start + + 1);
			}
			i++;
		}
		return max1 > max0;
	}

	public String makeFancyString(String s) {
		StringBuilder sb = new StringBuilder();
		char[] chs = s.toCharArray();
		int i = 0;
		while (i < chs.length) {
			int start = i;
			while (i < chs.length - 1 && chs[i] == chs[i + 1]) {
				i++;
			}
			sb.append(chs[start]);
			// 如果i移动的了话，直接加一个相同的字符就可以了，因为相同字符不能超过两个
			if (start < i) {
				sb.append(chs[i]);
			}
			i++;
		}
		return sb.toString();
	}

	public boolean winnerOfGame(String colors) {
		char[] chs = colors.toCharArray();
		int i = 0;
		int a = 0;
		int b = 0;
		while (i < chs.length) {
			int start = i;
			while (i < chs.length - 1 && chs[i] == chs[i + 1]) {
				i++;
			}
			// 三个相邻以上的就删除一个，中间留下2个
			if (i - start + 1 >= 3) {
				// 删除成功就加次数
				if (chs[start] == 'A') {
					a += (i - start + 1) - 2;
				} else {
					b += (i - start + 1) - 2;
				}
			}
			i++;
		}
		return a > b;
	}

	public List<String> findRepeatedDnaSequences(String s) {
		ArrayList<String> res = new ArrayList<>();
		HashMap<String, Integer> map = new HashMap<>();
		for (int i = 0; i <= s.length() - 10; i++) {
			String sub = s.substring(i, i + 10);
			map.put(sub, map.getOrDefault(sub, 0) + 1);
			if (map.get(sub) == 2) {
				res.add(sub);
			}
		}
		return res;
	}

	public int vowelStrings(String[] words, int left, int right) {
		int count = 0;
		for (int i = left; i <= right; i++) {
			char[] chs = words[i].toCharArray();
			if (test(chs[0]) && test(chs[chs.length - 1])) {
				count++;
			}
		}
		return count;
	}

	public boolean test(char word) {
		if (word == 'a' || word == 'e' || word == 'i' || word == 'o' || word == 'u') {
			return true;
		}
		return false;
	}

	public int[] vowelStrings(String[] words, int[][] queries) {
		Set<Character> vowels = CollectionUtil.newHashSet('a', 'e', 'i', 'o', 'u');
		int n = words.length;
		int[] s = new int[n + 1];
		for (int i = 0; i < n; ++i) {
			char a = words[i].charAt(0);
			char b = words[i].charAt(words[i].length() - 1);
			s[i + 1] = s[i] + (vowels.contains(a) && vowels.contains(b) ? 1 : 0);
		}
		int m = queries.length;
		int[] res = new int[m];
		for (int i = 0; i < m; ++i) {
			int l = queries[i][0];
			int r = queries[i][1];
			// 取[l, r]之间符合条件的个数
			res[i] = s[r + 1] - s[l];
		}
		return res;
	}
}
