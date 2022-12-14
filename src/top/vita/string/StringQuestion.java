package top.vita.string;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

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
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
