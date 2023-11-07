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
	 * 415. �ַ������ ���������ַ�����ʽ�ķǸ����� num1 ��num2 �� �������ǵĺͲ�ͬ�����ַ�����ʽ���ء�
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
	 * 66. ��һ ����һ���� ���� ��ɵ� �ǿ� ��������ʾ�ķǸ������� �ڸ����Ļ����ϼ�һ��
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
	 * 1. ����֮�� ����һ���������� nums ��һ������Ŀ��ֵ target�� �����ڸ��������ҳ� ��ΪĿ��ֵ target ���� ����
	 * ���������������ǵ������±ꡣ
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
	 * 454. ������� II �����ĸ��������� nums1��nums2��nums3 �� nums4 �� ���鳤�ȶ��� n
	 */
	public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>(nums1.length);
		int temp;
		int res = 0;
		// ��ǰ��������� �ܺ�(key) �� ���ִ���(val) ����map
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
				// �� 0 - ( i + j )�жϺ����������Ƿ����0
				if (map.containsKey(0 - temp)) {
					res += map.get(0 - temp);
				}
			}
		}

		return res;
	}

	/*
	 * 344. ��ת�ַ��� ��дһ���������������ǽ�������ַ�����ת������ �����ַ������ַ����� s ����ʽ������
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
	 * 541. ��ת�ַ��� II ����һ���ַ��� s ��һ������ k�� ���ַ�����ͷ����ÿ������ 2k ���ַ��� �ͷ�ת�� 2k �ַ��е�ǰ k ���ַ���
	 */
	public String reverseStr(String s, int k) {
		char[] ch = s.toCharArray();
		int start = 0;
		int end = 0;
		for (int i = 0; i < ch.length; i += 2 * k) {
			// �˴η�ת�����
			start = i;
			// �˴η�ת���յ㣬���β������k����ȫ����ת
			end = Math.min(ch.length - 1, start + k - 1);

			// ��������㷴ת����
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
	 * ��ָ Offer 05. �滻�ո� ��ʵ��һ�����������ַ��� s �е�ÿ���ո��滻��"%20"��
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
	 * 290. ���ʹ��� ����һ�ֹ��� pattern ��һ���ַ��� s ���ж� s �Ƿ���ѭ��ͬ�Ĺ��ɡ�
	 */
	public static boolean wordPattern(String pattern, String s) {
		String[] str = s.split(" ");
		if (pattern.length() != str.length) {
			return false;
		}
		HashMap<Character, String> map = new HashMap<Character, String>();

		for (int i = 0; i < pattern.length(); i++) {
			char temp = pattern.charAt(i);
			// ���map�������key
			if (map.containsKey(temp)) {
				// ��� value ��ֵ������ put ʱ��Ӧ��ֵ
				if (!map.get(temp).equals(str[i])) {
					return false;
				}
			} else {
				// ���map��û�����key�������Ѿ��� str[i] ��ֵ��
				if (map.containsValue(str[i])) {
					return false;
				} else {
					// �ַ�key�� ����value
					// һ���ַ���Ӧһ������
					map.put(temp, str[i]);
				}
			}
		}
		return true;
	}

	/*
	 * 763. ������ĸ���� �ַ��� S��Сд��ĸ��ɡ� ����Ҫ������ַ�������Ϊ�����ܶ��Ƭ�Σ� ͬһ��ĸ��������һ��Ƭ���С�
	 * ����һ����ʾÿ���ַ���Ƭ�εĳ��ȵ��б�
	 */
	public static List<Integer> partitionLabels(String s) {
		int[] last = new int[26];
		// ��ȡÿ���ַ������ֵ�λ��
		for (int i = 0; i < s.length(); i++) {
			last[s.charAt(i) - 'a'] = i;
		}
		int start = 0, end = 0;
		List<Integer> partition = new ArrayList<>();
		for (int i = 0; i < s.length(); i++) {
			// ȡ���ֵ����Ϊ���벻�ܱ������ֵ�λ��С
			end = Math.max(end, last[s.charAt(i) - 'a']);
			// ��� i ������ end ���������
			if (i == end) {
				// �������������
				partition.add(end - start + 1);
				// end �Ѿ�������������end + 1��ʼ
				start = end + 1;
			}
		}
		return partition;
	}

	/*
	 * 49. ��ĸ��λ�ʷ��� ����һ���ַ������飬���㽫 ��ĸ��λ�� �����һ�� ���԰�����˳�򷵻ؽ���б�
	 */
	public static List<List<String>> groupAnagrams(String[] strs) {
		HashMap<String, ArrayList<String>> map = new HashMap<String, ArrayList<String>>();
		for (String s : strs) {
			char[] ch = s.toCharArray();
			// ӵ����ͬ���ַ��Ļ����������һ����
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
	 * 3. ���ظ��ַ�����Ӵ� ����һ���ַ��� s �� �����ҳ����в������ظ��ַ��� ��Ӵ� �ĳ��ȡ�
	 */
	public static int lengthOfLongestSubstring(String s) {
		return 0;
	}

	/*
	 * 387. �ַ����еĵ�һ��Ψһ�ַ� ����һ���ַ��� s ���ҵ� ���ĵ�һ�����ظ����ַ����������������� �� ��������ڣ��򷵻� -1 ��
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
	 * 557. ��ת�ַ����еĵ��� III ����һ���ַ��� s ������Ҫ��ת�ַ�����ÿ�����ʵ��ַ�˳��ͬʱ�Ա����ո�͵��ʵĳ�ʼ˳��
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
//	 * 1374. ����ÿ���ַ��������������ַ���
//	 *  ����һ������ n�����㷵��һ���� n ���ַ����ַ�����
//	 *  ����ÿ���ַ��ڸ��ַ����ж�ǡ�ó��� ������ ��
//	 */
//    public String generateTheString(int n) {
//    	StringBuilder sb = new StringBuilder();
//    	while (n-- > 0) {
//    		
//    	}
//    }

	/*
	 * 3. ���ظ��ַ�����Ӵ� ����һ���ַ��� s �������ҳ����в������ظ��ַ��� ��Ӵ� �ĳ��ȡ�
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
	 * 43. �ַ������
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
				// ԭ��λ�õĽ��������˺�Ľ��
				int sum = res[i + j + 1] + value1 * value2;
				res[i + j + 1] = sum % 10;
				// �����λ
				res[i + j] += sum / 10;
			}
		}
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < res.length; i++) {
			// ����������ĵ�һλΪ0���򲻼���
			if (i == 0 && res[i] == 0) {
				continue;
			}
			sb.append(res[i]);
		}
		return sb.toString();
	}

	/*
	 * 3. ���ظ��ַ�����Ӵ�
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
	 * 1455. ��鵥���Ƿ�Ϊ�����������ʵ�ǰ׺
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
	 * 2114. �����е���൥����
	 */
	public int mostWordsFound(String[] sentences) {
		int max = 0;
		for (int i = 0; i < sentences.length; i++) {
			max = Math.max(max, sentences[i].split(" ").length);
		}
		return max;
	}

	/*
	 * 1684. ͳ��һ���ַ�������Ŀ
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
					// ֱ������������ѭ�� ͬ����break outer
					continue outer;
				}
			}
			count++;
		}
		return count;
	}

	/*
	 * 459. �ظ������ַ���
	 */
	public boolean repeatedSubstringPattern(String s) {
		char[] chs = s.toCharArray();
		if (chs.length == 1) {
			return false;
		}
		int[] next = new int[chs.length];
		int j = 0;
		// java�п�ʡ��
		next[0] = 0;
		// λ�õ���ϢΪ ��0������(i)����ͬǰ��׺����������0�������ǰһλ(i - 1)����ͬǰ��׺��
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
		// ��ʡ��
		next[1] = 0;
		int i = 2;
		int cn = 0;
		while (i < chs.length) {
			if (chs[i - 1] == chs[cn]) {
				// ����ǰλ�õ���Ϣ��Ϊǰһλ����Ϣ��1
				next[i++] = ++cn;
			} else if (cn > 0) {
				// ��ǰλ�õ��ַ���i-1ƥ�䲻�ϣ���cn�ƶ�����ǰcn��Ӧ��λ�ã���ǰλ�õ�ǰ׺�ĺ�һλ��
				cn = next[cn];
			} else {
				// û��ǰ׺������ǰ��Ϣ����Ϊ0
				next[i++] = 0;
			}
		}
		return next;
	}

	/*
	 * 13. ��������ת����
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
			// �ַ������ֵ��С�����ұ�
			if (map.get(chs[i]) >= map.get(chs[i + 1])) {
				res += map.get(chs[i]);
			} else { // С���ұߣ���ȥ��ǰ
				res -= map.get(chs[i]);
			}
		}
		// �������һλ
		res += map.get(chs[chs.length - 1]);
		return res;
	}

	/*
	 * 12. ����ת��������
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
	 * 16. ��ӽ�������֮��
	 */
	public int threeSumClosest(int[] nums, int target) {
		int res = Integer.MAX_VALUE;
		Arrays.sort(nums);
		// ��2��Ϊ���ڲ�ѭ��
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
	 * 17. �绰�������ĸ���
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
	 * 18. ����֮��
	 */
	public List<List<Integer>> fourSum(int[] nums, int target) {
		if (nums.length < 4) {
			return new ArrayList<List<Integer>>();
		}
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		Arrays.sort(nums);
		for (int i = 0; i < nums.length - 3; i++) {
			// �����ظ�
			if (i > 0 && nums[i] == nums[i - 1]) {
				continue;
			}
			for (int j = i + 1; j < nums.length - 2; j++) {
				// �����ظ�
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
	 * 22. ��������
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
			// ���������ʣ�����������ʣ��Ļ�������ƴ��������
			generate(left, right - 1, str + ")", res);
		}
	}

	/*
	 * 28. ʵ�� strStr()
	 */
	public int strStr(String haystack, String needle) {
		char[] chs1 = haystack.toCharArray();
		char[] chs2 = needle.toCharArray();

		// ��ȡnext
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
	 * 5. ������Ӵ�
	 */
	public String longestPalindrome(String s) {
		String res = "";
		for (int i = 0; i < s.length(); i++) {
			// ����ʱ����һ�����ĵ���������ɢ
			String s1 = palindrome(s, i, i);
			// ż��ʱ�����м���������ĵ���������ɢ
			String s2 = palindrome(s, i, i + 1);
			
			res = res.length() > s1.length() ? res : s1;
			res = res.length() > s2.length() ? res : s2;
		}
		return res;
	}

	private String palindrome(String s, int left, int right) {
		while (left >= 0 && right < s.length()) {
			if (s.charAt(left) == s.charAt(right)) {
				// ��������ɢ
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
    	// Բ��
    	int C = -1;
    	// �Ұ뾶
    	int R = -1;
    	int max = Integer.MIN_VALUE;
    	for (int i = 0; i < str.length; i++) {
			pArr[i] = R > i ? Math.min(pArr[2 * C - i], R - i) : 1;
			
			while(i + pArr[i] < str.length && i - pArr[i] > -1) {
				// i + pArr[i] ���� i'���� i �ĶԳ�
				// i - pArr[i] ��Ϊ i
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
	 * 344. ��ת�ַ���
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
	 * 541. ��ת�ַ��� II
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
	 * ��ָ Offer 05. �滻�ո�
	 */
	public String replaceSpace2(String s) {
		if(s == null || s.length() == 0){
			return s;
		}
		StringBuilder space = new StringBuilder();
		for (char ch : s.toCharArray()) {
			if (ch == ' '){
				// ���������ո�������%20
				space.append("  ");
			}
		}
		if (space.length() == 0){
			return s;
		}
		// ָ�����ո�ǰ�����һλ�ַ�
		int left = s.length() - 1;
		// ������%20����Ҫ�Ŀռ�����ַ������ײ�ʹ�õ���StringBuilder��append����Ȼ��toString()
		s += space;
		// ָ�����һ���ո��ַ�
		int right = s.length() - 1;
		char[] chs = s.toCharArray();
		while (left >= 0){
			if (chs[left] == ' '){
				// ����ַ���
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
	 * 151. ��ת�ַ����еĵ���
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
			// �ҵ����������һ���ַ���λ��
			while (right < sb.length() && sb.charAt(right) != ' '){
				right++;
			}
			reverseStrings(sb, left, right - 1);
			// ָ���һ������
			left = right + 1;
			right = left + 1;
		}
	}

	/**
	 * ��ָ Offer 58 - II. ����ת�ַ���
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
		// �����������ַ����Կո���
		String[] strings = s.trim().split(" ");
		StringBuilder sb = new StringBuilder();
		// �Ӻ���ǰ����
		for (int i = strings.length - 1; i >= 0; i--) {
			// ȥ��������Ŀո�
			if (strings[i].equals("")) {
				continue;
			}
			// ��ͷ�ˣ�appendȻ��ȥ�ո�
			if (i == 0) {
				sb.append(strings[i].trim());
			} else {
				// ���ж���Ŀո�ȥ�����ټ���ȥ
				sb.append(strings[i].trim()).append(" ");
			}
		}
		// ���String
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
		// ��Ҫ���ַ���������Ҳ����t�ַ������ַ��ĸ���
		int count = t.length();
		// ����ַ�������ʼλ��
		int begin = -1;
		// �����������С�Ӵ��ĳ���
		int size = s.length() + 1;
		char[] chs = s.toCharArray();
		while (right < s.length()) {
			// �ҵ�һ��t���ַ�
			// ѭ��������Ҫ�ĸ�����������Ҫ���ַ��Ļ��ᱻ����-1������С
			if (hash[chs[right] - 'A']-- > 0) {
				// ������Ҫ���ַ�����
				count--;
			}
			// ֱ��������ӣ�ʡ���ں����������ʱ��Ҫ��һ
			right++;
			// �Ѿ�����t�ַ������ַ�
			if (count == 0) {
				// ��ʼ���󴰿ڿ�ʼ������С����
				// ֻҪ�󴰿��ҵ�һ��t���ַ�����С���ھ�ֹͣ
				while (hash[chs[left] - 'A']++ < 0) {
					// �ָ�֮ǰ��ȥ���ַ�����
					left++;
				}
				// ��¼���Ⱥ���ʼλ��
				if (right - left < size) {
					size = right - left;
					begin = left;
				}
				// ��Ϊ�ƶ����Ѿ�����Ҫ�ַ����󴰿ڣ�������Ҫ���ַ���һ
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
			// ���i�ƶ����˻���ֱ�Ӽ�һ����ͬ���ַ��Ϳ����ˣ���Ϊ��ͬ�ַ����ܳ�������
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
			// �����������ϵľ�ɾ��һ�����м�����2��
			if (i - start + 1 >= 3) {
				// ɾ���ɹ��ͼӴ���
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
			// ȡ[l, r]֮����������ĸ���
			res[i] = s[r + 1] - s[l];
		}
		return res;
	}
}
