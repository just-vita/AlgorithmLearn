package top.vita.hash;

import java.lang.reflect.Array;
import java.util.*;

/**
 * @Author vita
 * @Date 2023/4/6 12:51
 */
public class HashQuertion {
    /**
     * 1. 两数之和
     */
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++){
            int diff = target - nums[i];
            if (map.containsKey(diff)){
                return new int[]{map.get(diff), i};
            } else{
                map.put(nums[i], i);
            }
        }
        return new int[]{};
    }

    /**
     * 202. 快乐数
     */
    public boolean isHappy(int n) {
        HashSet set = new HashSet();
        while (true){
            int sum = getSum(n);
            if (sum == 1){
                return true;
            }
            // 陷入死循环
            if (set.contains(sum)){
                return false;
            } else{
                set.add(sum);
            }
            n = sum;
        }
    }

    private int getSum(int n){
        int sum = 0;
        while (n > 0){
            sum += (n % 10) * (n % 10);
            n /= 10;
        }
        return sum;
    }

    /**
     * 349. 两个数组的交集
     */
    public int[] intersection(int[] nums1, int[] nums2) {
        HashSet<Integer> resSet = new HashSet<>();
        HashSet<Integer> set = new HashSet<>();
        for (int i : nums1){
            set.add(i);
        }
        for (int i : nums2){
            if (set.contains(i)){
                resSet.add(i);
            }
        }
        int[] res = new int[resSet.size()];
        int j = 0;
        for (int i : resSet){
            res[j++] = i;
        }
        return res;
    }

    /**
     * 242. 有效的字母异位词
     */
    public boolean isAnagram(String s, String t) {
        int[] record = new int[26];
        for (int i = 0; i < s.length(); i++){
            record[s.charAt(i) - 'a']++;
        }
        for (int i = 0; i < t.length(); i++){
            record[t.charAt(i) - 'a']--;
        }
        for (int i = 0; i < 26; i++){
            if (record[i] != 0){
                return false;
            }
        }
        return true;
    }

    /**
     * 454. 四数相加 II
     */
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums1.length; i++){
            for (int j = 0; j < nums2.length; j++){
                int sum = nums1[i] + nums2[j];
                if (map.containsKey(sum)){
                    map.put(sum, map.get(sum) + 1);
                } else{
                    map.put(sum, 1);
                }
            }
        }
        int count = 0;
        for (int i = 0; i < nums3.length; i++){
            for (int j = 0; j < nums4.length; j++){
                int sum = nums3[i] + nums4[j];
                if (map.containsKey(0 - sum)){
                    count += map.get(0 - sum);
                }
            }
        }
        return count;
    }

    /**
     * 383. 赎金信
     */
    public boolean canConstruct(String ransomNote, String magazine) {
        int[] record = new int[26];
        // 是由magazine里面的字符组成，所以先记录magazine中字符出现的次数
        for (char ch : magazine.toCharArray()) {
            record[ch - 'a']++;
        }
        for (char ch : ransomNote.toCharArray()) {
            record[ch - 'a']--;
            if (record[ch - 'a'] < 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * 15. 三数之和
     */
    public List<List<Integer>> threeSum(int[] nums) {
        ArrayList<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++){
            if (i > 0 && nums[i] == nums[i - 1]){
                continue;
            }
            int target = -nums[i];
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right){
                int sum = nums[left] + nums[right];
                if (sum == target){
                    res.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    while (left < right && nums[left] == nums[++left]);
                    while (left < right && nums[right] == nums[--right]);
                } else if (sum > target){
                    right--;
                } else{
                    left++;
                }
            }
        }
        return res;
    }

    public List<List<Integer>> threeSum2(int[] nums) {
        ArrayList<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++){
            if (i > 0 && nums[i] == nums[i - 1]){
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right){
                int sum = nums[i] + nums[left] + nums[right];
                if (sum == 0){
                    res.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    while (left < right && nums[left] == nums[left + 1]){
                        left++;
                    };
                    while (left < right && nums[right] == nums[right - 1]){
                        right--;
                    };
                    left++;
                    right--;
                } else if (sum > 0){
                    right--;
                } else{
                    left++;
                }
            }
        }
        return res;
    }

    /**
     * 18. 四数之和
     */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        ArrayList<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 3; i++){
            if (i > 0 && nums[i] == nums[i - 1]){
                continue;
            }
            for (int j = i + 1; j < nums.length - 2; j++){
                if (j > i + 1 && nums[j] == nums[j - 1]){
                    continue;
                }
                int left = j + 1;
                int right = nums.length - 1;
                while (left < right){
                    int sum = nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum == target){
                        res.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        while (left < right && nums[left] == nums[left + 1]){
                            left++;
                        };
                        while (left < right && nums[right] == nums[right - 1]){
                            right--;
                        };
                        left++;
                        right--;
                    } else if (sum > target){
                        right--;
                    } else{
                        left++;
                    }
                }

            }
        }
        return res;
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chs = str.toCharArray();
            Arrays.sort(chs);
            // 使用排序后的字符作为键，把用了这些字符的字符串列表作为值
            String key = new String(chs);
            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values());
    }

    public int lengthOfLongestSubstring(String s) {
        if (s.length() == 0) {
            return 0;
        }
        char[] chs = s.toCharArray();
        int max = 0;
        int start = 0;
        int[] hash = new int[128];
        Arrays.fill(hash, -1);
        for (int i = 0; i < chs.length; i++) {
            // 看看这个字符是否出现过
            // 出现过的话就将上一次出现的位置+1作为子串的起始点
            start = Math.max(start, hash[chs[i]] + 1);
            // 得到长度
            max = Math.max(max, i - start + 1);
            // 记录字符位置
            hash[chs[i]] = i;
        }
        return max;
    }

    public List<Integer> findAnagrams(String s, String p) {
        if (s.length() < p.length()) {
            return new ArrayList<>();
        }
        List<Integer> res = new ArrayList<>();
        int[] sHash = new int[26];
        int[] pHash = new int[26];
        // 记录字母频次
        for (int i = 0; i < p.length(); i++) {
            sHash[s.charAt(i) - 'a']++;
            pHash[p.charAt(i) - 'a']++;
        }
        // 判断 [0, p.length() - 1] 位置是否是字母异位词
        // 如果直接能对应上，代表 [0, p.length() - 1] 位置是结果集之一
        if (Arrays.equals(sHash, pHash)) {
            res.add(0);
        }
        for (int i = 0; i < s.length() - p.length(); i++) {
            // 窗口左边界缩小一格（因为上面已经判断了 [0, p.length() - 1]
            sHash[s.charAt(i) - 'a']--;
            // 窗口右边界扩充一格
            sHash[s.charAt(i + p.length()) - 'a']++;
            // 直接判断数组是否相等
            if (Arrays.equals(sHash, pHash)) {
                res.add(i + 1);
            }
        }
        return res;
    }

    public List<Integer> findAnagrams1(String s, String p) {
        List<Integer> res = new ArrayList<>();
        int[] hash = new int[26];
        // 记录字母频次
        for (int i = 0; i < p.length(); i++) {
            hash[p.charAt(i) - 'a']++;
        }
        int left = 0;
        int right = 0;
        while (right < s.length()) {
            // 如果出现了p字符串中没有的字符，会先进else再进if
            // 会先移动left再移动right，加上之后又会减去
            // 也就是说，哈希表中的数据不会受到无关字符的影响
            if (hash[s.charAt(right) - 'a'] > 0) {
                // 直接减少
                hash[s.charAt(right) - 'a']--;
                // 只有遇到p字符串中的字符时，right指针才会超过left指针，形成窗口
                // 窗口右边界扩充一格
                right++;
                if (right - left == p.length()) {
                    res.add(left);
                }
            } else {
                // 在窗口移动时，将窗口左边的字母重新加入哈希表
                hash[s.charAt(left) - 'a']++;
                // 窗口左边界缩小一格
                left++;
            }
        }
        return res;
    }

































































































































































}
