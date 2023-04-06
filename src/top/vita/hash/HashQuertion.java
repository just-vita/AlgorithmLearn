package top.vita.hash;

import java.util.HashMap;
import java.util.HashSet;

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

}
