package top.vita.hash;

import java.util.HashMap;

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
}
