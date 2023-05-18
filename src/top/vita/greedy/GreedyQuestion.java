package top.vita.greedy;

import java.util.Arrays;

/**
 * @Author vita
 * @Date 2023/5/17 22:17
 */
@SuppressWarnings("all")
public class GreedyQuestion {
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int index = s.length - 1;
        int res = 0;
        // 大饼干优先满足大胃口
        for (int i = g.length - 1; i >= 0; i--) {
            // 用index变量代替循环
            if (index >= 0 && s[index] >= g[i]){
                res++;
                index--;
            }
        }
        return res;
    }

    public int wiggleMaxLength(int[] nums) {
        int curDiff = 0;
        int preDiff = 0;
        // 至少有一个峰值
        int res = 1;
        for (int i = 0; i < nums.length - 1; i++){
            // 计算当前差值
            curDiff = nums[i + 1] - nums[i];
            // 如果差值和上一个的差值呈一正一负，则代表找到一个峰值
            if ((curDiff > 0 && preDiff <= 0) || (curDiff < 0 && preDiff >= 0)){
                res++;
                preDiff = curDiff;
            }
        }
        return res;
    }

    public int maxSubArray(int[] nums) {
        int maxSum = Integer.MIN_VALUE;
        int curSum = 0;
        for (int i = 0; i < nums.length; i++) {
            curSum += nums[i];
            maxSum = Math.max(maxSum, curSum);
            if (curSum < 0){
                curSum = 0;
            }
        }
        return maxSum;
    }








}
