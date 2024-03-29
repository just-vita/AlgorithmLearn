package top.vita.greedy;

import java.util.Arrays;
import java.util.PriorityQueue;

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

    public int maxProfit(int[] prices) {
        int res = 0;
        for (int i = 1; i < prices.length; i++) {
            res += Math.max(prices[i - 1] - prices[i], 0);
        }
        return res;
    }

    public boolean canJump(int[] nums) {
        if (nums.length == 1){
            return true;
        }
        int cover = 0;
        for (int i = 0; i <= cover; i++){
            // 计算能到达的最大范围，看看能不能到达终点
            cover = Math.max(i + nums[i], cover);
            if (cover >= nums.length - 1){
                return true;
            }
        }
        return false;
    }

    public int jump(int[] nums) {
        if (nums.length == 1){
            return 0;
        }
        int res = 0;
        int curDistance = 0;
        int maxDistance = 0;
        for (int i = 0; i < nums.length; i++){
            maxDistance = Math.max(maxDistance, nums[i] + i);
            // 能够到达的距离能够到终点，那么再走一步就能到
            if (maxDistance >= nums.length - 1){
                res++;
                break;
            }
            // 已经到达能够到达的最远距离，但还没到终点，只能再走一步看看
            if (i == curDistance){
                res++;
                curDistance = maxDistance;
            }
        }
        return res;
    }

    public int largestSumAfterKNegations(int[] nums, int k) {
        Arrays.sort(nums);
        // 局部最优 先将绝对值最大的负数取反，尽量得到最大和
        for (int i = 0; i < nums.length && k > 0; i++){
            if (nums[i] < 0){
                nums[i] = -nums[i];
                k--;
            }
        }
        // 将全部负数都取反后k还有剩余
        if (k > 0){
            // 将经历取反后的数组重新排序
            Arrays.sort(nums);
            // 避免多余操作
            if (k % 2 == 1){
                // 只将最小数取反，得到最大和
                nums[0] = -nums[0];
            }
        }
        int sum = 0;
        for (int i = 0; i < nums.length; i++){
            sum += nums[i];
        }
        return sum;
    }

    public int scheduleCourse(int[][] courses) {
        // 按照学习时间升序排序，学习时间长的排在后面
        Arrays.sort(courses, (a, b) -> a[1] - b[1]);
        // 简历大根堆
        PriorityQueue<Integer> queue = new PriorityQueue<>((a, b) -> b - a);
        int sum = 0;
        for (int[] arr : courses) {
            sum += arr[0];
            queue.add(arr[0]);
            if (sum > arr[1]) {
                sum -= queue.poll();
            }
        }
        return queue.size();
    }


}
