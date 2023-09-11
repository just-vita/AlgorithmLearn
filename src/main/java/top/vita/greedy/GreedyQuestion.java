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
        // ��������������θ��
        for (int i = g.length - 1; i >= 0; i--) {
            // ��index��������ѭ��
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
        // ������һ����ֵ
        int res = 1;
        for (int i = 0; i < nums.length - 1; i++){
            // ���㵱ǰ��ֵ
            curDiff = nums[i + 1] - nums[i];
            // �����ֵ����һ���Ĳ�ֵ��һ��һ����������ҵ�һ����ֵ
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
            // �����ܵ�������Χ�������ܲ��ܵ����յ�
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
            // �ܹ�����ľ����ܹ����յ㣬��ô����һ�����ܵ�
            if (maxDistance >= nums.length - 1){
                res++;
                break;
            }
            // �Ѿ������ܹ��������Զ���룬����û���յ㣬ֻ������һ������
            if (i == curDistance){
                res++;
                curDistance = maxDistance;
            }
        }
        return res;
    }

    public int largestSumAfterKNegations(int[] nums, int k) {
        Arrays.sort(nums);
        // �ֲ����� �Ƚ�����ֵ���ĸ���ȡ���������õ�����
        for (int i = 0; i < nums.length && k > 0; i++){
            if (nums[i] < 0){
                nums[i] = -nums[i];
                k--;
            }
        }
        // ��ȫ��������ȡ����k����ʣ��
        if (k > 0){
            // ������ȡ�����������������
            Arrays.sort(nums);
            // ����������
            if (k % 2 == 1){
                // ֻ����С��ȡ�����õ�����
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
        // ����ѧϰʱ����������ѧϰʱ�䳤�����ں���
        Arrays.sort(courses, (a, b) -> a[1] - b[1]);
        // ���������
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
