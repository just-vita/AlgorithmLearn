package top.vita.greedy;

import java.util.Arrays;

/**
 * @Author vita
 * @Date 2023/5/17 22:17
 */
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











}
