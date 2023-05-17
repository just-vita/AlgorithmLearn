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











}
