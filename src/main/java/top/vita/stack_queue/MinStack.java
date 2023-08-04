package top.vita.stack_queue;

import java.util.Stack;

class MinStack {

    int min = Integer.MAX_VALUE;
    Stack<Integer> stack;
    
    /** initialize your data structure here. */
    public MinStack() {
        stack = new Stack<>();
    }
    
    public void push(int x) {
        // 压入加入x前的最小数
        stack.push(min);
        // 更新加入x后的最小数
        if (x < min) {
            min = x;
        }
        // 压入x
        stack.push(x);
    }
    
    public void pop() {
        // 将x弹出
        stack.pop();
        // 将最小值恢复到压入x之前
        min = stack.peek();
        // 将当前最小值弹出
        stack.pop();
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int min() {
        return min;
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.min();
 */