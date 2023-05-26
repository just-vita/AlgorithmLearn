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
        // ѹ�����xǰ����С��
        stack.push(min);
        // ���¼���x�����С��
        if (x < min) {
            min = x;
        }
        // ѹ��x
        stack.push(x);
    }
    
    public void pop() {
        // ��x����
        stack.pop();
        // ����Сֵ�ָ���ѹ��x֮ǰ
        min = stack.peek();
        // ����ǰ��Сֵ����
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