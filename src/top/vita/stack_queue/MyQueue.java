package top.vita.stack_queue;

import java.util.Stack;

class MyQueue {
    // 入栈
    Stack<Integer> in;
    // 出栈
    Stack<Integer> out;

    public MyQueue() {
        in = new Stack<>();
        out = new Stack<>();
    }
    
    public void push(int x) {
        // 入栈操作直接加入入栈中
        in.push(x);
    }
    
    public int pop() {
        // 当出栈为空时，如果入栈还不为空
        if (out.isEmpty()){
            while (!in.isEmpty()){
                // 就将入栈的所有元素都加入入栈中
                out.push(in.pop());
            }
        }
        return out.pop();
    }
    
    public int peek() {
        // if (out.isEmpty()){
        //     while (!in.isEmpty()){
        //         out.push(in.pop());
        //     }
        // }

        // 调用已有的pop操作，用完再push回去
        int tmp = pop();
        out.push(tmp);
        return tmp;
    }
    
    public boolean empty() {
        return out.isEmpty() && in.isEmpty();
    }
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = new MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * boolean param_4 = obj.empty();
 */