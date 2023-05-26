package top.vita.stack_queue;

import java.util.Stack;

class CQueue {
    Stack<Integer> out;
    Stack<Integer> in;
    public CQueue() {
        out = new Stack<>();
        in = new Stack<>();
    }
    
    public void appendTail(int value) {
        in.push(value);
    }
    
    public int deleteHead() {
        if (out.isEmpty()){
            while (!in.isEmpty()){
                out.push(in.pop());
            }
        }
        if (out.isEmpty()){
            return -1;
        }
        return out.pop();
    }
}

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue obj = new CQueue();
 * obj.appendTail(value);
 * int param_2 = obj.deleteHead();
 */