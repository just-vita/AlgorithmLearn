package top.vita.stack_queue;

import java.util.Stack;

class MyQueue {
    // ��ջ
    Stack<Integer> in;
    // ��ջ
    Stack<Integer> out;

    public MyQueue() {
        in = new Stack<>();
        out = new Stack<>();
    }
    
    public void push(int x) {
        // ��ջ����ֱ�Ӽ�����ջ��
        in.push(x);
    }
    
    public int pop() {
        // ����ջΪ��ʱ�������ջ����Ϊ��
        if (out.isEmpty()){
            while (!in.isEmpty()){
                // �ͽ���ջ������Ԫ�ض�������ջ��
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

        // �������е�pop������������push��ȥ
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