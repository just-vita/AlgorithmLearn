package top.vita.stack_queue;

import java.util.LinkedList;
import java.util.Queue;

class MyStack {
    // 主队列，用来存储与栈相同的数据
    Queue<Integer> queue;
    // 备用队列
    Queue<Integer> bakQueue;

    public MyStack() {
        queue = new LinkedList<>();
        bakQueue = new LinkedList<>();
    }
    
    public void push(int x) {
        // 先存入备用队列
        bakQueue.offer(x);
        // 把之前的数据盖到备用队列上
        while (!queue.isEmpty()){
            bakQueue.offer(queue.poll());
        }
        // 将两个队列转换，使备用队列永远是空的（联想到垃圾回收的标记复制算法了
        Queue<Integer> queueTemp = queue;
        queue = bakQueue;
        bakQueue = queueTemp;
    }
    
    public int pop() {
        return queue.poll();
    }
    
    public int top() {
        return queue.peek();
    }
    
    public boolean empty() {
        return queue.isEmpty();
    }
}
