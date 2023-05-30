package top.vita.stack_queue;

import java.util.Deque;
import java.util.LinkedList;

class MaxQueue {
    // 存储最大值
    Deque<Integer> max;
    // 正常存储数据
    Deque<Integer> queue;

    public MaxQueue() {
        max = new LinkedList<>();
        queue = new LinkedList<>();
    }
    
    public int max_value() {
        if (queue.isEmpty()) {
            return -1;
        }
        return max.peekFirst();
    }
    
    public void push_back(int value) {
        queue.addLast(value);
        // 找到新的最大值时从队尾开始弹出比value小的值，保证队列中的顺序
        while (!max.isEmpty() && max.peekLast() < value) {
            max.removeLast();
        }
        max.addLast(value);
    }
    
    public int pop_front() {
        if (queue.isEmpty()) {
            return -1;
        }
        int temp = queue.peekFirst();
        // 如果弹出的是当前最大的值，就将存储最大值的队列中的值也弹出
        if (!max.isEmpty() && max.peekFirst() == temp) {
            max.removeFirst();
        }
        queue.removeFirst();
        return temp;
    }
}