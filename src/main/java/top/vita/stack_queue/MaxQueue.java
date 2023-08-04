package top.vita.stack_queue;

import java.util.Deque;
import java.util.LinkedList;

class MaxQueue {
    // �洢���ֵ
    Deque<Integer> max;
    // �����洢����
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
        // �ҵ��µ����ֵʱ�Ӷ�β��ʼ������valueС��ֵ����֤�����е�˳��
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
        // ����������ǵ�ǰ����ֵ���ͽ��洢���ֵ�Ķ����е�ֵҲ����
        if (!max.isEmpty() && max.peekFirst() == temp) {
            max.removeFirst();
        }
        queue.removeFirst();
        return temp;
    }
}