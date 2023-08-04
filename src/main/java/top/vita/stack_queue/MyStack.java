package top.vita.stack_queue;

import java.util.LinkedList;
import java.util.Queue;

class MyStack {
    // �����У������洢��ջ��ͬ������
    Queue<Integer> queue;
    // ���ö���
    Queue<Integer> bakQueue;

    public MyStack() {
        queue = new LinkedList<>();
        bakQueue = new LinkedList<>();
    }
    
    public void push(int x) {
        // �ȴ��뱸�ö���
        bakQueue.offer(x);
        // ��֮ǰ�����ݸǵ����ö�����
        while (!queue.isEmpty()){
            bakQueue.offer(queue.poll());
        }
        // ����������ת����ʹ���ö�����Զ�ǿյģ����뵽�������յı�Ǹ����㷨��
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
