package top.vita.linked;

import java.util.HashMap;

public class LRUCache {

    static class Node {
        private Node pre;
        private Node next;
        private int key;
        private int value;

        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    private int capacity;
    private int size;

    private Node head;
    private Node tail;

    private HashMap<Integer, Node> map = new HashMap<>();

    public LRUCache(int capacity) {
        this.capacity = capacity;
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head.next = tail;
        tail.pre = head;
    }
    
    public int get(int key) {
        if (!map.containsKey(key)) {
            return -1;
        }
        Node node = map.get(key);
        // ���������Ƴ��ڵ�
        unlink(node);
        // ���¼�������ͷ��
        addToHead(node);
        return node.value;
    }

    public void put(int key, int value) {
        if (map.containsKey(key)) {
            // ���Ƴ������º����ƶ�������ͷ��
            Node node = map.get(key);
            unlink(node);
            node.value = value;
            addToHead(node);
        } else {
            Node cur = new Node(key, value);
            if (capacity == size) {
                // �������㣬���½ڵ��ƶ���ͷ����ͬʱ�Ƴ�β���ڵ�
                addToHead(cur);
                removeTail();
            } else {
                // �������㣬ֱ���ƶ�������ͷ��
                addToHead(cur);
            }
        }
    }

    private void addToHead(Node node) {
        // ����ͷ������һ���ڵ�
        node.next = head.next;
        node.pre = head;
        head.next.pre = node;
        head.next = node;
        size++;
        map.put(node.key, node);
    }

    private void unlink(Node node) {
        // ����������
        node.pre.next = node.next;
        node.next.pre = node.pre;
        size--;
    }

    private void removeTail() {
        // ɾ��β���ڵ㣬ע������ȱ������ʱ��������Ȼ��map remove��ʱ�����һ�������ֵ
        Node node = tail.pre;
        unlink(node);
        map.remove(node.key);
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */