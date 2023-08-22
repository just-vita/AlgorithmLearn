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
        // 从链表中移除节点
        unlink(node);
        // 重新加入链表头部
        addToHead(node);
        return node.value;
    }

    public void put(int key, int value) {
        if (map.containsKey(key)) {
            // 先移除，更新后再移动到链表头部
            Node node = map.get(key);
            unlink(node);
            node.value = value;
            addToHead(node);
        } else {
            Node cur = new Node(key, value);
            if (capacity == size) {
                // 容量不足，把新节点移动到头部的同时移除尾部节点
                addToHead(cur);
                removeTail();
            } else {
                // 容量充足，直接移动到链表头部
                addToHead(cur);
            }
        }
    }

    private void addToHead(Node node) {
        // 链表头部增加一个节点
        node.next = head.next;
        node.pre = head;
        head.next.pre = node;
        head.next = node;
        size++;
        map.put(node.key, node);
    }

    private void unlink(Node node) {
        // 解除链表关联
        node.pre.next = node.next;
        node.next.pre = node.pre;
        size--;
    }

    private void removeTail() {
        // 删除尾部节点，注意必须先保存好临时变量，不然到map remove的时候会是一个变过的值
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