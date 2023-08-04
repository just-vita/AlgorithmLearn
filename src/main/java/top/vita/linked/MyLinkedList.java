package top.vita.linked;

public class MyLinkedList {
	LinkedNode head;
	int size;
	
    public MyLinkedList() {
    	 head = new LinkedNode(-1);
    	 size = 0;
    }
    
    public int get(int index) {
    	if (index > size || index < 0) {
    		return -1;
    	}
    	LinkedNode cur = head.next;
    	while (index-- > 1) {
    		cur = cur.next;
    	}
    	return cur.val;
    }
    
    public void addAtHead(int val) {
    }
    
    public void addAtTail(int val) {
    }
    
    public void addAtIndex(int index, int val) {
    	LinkedNode node = new LinkedNode(val);
    	LinkedNode cur = head.next;
    	while (index-- > 0) {
    		cur = cur.next;
    	}
    	cur.next = node;
    	size++;
    }
    
    public void deleteAtIndex(int index) {
    	LinkedNode cur = head.next;
    	while (index-- > 1) {
    		cur = cur.next;
    	}
    	cur.next = cur.next.next;
    	size--;
    }
}

class LinkedNode {
    int val;
    LinkedNode next;
	public LinkedNode(int val) {
		this.val = val;
	}
};