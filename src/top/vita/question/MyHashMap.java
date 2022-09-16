package top.vita.question;

/*
 * 不使用任何内建的哈希表库设计一个哈希映射（HashMap）。
	实现 MyHashMap 类：
	
	MyHashMap() 用空映射初始化对象
	void put(int key, int value) 向 HashMap 插入一个键值对 (key, value) 。如果 key 已经存在于映射中，则更新其对应的值 value 。
	int get(int key) 返回特定的 key 所映射的 value ；如果映射中不包含 key 的映射，返回 -1 。
	void remove(key) 如果映射中存在 key 的映射，则移除 key 和它所对应的 value 。
 */
public class MyHashMap {
	MyLinkedList linkedList;
    public MyHashMap() {
    	linkedList = new MyLinkedList();
    }
    
    public void put(int key, int value) {
    	linkedList.add(new Node(key,value));
    }
    
    public int get(int key) {
    	return linkedList.findByKey(key);
    }
    
    public void remove(int key) {
    	linkedList.deleteByKey(key);
    }
	
	public static void main(String[] args) {
		MyHashMap map = new MyHashMap();
		map.put(1, 1);
		map.put(2, 2);
		System.out.println(map.get(1));
		System.out.println(map.get(3));
		map.put(2, 1);
		System.out.println(map.get(2));
		map.remove(2);
		System.out.println(map.get(2));
		
	}
}

class MyLinkedList{
	private Node head;
	
	public void add(Node node) {
		if (head == null) {
			head = node;
			return;
		}
		Node cur = head;
		while (true) {
			if (cur.key == node.key) {
				cur.value = node.value;
				return;
			}
			if (cur.next == null) {
				break;
			}
			cur = cur.next;
		}
		if (cur.key == node.key) {
			cur.value = node.value;
		}
		cur.next = node;
	}
	
	public void show() {
		if (head == null) {
			System.out.println(head);
			return;
		}
		Node cur = head;
		while (cur != null) {
			System.out.println(cur);
			cur = cur.next;
		}
	}
	
	public int findByKey(int key) {
		if (head == null) {
			return -1;
		}
		Node cur = head;
		while (cur != null) {
			if (cur.key == key) {
				return cur.value;
			}
			cur = cur.next;
		}
		return -1;
	}
	
	public void deleteByKey(int key) {
		if (head == null) {
			return;
		}
		if (head.key == key) {
			head = head.next;
		}
		Node cur = head;
		while (cur != null) {
			if (cur.next != null && cur.next.key == key) {
				cur.next = cur.next.next;
				return;
			}
			cur = cur.next;
		}
	}
}

class Node{
	int key;
	int value;
	Node next;
	
	public Node(int key, int value) {
		this.key = key;
		this.value = value;
	}
}
