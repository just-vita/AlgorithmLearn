package top.vita.question;

/*
 * ��ʹ���κ��ڽ��Ĺ�ϣ������һ����ϣӳ�䣨HashMap����
	ʵ�� MyHashMap �ࣺ
	
	MyHashMap() �ÿ�ӳ���ʼ������
	void put(int key, int value) �� HashMap ����һ����ֵ�� (key, value) ����� key �Ѿ�������ӳ���У���������Ӧ��ֵ value ��
	int get(int key) �����ض��� key ��ӳ��� value �����ӳ���в����� key ��ӳ�䣬���� -1 ��
	void remove(key) ���ӳ���д��� key ��ӳ�䣬���Ƴ� key ��������Ӧ�� value ��
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
