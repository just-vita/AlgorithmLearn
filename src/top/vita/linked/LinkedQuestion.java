package top.vita.linked;

import java.util.HashSet;
import java.util.Stack;

public class LinkedQuestion {

	public static void main(String[] args) {

	}

	/*
	 * 206. ��ת���� ���㵥�����ͷ�ڵ� head �����㷴ת���������ط�ת�������
	 */
	public static ListNode reverseList(ListNode head) {
		ListNode pre = null;
		ListNode cur = head;
		while (cur != null) {
			ListNode next = cur.next;
			cur.next = pre;
			pre = cur;

			cur = next;
		}
		return pre;
	}

	/*
	 * 19. ɾ������ĵ����� N ����� ����һ������ɾ������ĵ����� n ����㣬���ҷ��������ͷ��㡣
	 */
	public ListNode removeNthFromEnd(ListNode head, int n) {
		ListNode header = new ListNode(-1);
		header.next = head;
		ListNode fast = header;
		ListNode slow = header;
		// �Ƚ���ָ���ƶ��� n ��λ��
		while (n-- > 0 && fast != null) {
			fast = fast.next;
		}
		// ����ָ���ƶ��� n + 1 ��λ��
		fast = fast.next;
		// ͬʱ�ƶ�����ָ�룬��ָ��Ϊnullʱֹͣ�ƶ�
		while (fast != null) {
			fast = fast.next;
			slow = slow.next;
		}
		slow.next = slow.next.next;
		return header.next;
	}

	/*
	 * ������ 02.07. �����ཻ ���������������ͷ�ڵ� headA �� headB �� �����ҳ������������������ཻ����ʼ�ڵ㡣
	 * �����������û�н��㣬����null ��
	 */
	public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
		HashSet set = new HashSet();
		ListNode cur = headA;
		while (cur != null) {
			set.add(cur);
			cur = cur.next;
		}
		cur = headB;
		while (cur != null) {
			if (set.contains(cur)) {
				return cur;
			}
			cur = cur.next;
		}
		return null;
	}

	/*
	 * 142. �������� II ����һ�������ͷ�ڵ� head ����������ʼ�뻷�ĵ�һ���ڵ㡣 ��������޻����򷵻� null��
	 * �����������ĳ���ڵ㣬����ͨ����
	 */
	public ListNode detectCycle(ListNode head) {
		HashSet set = new HashSet();
		ListNode cur = head;

		while (cur != null) {
			if (set.contains(cur)) {
				return cur;
			}
			set.add(cur);
			cur = cur.next;
		}

		return null;
	}

	/*
	 * 242. ��Ч����ĸ��λ�� ���������ַ��� s �� t ����дһ���������ж� t �Ƿ��� s ����ĸ��λ�ʡ�
	 */
	public boolean isAnagram(String s, String t) {
		// ����26����ĸ��������λ�ã�Ĭ�ϳ�ʼֵ����0
		int[] record = new int[26];
		for (char ch : s.toCharArray()) {
			record[ch - 'a']++;
		}
		for (char ch : t.toCharArray()) {
			record[ch - 'a']--;
		}
		for (int i : record) {
			if (i != 0) {
				return false;
			}
		}

		return true;
	}

	/*
	 * 383. ����� ���������ַ�����ransomNote �� magazine �� �ж� ransomNote �ܲ����� magazine ������ַ����ɡ�
	 */
	public boolean canConstruct(String ransomNote, String magazine) {
		int[] record = new int[26];
		for (char ch : ransomNote.toCharArray()) {
			record[ch - 'a']++;
		}
		for (char ch : magazine.toCharArray()) {
			record[ch - 'a']--;
		}
		for (int i : record) {
			if (i > 0) {
				return false;
			}
		}
		return true;
	}

	/*
	 * 2. ������� 
	 * �������� �ǿ� ��������ʾ�����Ǹ���������
	 * ����ÿλ���ֶ��ǰ��� ���� �ķ�ʽ�洢�ģ�����ÿ���ڵ�ֻ�ܴ洢 һλ ���֡�
	 */
	public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
		ListNode root = new ListNode(0);
		ListNode cur = root;
		int carry = 0;
		// carry != 0 ʱҲҪ����һ��ѭ�����ѽ�λ��������
		while (l1 != null || l2 != null || carry != 0) {
			int x = l1 == null ? 0 : l1.val;
			int y = l2 == null ? 0 : l2.val;
			int sum = x + y + carry;
			carry = sum / 10;
			ListNode sumNode = new ListNode(sum % 10);
			cur.next = sumNode;
			cur = sumNode;

			if (l1 != null)
				l1 = l1.next;
			if (l2 != null)
				l2 = l2.next;
		}
		return root.next;
	}
	
	/*
	 * 160. �ཻ���� 
	 * ���������������ͷ�ڵ� headA �� headB ��
	 * �����ҳ������������������ཻ����ʼ�ڵ㡣
	 * ����������������ཻ�ڵ㣬����null ��
	 */
    public ListNode getIntersectionNode2(ListNode headA, ListNode headB) {
        if(headA == null || headB == null) return null;
        ListNode nodeA = headA,nodeB = headB;
        while (nodeA != nodeB) {
        	nodeA = nodeA == null ? headB : nodeA.next;
        	nodeB = nodeB == null ? headA : nodeB.next;
        }
        return nodeA;
    }
    
    /*
	 * 82. ɾ�����������е��ظ�Ԫ�� II 
	 * ����һ��������������ͷ head �� 
	 * ɾ��ԭʼ�����������ظ����ֵĽڵ㣬
	 * ֻ���²�ͬ������ ������ �����������
	 */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode next = head.next;
        if (head.val == next.val) {
        	while (next != null && head.val == next.val) {
        		next = next.next;
        	}
        	head = deleteDuplicates(next);
        }else {
        	head.next = deleteDuplicates(next);
        }
        
    	return head;
    }
    
	/*
	 * 24. �������������еĽڵ�
	 * ����һ���������������������ڵĽڵ㣬
	 * �����ؽ����������ͷ�ڵ㡣
	 * ������ڲ��޸Ľڵ��ڲ���ֵ���������ɱ��⣨����ֻ�ܽ��нڵ㽻������
	 */
    public ListNode swapPairs(ListNode head) {
    	if (head == null || head.next == null) return head;
    	ListNode next = head.next;
    	// �����м���һ���Ѿ��źõ�����
    	head.next = swapPairs(next.next);
    	// ��head��next����
    	next.next = head;
    	return next;
    }
    
    /*
	 * 25. K ��һ�鷭ת���� 
	 * ���������ͷ�ڵ� head ��ÿ k ���ڵ�һ����з�ת��
	 * ���㷵���޸ĺ������
	 */
    public ListNode reverseKGroup(ListNode head, int k) {
    	ListNode post = head;
    	for (int i = 0; i < k; i++) {
    		// ��������k����ֱ�ӷ���
			if (post == null) return head;
			post = post.next;
		}
    	
    	ListNode pre = null, cur = head;
    	while (cur != post) {
    		// �ͺ�һ���ڵ㽻��
    		ListNode temp = cur.next;
    		cur.next = pre;
    		pre = cur;
    		cur = temp;
    	}
    	head.next = reverseKGroup(cur, k);
    	return pre;
    }
    
    /*
	 * 143. �������� 
	 * ����һ�������� L ��ͷ�ڵ� head �������� L ��ʾΪ��
	 * L0 �� L1 �� �� �� Ln - 1 �� Ln 
	 * �뽫���������к��Ϊ��
	 * L0 �� Ln �� L1 �� Ln - 1 �� L2 �� Ln - 2 �� ��
	 */
    public void reorderList(ListNode head) {
    	if (head == null) return;
    	ListNode l1 = head;
    	ListNode l2 = afterNode(head);
    	
    	// ����
    	ListNode l1_tmp, l2_tmp;
    	while(l1 != null && l2 != null) {
    		l1_tmp = l1.next;
    		l2_tmp = l2.next;
    		
    		l1.next = l2;
    		l1 = l1_tmp;
    		
    		l2.next = l1;
    		l2 = l2_tmp;
    	}
    }

	private ListNode afterNode(ListNode head) {
		// ����ָ��
		ListNode slow = head;
		ListNode fast = head;
		// fast������ʱslowָ���м�ڵ�
		while (fast.next != null && fast.next.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}
		
		ListNode pre = null, cur = slow.next, next;
		slow.next = null;
		// ��ת
		while (cur != null) {
			next = cur.next;
			cur.next = pre;
			pre = cur;
			cur = next;
		}
		return pre;
	}
	
	/*
	 * 21. �ϲ������������� 
	 * ��������������ϲ�Ϊһ���µ� ���� �������ء�
	 * ��������ͨ��ƴ�Ӹ�����������������нڵ���ɵġ�
	 */
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
    	ListNode cur1 = list1;
    	ListNode cur2 = list2;
    	ListNode header = new ListNode(0);
    	ListNode res = header;
    	while (cur1 != null && cur2 != null) {
    		if (cur1.val < cur2.val) {
    			res.next = cur1;
    			cur1 = cur1.next;
    			res = res.next;
    		}else {
    			res.next = cur2;
    			cur2 = cur2.next;
    			res = res.next;
    		}
    	}
    	while (cur1 != null) {
    		res.next = cur1;
    		cur1 = cur1.next;
    		res = res.next;
    	}
    	
    	while (cur2 != null) {
    		res.next = cur2;
    		cur2 = cur2.next;
    		res = res.next;
    	}
    	
    	return header.next;
    }
    
    /*
	 * 141. �������� 
	 * ����һ�������ͷ�ڵ� head ���ж��������Ƿ��л���
	 */
    public boolean hasCycle(ListNode head) {
        if (head == null) return false;
        ListNode slow = head;
        ListNode fast = head.next;
        while (slow != fast) {
        	if (fast == null || fast.next == null) return false;
        	slow = slow.next;
        	fast = fast.next.next;
        }
        return true;
    }
    
    /*
	 * 876. ������м��� 
	 * ����һ��ͷ���Ϊ head �ķǿյ���������������м��㡣
	 * ����������м��㣬�򷵻صڶ����м��㡣
	 */
    public ListNode middleNode(ListNode head) {
    	ListNode cur = head;
    	int count = 0;
    	while (cur != null) {
    		count++;
    		cur = cur.next;
    	}
    	count = count / 2;
    	cur = head;
    	while (count-- > 0) {
    		cur = cur.next;
    	}
    	return cur;
    }
    
    /*
	 * 83. ɾ�����������е��ظ�Ԫ�� 
	 * ����һ��������������ͷ head �� ɾ�������ظ���Ԫ�أ�
	 * ʹÿ��Ԫ��ֻ����һ�� ������ ����������� ��
	 */
    public ListNode deleteDuplicates1(ListNode head) {
        if(head == null || head.next == null){
            return head;
        }
        head.next = deleteDuplicates(head.next);
        if(head.val == head.next.val) head = head.next;
        return head;
    }
    
    public ListNode getIntersectNode(ListNode head1, ListNode head2) {
    	if (head1 == null || head2 == null) return null;
    	ListNode loop1 = getLoopNode(head1);
    	ListNode loop2 = getLoopNode(head2);
    	if (loop1 == null && loop2 == null) {
    		return noLoop(head1, head2);
    	}
    	if (loop1 != null && loop2 != null) {
    		return bothLoop(head1, head2, loop1, loop2);
    	}
    	return null;
    }
    
	/*
	 * ��ȡ��һ���뻷�ڵ�
	 */
    public ListNode getLoopNode(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) return null;
        ListNode slow = head.next;
        ListNode fast = head.next.next;
        while (slow != fast){
            if (fast.next == null || fast.next.next == null){
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        // ������ʹ��ָ��ص�ͷ�ڵ㣬���ջᵽ���һ���뻷�ڵ�
        fast = head;
        while (slow != fast){
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }
    
	/*
	 * ���������л����뻷�ڵ㲻һ������� 
	 * �ȱ���һ�εó����һ���ڵ�������� 
	 * ���ڵ���ͬ��Ϊ�ཻ �ó��������߲�ֵ����Ȼ�󳤶�����һ����
	 * ��һ���ڵ���ǵ�һ���ཻ�Ľڵ�
	 */
    public ListNode noLoop(ListNode head1,ListNode head2) {
		ListNode cur1 = head1;
		ListNode cur2 = head2;
		int n = 0;
		while (cur1 != null) {
			n++;
			cur1 = cur1.next;
		}
		while (cur2 != null) {
			n--;
			cur2 = cur2.next;
		}
		cur1 = n > 0 ? head1 : head2;
		cur2 = cur1 == head1 ? head2 : head1;
		n = Math.abs(n);
		// ���������߲�ֵ��
		while (n != 0) {
			n--;
			cur1 = cur1.next;
		}
		// ͬʱ�ƶ�
		while (cur1 != cur2) {
			cur1 = cur1.next;
			cur2 = cur2.next;
		}
		return cur1;
    }
    
	public ListNode bothLoop(ListNode head1, ListNode head2,
							 ListNode loop1, ListNode loop2) {
		ListNode cur1 = null;
		ListNode cur2 = null;
		if (loop1 == loop2) {
			/*
			 * ���������л����뻷�ڵ�һ�������
			 * �ȱ���һ�εó����һ���ڵ�������� 
			 * ���ڵ���ͬ��Ϊ�ཻ 
			 * �ó��������߲�ֵ����Ȼ�󳤶�����һ���� 
			 * ��һ���ڵ���ǵ�һ���ཻ�Ľڵ�
			 */
			cur1 = head1;
			cur2 = head2;
			int n = 0;
			while (cur1 != loop1) {
				n++;
				cur1 = cur1.next;
			}
			while (cur2 != loop2) {
				n--;
				cur2 = cur2.next;
			}
			cur1 = n > 0 ? head1 : head2;
			cur2 = cur1 == head1 ? head2 : head1;
			n = Math.abs(n);
			// ���������߲�ֵ��
			while (n != 0) {
				n--;
				cur1 = cur1.next;
			}
			// ͬʱ�ƶ�
			while (cur1 != cur2) {
				cur1 = cur1.next;
				cur2 = cur2.next;
			}
			return cur1;
		}else {
			/*
			 *  ���������л����뻷�ڵ㲻һ�������
			 *  �ó�����һֱ�ߣ�����������뻷�ڵ�(loop2)�������ͬһ��
			 *  ���������뻷�ڵ㼴��
			 */
			cur1 = loop1;
			while (cur1 != null) {
				if (cur1 == loop2) {
					return loop1;
				}
				cur1 = cur1.next;
			}
			return null;
		}
	}
	
	
	/*
	 * 206. ��ת����
	 */
    public ListNode reverseList1(ListNode head) {
    	return reverse(null,head);
    }

	private ListNode reverse(ListNode pre, ListNode cur) {
		if (cur == null) return pre;
		ListNode next = cur.next;
		cur.next = pre;
		return reverse(cur, next);
	}
	
	/*
	 * 1249. �Ƴ���Ч������
	 */
    public String minRemoveToMakeValid(String s) {
    	Stack<Integer> stack = new Stack<>();
    	char[] chs = s.toCharArray();
    	boolean[] isBrackets = new boolean[s.length()];
    	for (int i = 0; i < chs.length; i++) {
			if (chs[i] == '(') {
				stack.push(i);
				isBrackets[i] = true;
			}else if(chs[i] == ')') {
				if (!stack.isEmpty()) {
					isBrackets[stack.pop()] = false;
				}else {
					isBrackets[i] = true;
				}
			}
		}
    	StringBuilder sb = new StringBuilder();
    	for (int i = 0; i < chs.length; i++) {
			if (!isBrackets[i]) {
				sb.append(chs[i]);
			}
		}
    	return sb.toString();
    }
	
	/*
	 * 24. �������������еĽڵ�
	 */
	public ListNode swapPairs1(ListNode head) {
		ListNode next = head.next;
		head.next = swapPairs1(next.next);
		next.next = head;
		return head;
	}
	
	/*
	 * 19. ɾ������ĵ����� N �����
	 */
	int cur = 0;
	public ListNode removeNthFromEnd1(ListNode head, int n) {
		if (head == null){
			return null;
		}
		// �ݹ鵽����null�ŻῪʼ��ջ�����Դ�ʱ��ʼ���ǴӺ���ǰ��
		head.next = removeNthFromEnd1(head.next, n);
		cur++;
		// �����������ʱ����next
		if (cur == n){
			return head.next;
		}
		return head;
	}

	/**
	 * 203. �Ƴ�����Ԫ��
	 */
	public ListNode removeElements(ListNode head, int val) {
		ListNode header = new ListNode(-1);
		header.next = head;
		ListNode cur = header;
		while (cur.next != null){
			if (cur.next.val == val){
				cur.next = cur.next.next;
			} else {
				cur = cur.next;
			}
		}
		return header.next;
	}

	/**
	 * 206. ��ת����
	 */
	public ListNode reverseList2(ListNode head) {
		return reverse1(null, head);
	}

	public ListNode reverse1(ListNode pre, ListNode cur){
		if (cur == null){
			return pre;
		}
		ListNode next = cur.next;
		cur.next = pre;
		return reverse(cur, next);
	}

	/**
	 * 24. �������������еĽڵ�
	 */
	public ListNode swapPairs2(ListNode head) {
		if (head == null || head.next == null){
			return head;
		}
		ListNode next = head.next;
		// �����м��Ѿ���һ����Ҫ���ź��������
		// �Ƚ���ǰ�ڵ��next�ĳɽ���е�next (1 -> 3)
		head.next = swapPairs2(next.next);
		// ��ԭ����nextָ��ǰ�ڵ� (2 -> 1)
		next.next = head;
		return next;
	}

	/**
	 * 19. ɾ������ĵ����� N �����
	 */
	public ListNode removeNthFromEnd2(ListNode head, int n) {
		ListNode header = new ListNode(-1);
		header.next = head;
		ListNode fast = header;
		ListNode slow = header;
		while (n-- > 0 && fast != null){
			fast = fast.next;
		}
		fast = fast.next;
		while (fast != null){
			fast = fast.next;
			slow = slow.next;
		}
		slow.next = slow.next.next;
		return header.next;
	}

    /**
     * ������ 02.07. �����ཻ
     */
    public ListNode getIntersectionNode1(ListNode headA, ListNode headB) {
        ListNode cur = headA;
        HashSet set = new HashSet();
        while (cur != null){
            set.add(cur);
            cur = cur.next;
        }
        cur = headB;
        while (cur != null){
            if (set.contains(cur)){
                return cur;
            }
            cur = cur.next;
        }
        return null;
    }

	/**
	 * 142. �������� II
	 */
	public ListNode detectCycle2(ListNode head) {
		if (head == null || head.next == null || head.next.next == null) {
			return null;
		}
		ListNode slow = head.next;
		ListNode fast = head.next.next;
		while (slow != fast){
			if (fast.next == null || fast.next.next == null) {
				return null;
			}
			slow = slow.next;
			fast = fast.next.next;
		}
		fast = head;
		while (slow != fast){
			slow = slow.next;
			fast = fast.next;
		}
		return slow;
	}

	public ListNode rotateRight(ListNode head, int k) {
		if (head == null || head.next == null || k == 0){
			return head;
		}
		ListNode cur = head;
		// ��ȡ������
		int len = 1;
		while (cur.next != null){
			len++;
			cur = cur.next;
		}
		k = k % len;
		// ����Ҫ��ת
		if (k == 0){
			return head;
		}
		// ͷβ����
		cur.next = head;
		int i = 0;
		// �ҵ�������len-k���ڵ㣬Ҳ����������ת��Ľ�β�ڵ�
		while (i++ < len - k){
			cur = cur.next;
		}
		// �õ���ת���ͷ�ڵ�
		ListNode newHead = cur.next;
		// ��������ת��Ľ�β�ڵ��next�ÿ�
		cur.next = null;
		return newHead;
	}

	public int[] reversePrint(ListNode head) {
		int len = 0;
		ListNode cur = head;
		while (cur != null){
			len++;
			cur = cur.next;
		}
		// ����һ�����飬����洢����Ԫ��
		int[] res = new int[len];
		int right = len - 1;
		cur = head;
		while (cur != null){
			res[right--] = cur.val;
			cur = cur.next;
		}
		return res;
	}

	public Node copyRandomList(Node head) {
		if (head == null){
			return head;
		}
		// �������Ϊ1,1,2,2,3,3...
		Node cur = head;
		while (cur != null){
			Node copyNode = new Node(cur.val);
			copyNode.next = cur.next;
			cur.next = copyNode;
			cur = cur.next.next;
		}
		cur = head;
		while (cur != null){
			if (cur.random != null){
				// �����Ƴ��Ľڵ��random
				cur.next.random = cur.random.next;
			}
			cur = cur.next.next;
		}
		// �õ����Ƴ��Ľڵ��ͷ�ڵ�
		Node copyHead = head.next;
		// ��Ϊ1,1,2,2,3,3...�������ֳ�1,2,3... 1,2,3...
		cur = head;
		Node copyNode = head.next;
		while (cur != null){
			cur.next = cur.next.next;
			cur = cur.next;
			if (copyNode.next != null){
				copyNode.next = copyNode.next.next;
				copyNode = copyNode.next;
			}
		}
		return copyHead;
	}
}

