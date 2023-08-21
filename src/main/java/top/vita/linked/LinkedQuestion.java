package top.vita.linked;

import java.util.*;

@SuppressWarnings("all")
public class LinkedQuestion {

	public static void main(String[] args) {

	}

	public boolean hasCycle1(ListNode head) {
		if (head == null) {
			return false;
		}
		ListNode slow = head;
		ListNode fast = head.next;
		while (slow != fast) {
			if (fast == null || fast.next == null) {
				return false;
			}
			slow = slow.next;
			fast = fast.next.next;
		}
		return true;
	}

	/*
	 * 206. 反转链表 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
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
	 * 19. 删除链表的倒数第 N 个结点 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
	 */
	public ListNode removeNthFromEnd(ListNode head, int n) {
		ListNode header = new ListNode(-1);
		header.next = head;
		ListNode fast = header;
		ListNode slow = header;
		// 先将快指针移动到 n 的位置
		while (n-- > 0 && fast != null) {
			fast = fast.next;
		}
		// 将快指针移动到 n + 1 的位置
		fast = fast.next;
		// 同时移动快慢指针，快指针为null时停止移动
		while (fast != null) {
			fast = fast.next;
			slow = slow.next;
		}
		slow.next = slow.next.next;
		return header.next;
	}

	/*
	 * 面试题 02.07. 链表相交 给你两个单链表的头节点 headA 和 headB ， 请你找出并返回两个单链表相交的起始节点。
	 * 如果两个链表没有交点，返回null 。
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
	 * 142. 环形链表 II 给定一个链表的头节点 head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
	 * 如果链表中有某个节点，可以通过连
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
	 * 242. 有效的字母异位词 给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
	 */
	public boolean isAnagram(String s, String t) {
		// 根据26个字母开辟数组位置，默认初始值都是0
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
	 * 383. 赎金信 给你两个字符串：ransomNote 和 magazine ， 判断 ransomNote 能不能由 magazine 里面的字符构成。
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
	 * 2. 两数相加 
	 * 给你两个 非空 的链表，表示两个非负的整数。
	 * 它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
	 */
	public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
		ListNode root = new ListNode(0);
		ListNode cur = root;
		int carry = 0;
		// carry != 0 时也要进行一次循环，把进位加入链表
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
	 * 160. 相交链表 
	 * 给你两个单链表的头节点 headA 和 headB ，
	 * 请你找出并返回两个单链表相交的起始节点。
	 * 如果两个链表不存在相交节点，返回null 。
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
	 * 82. 删除排序链表中的重复元素 II 
	 * 给定一个已排序的链表的头 head ， 
	 * 删除原始链表中所有重复数字的节点，
	 * 只留下不同的数字 。返回 已排序的链表
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
	 * 24. 两两交换链表中的节点
	 * 给你一个链表，两两交换其中相邻的节点，
	 * 并返回交换后链表的头节点。
	 * 你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
	 */
    public ListNode swapPairs(ListNode head) {
    	if (head == null || head.next == null) return head;
    	ListNode next = head.next;
    	// 看成中间是一堆已经排好的链表
    	head.next = swapPairs(next.next);
    	// 将head和next交换
    	next.next = head;
    	return next;
    }
    
    /*
	 * 25. K 个一组翻转链表 
	 * 给你链表的头节点 head ，每 k 个节点一组进行翻转，
	 * 请你返回修改后的链表。
	 */
    public ListNode reverseKGroup(ListNode head, int k) {
    	ListNode post = head;
    	for (int i = 0; i < k; i++) {
    		// 数量不足k个，直接返回
			if (post == null) return head;
			post = post.next;
		}
    	
    	ListNode pre = null, cur = head;
    	while (cur != post) {
    		// 和后一个节点交换
    		ListNode temp = cur.next;
    		cur.next = pre;
    		pre = cur;
    		cur = temp;
    	}
    	head.next = reverseKGroup(cur, k);
    	return pre;
    }
    
    /*
	 * 143. 重排链表 
	 * 给定一个单链表 L 的头节点 head ，单链表 L 表示为：
	 * L0 → L1 → … → Ln - 1 → Ln 
	 * 请将其重新排列后变为：
	 * L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
	 */
    public void reorderList(ListNode head) {
    	if (head == null) return;
    	ListNode l1 = head;
    	ListNode l2 = afterNode(head);
    	
    	// 交换
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
		// 快慢指针
		ListNode slow = head;
		ListNode fast = head;
		// fast遍历完时slow指向中间节点
		while (fast.next != null && fast.next.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}
		
		ListNode pre = null, cur = slow.next, next;
		slow.next = null;
		// 反转
		while (cur != null) {
			next = cur.next;
			cur.next = pre;
			pre = cur;
			cur = next;
		}
		return pre;
	}
	
	/*
	 * 21. 合并两个有序链表 
	 * 将两个升序链表合并为一个新的 升序 链表并返回。
	 * 新链表是通过拼接给定的两个链表的所有节点组成的。
	 */
    public ListNode mergeTwoLists2(ListNode list1, ListNode list2) {
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
	 * 141. 环形链表 
	 * 给你一个链表的头节点 head ，判断链表中是否有环。
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
	 * 876. 链表的中间结点 
	 * 给定一个头结点为 head 的非空单链表，返回链表的中间结点。
	 * 如果有两个中间结点，则返回第二个中间结点。
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
	 * 83. 删除排序链表中的重复元素 
	 * 给定一个已排序的链表的头 head ， 删除所有重复的元素，
	 * 使每个元素只出现一次 。返回 已排序的链表 。
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
	 * 获取第一个入环节点
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
        // 相遇后使快指针回到头节点，最终会到达第一个入环节点
        fast = head;
        while (slow != fast){
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }
    
	/*
	 * 两个链表都有环且入环节点不一样的情况 
	 * 先遍历一次得出最后一个节点和链表长度 
	 * 最后节点相同则为相交 让长链表先走差值步，然后长短链表一起走
	 * 下一个节点就是第一个相交的节点
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
		// 长链表先走差值步
		while (n != 0) {
			n--;
			cur1 = cur1.next;
		}
		// 同时移动
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
			 * 两个链表都有环且入环节点一样的情况
			 * 先遍历一次得出最后一个节点和链表长度 
			 * 最后节点相同则为相交 
			 * 让长链表先走差值步，然后长短链表一起走 
			 * 下一个节点就是第一个相交的节点
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
			// 长链表先走差值步
			while (n != 0) {
				n--;
				cur1 = cur1.next;
			}
			// 同时移动
			while (cur1 != cur2) {
				cur1 = cur1.next;
				cur2 = cur2.next;
			}
			return cur1;
		}else {
			/*
			 *  两个链表都有环但入环节点不一样的情况
			 *  让长链表一直走，如果遇到了入环节点(loop2)则代表在同一环
			 *  返回任意入环节点即可
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
	 * 206. 反转链表
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
	 * 1249. 移除无效的括号
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
	 * 24. 两两交换链表中的节点
	 */
	public ListNode swapPairs1(ListNode head) {
		ListNode next = head.next;
		head.next = swapPairs1(next.next);
		next.next = head;
		return head;
	}
	
	/*
	 * 19. 删除链表的倒数第 N 个结点
	 */
	int cur = 0;
	public ListNode removeNthFromEnd1(ListNode head, int n) {
		if (head == null){
			return null;
		}
		// 递归到发现null才会开始出栈，所以此时开始就是从后往前了
		head.next = removeNthFromEnd1(head.next, n);
		cur++;
		// 当到达这个数时返回next
		if (cur == n){
			return head.next;
		}
		return head;
	}

	/**
	 * 203. 移除链表元素
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
	 * 206. 反转链表
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
	 * 24. 两两交换链表中的节点
	 */
	public ListNode swapPairs2(ListNode head) {
		if (head == null || head.next == null){
			return head;
		}
		ListNode next = head.next;
		// 假设中间已经是一个按要求排好序的链表
		// 先将当前节点的next改成结果中的next (1 -> 3)
		head.next = swapPairs2(next.next);
		// 将原本的next指向当前节点 (2 -> 1)
		next.next = head;
		return next;
	}

	/**
	 * 19. 删除链表的倒数第 N 个结点
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
     * 面试题 02.07. 链表相交
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
	 * 142. 环形链表 II
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
		// 获取链表长度
		int len = 1;
		while (cur.next != null){
			len++;
			cur = cur.next;
		}
		k = k % len;
		// 不需要旋转
		if (k == 0){
			return head;
		}
		// 头尾相连
		cur.next = head;
		int i = 0;
		// 找到倒数第len-k个节点，也就是链表旋转后的结尾节点
		while (i++ < len - k){
			cur = cur.next;
		}
		// 得到旋转后的头节点
		ListNode newHead = cur.next;
		// 将链表旋转后的结尾节点的next置空
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
		// 创建一个数组，逆序存储链表元素
		int[] res = new int[len];
		int right = len - 1;
		cur = head;
		while (cur != null){
			res[right--] = cur.val;
			cur = cur.next;
		}
		return res;
	}

	class Node {
		int val;
		Node next;
		Node random;

		public Node(int val) {
			this.val = val;
			this.next = null;
			this.random = null;
		}
	}

	public Node copyRandomList(Node head) {
		if (head == null){
			return head;
		}
		// 将链表变为1,1,2,2,3,3...
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
				// 处理复制出的节点的random
				cur.next.random = cur.random.next;
			}
			cur = cur.next.next;
		}
		// 得到复制出的节点的头节点
		Node copyHead = head.next;
		// 将为1,1,2,2,3,3...的链表拆分成1,2,3... 1,2,3...
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

	public ListNode getKthFromEnd(ListNode head, int k) {
		ListNode fast = head;
		ListNode slow = head;
		while (fast != null && k-- > 0){
			fast = fast.next;
		}
		while (fast != null){
			fast = fast.next;
			slow = slow.next;
		}
		return slow;
	}

	public ListNode mergeTwoLists1(ListNode l1, ListNode l2) {
		ListNode cur1 = l1;
		ListNode cur2 = l2;
		ListNode header = new ListNode(0);
		ListNode res = header;
		while (cur1 != null && cur2 != null){
			if (cur1.val < cur2.val){
				res.next = cur1;
				cur1 = cur1.next;
				res = res.next;
			} else {
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

	public ListNode mergeKLists(ListNode[] lists) {
		return merge(lists, 0, lists.length - 1);
	}

	private ListNode merge(ListNode[] lists, int left, int right) {
		// 分成一个链表再进行处理
		if (left == right) {
			return lists[left];
		}
		// 防止 mid+1 大于 right 造成死递归
		if (left > right) {
			return null;
		}
		// 归并排序思想
		int mid = (left + right) >> 1;
		return mergeTwoLists(merge(lists, left, mid), merge(lists, mid + 1, right));
	}

	private ListNode mergeTwoLists12(ListNode l1, ListNode l2) {
		ListNode cur1 = l1;
		ListNode cur2 = l2;
		ListNode header = new ListNode(0);
		ListNode res = header;
		while (cur1 != null && cur2 != null) {
			if (cur1.val < cur2.val) {
				res.next = cur1;
				cur1 = cur1.next;
			} else {
				res.next = cur2;
				cur2 = cur2.next;
			}
			res = res.next;
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

	public ListNode getIntersectionNode12(ListNode headA, ListNode headB) {
		if (headA == null || headB == null){
			return null;
		}
		ListNode cur1 = headA;
		ListNode cur2 = headB;
		while (cur1 != cur2) {
			cur1 = cur1 == null ? headB : cur1.next;
			cur2 = cur2 == null ? headA : cur2.next;
		}
		return cur1;
	}

	public ListNode reverseList3(ListNode head) {
		ListNode pre = null;
		ListNode cur = head;
		while (cur != null) {
			// 暂存next
			ListNode next = cur.next;
			// 交换 next 和 pre
			// pre => cur => next ==> cur => pre
			cur.next = pre;
			pre = cur;
			// 使用暂存的next继续往后遍历
			cur = next;
		}
		return pre;
	}

	ListNode front;

	public boolean isPalindrome(ListNode head) {
		front = head;
		return checkIsPalindrome(head);
	}

	private boolean checkIsPalindrome(ListNode back) {
		// 利用递归的特性，等递归到底开始回溯的时候，得到链表最后一个节点back，实现从后往前遍历的效果
		if (back != null) {
			// 开始递归，如果有一层出现了false，就会一直往上传
			if (!checkIsPalindrome(back.next)) {
				return false;
			}
			// 如果上一层递归是回文，就判断当前的front和back是否相同来判断是否回文
			if (back.val != front.val){
				return false;
			}
			// 移动前指针
			front = front.next;
		}
		// 如果这一层递归正常结束，则认为从front到back是回文
		return true;
	}

	public boolean isPalindrome1(ListNode head) {
		ListNode fast = head;
		ListNode slow = head;
		while (fast.next != null && fast.next.next != null) {
			fast = fast.next.next;
			slow = slow.next;
		}
		// 找到指向链表中间的节点
		ListNode mid = slow;

		// 反转中间之后的节点
		ListNode pre = null;
		ListNode cur = mid.next;
		while (cur != null) {
			ListNode next = cur.next;
			cur.next = pre;
			pre = cur;
			cur = next;
		}

		// 逐步比较反转后的节点，实现从两边向中间遍历的效果
		ListNode right = pre;
		boolean flag = true;
		while (flag && right != null) {
			// 有一个不是回文，那整体都不是回文
			if (head.val != right.val) {
				flag = false;
			}
			head = head.next;
			right = right.next;
		}
		return flag;
	}

	public boolean hasCycle12(ListNode head) {
		if (head == null) {
			return false;
		}
		ListNode fast = head;
		ListNode slow = head;
		while (fast.next != null && fast.next.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (fast == slow) {
				return true;
			}
		}
		return false;
	}

	public ListNode detectCycle1(ListNode head) {
		ListNode fast = head;
		ListNode slow = head;
		// 找到第一个相等的地方
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (fast == slow) {
				// 重置快指针，下一个指针相遇的地方就是入环节点
				while (slow != head) {
					head = head.next;
					slow = slow.next;
				}
				return head;
			}
		}
		return null;
	}

	public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
		// 递归到最后时，会直接将不为null的节点作为next
		if (list1 == null) {
			return list2;
		} else if (list2 == null) {
			return list1;
		} else if (list1.val <= list2.val) {
			// 用更小的节点去指向剩下节点的合并结果，将问题抛给后面的节点来计算
			list1.next = mergeTwoLists(list1.next, list2);
			return list1;
		} else {
			list2.next = mergeTwoLists(list1, list2.next);
			return list2;
		}
	}

	public ListNode addTwoNumbers1(ListNode l1, ListNode l2) {
		ListNode root = new ListNode(0);
		ListNode cur = root;
		// 代表进位 逢十进一
		int carry = 0;
		// carry != 0 时也要进行一次循环，把进位加入节点
		while (l1 != null || l2 != null || carry != 0) {
			int x = l1 == null ? 0 : l1.val;
			int y = l2 == null ? 0 : l2.val;
			// 增加两个节点的数字，要加上前一位的进位
			int sum = x + y + carry;
			carry = sum / 10;
			// 只取个位数字
			ListNode sumNode = new ListNode(sum % 10);
			cur.next = sumNode;
			cur = cur.next;

			if (l1 != null) {
				l1 = l1.next;
			}
			if (l2 != null) {
				l2 = l2.next;
			}
		}
		return root.next;
	}


	int curNum = 0;
	public ListNode removeNthFromEnd12(ListNode head, int n) {
		if (head == null) {
			return null;
		}
		// 递归到底的时候才会开始返回，也就相当于是逆序遍历链表了
		head.next = removeNthFromEnd12(head.next, n);
		curNum++;
		if (curNum == n) {
			return head.next;
		}
		return head;
	}

	public ListNode reverseKGroup1(ListNode head, int k) {
		// 找到下一组的第一个节点，方便跳出循环
		ListNode test = head;
		for (int i = 0; i < k; i++) {
			// 如果这一组不够k个直接返回原节点
			if (test == null) {
				return head;
			}
			test = test.next;
		}
		ListNode cur = head;
		ListNode pre = null;
		// 处理这一组的节点
		while (cur != test) {
			// 将这组的k个节点逐个进行位置互换，也就是反转
			ListNode next = cur.next;
			cur.next = pre;
			pre = cur;
			cur = next;
		}

		// 交给下一组继续处理，cur就是下一组的第一个节点（循环后后cur = test）
		head.next = reverseKGroup(cur, k);
		// 因为反转过，pre指向的就是当前head位置的节点
		return pre;
		// 比如12345的案例，到5时开始出倒数第一层栈，递归结果为5
		// 然后倒数第二层栈，head参数是3，3的next指向倒数第一层的结果5
		// 最后倒数第三层栈，head参数是1，1的next指向倒数第二层的结果3
	}

	public ListNode insertionSortList(ListNode head) {
		ListNode header = new ListNode(0);
		header.next = head;
		// 标记已排序数组的最后节点
		ListNode lastNode = head;
		ListNode cur = head.next;
		while (cur != null) {
			// 当前数比最大数大，则不用排序，相当于直接添加到末尾
			if (lastNode.val <= cur.val) {
				lastNode = lastNode.next;
			} else {
				// 比最大数小，就要用pre指针找到最后一个小于当前数的节点
				ListNode pre = header;
				// 使用next是因为最后pre = pre.next后会指向比cur大的节点
				while (pre.next.val <= cur.val) {
					pre = pre.next;
				}
				// 将下一个需要进行插入排序的节点加入，方便继续向后遍历
				lastNode.next = cur.next;
				// 当前节点的next指向已排序节点中的比它大的第一个节点
				cur.next = pre.next;
				// 修改原链表，真正的将cur加入已排序节点中
				pre.next = cur;
			}
			// cur根据已排序节点改变，因为cur.next可能指向已排序的节点
			cur = lastNode.next;
		}
		return header.next;
	}

	public ListNode sortList1(ListNode head) {
		if (head == null) {
			return null;
		}
		ListNode header = new ListNode(0);
		header.next = head;
		// 标记已排序数组的最后节点
		ListNode lastNode = head;
		ListNode cur = head.next;
		while (cur != null) {
			// 当前数比最大数大，则不用排序，相当于直接添加到末尾
			if (lastNode.val <= cur.val) {
				lastNode = lastNode.next;
			} else {
				// 比最大数小，就要用pre指针找到最后一个小于当前数的节点
				ListNode pre = header;
				// 使用next是因为最后pre = pre.next后会指向比cur大的节点
				while (pre.next.val <= cur.val) {
					pre = pre.next;
				}
				// 将下一个需要进行插入排序的节点加入，方便继续向后遍历
				lastNode.next = cur.next;
				// 当前节点的next指向已排序节点中的比它大的第一个节点
				cur.next = pre.next;
				// 修改原链表，真正的将cur加入已排序节点中
				pre.next = cur;
			}
			// cur根据已排序节点改变，因为cur.next可能指向已排序的节点
			cur = lastNode.next;
		}
		return header.next;
	}

	public ListNode sortList(ListNode head) {
		return sortList(head, null);
	}

	private ListNode sortList(ListNode head, ListNode tail) {
		// 归并排序解法
		if (head == null) {
			return null;
		}
		// 递归结束条件，链表中只有一个节点
		if (head.next == tail) {
			head.next = null;
			return head;
		}
		ListNode slow = head;
		ListNode fast = head;
		while (fast != tail) {
			slow = slow.next;
			fast = fast.next;
			// 还没到当前处理的链表结尾，就让快指针多走一步
			if (fast != tail) {
				fast = fast.next;
			}
		}
		// 当fast到达结尾时，slow就指向了中间的节点
		ListNode mid = slow;
		// 递归分割链表
		ListNode res1 = sortList(head, mid);
		ListNode res2 = sortList(mid, tail);
		// 最后直接使用合并链表的方法合并
		return merge(res1, res2);
	}

	private ListNode merge(ListNode list1, ListNode list2) {
		ListNode header = new ListNode(0);
		ListNode res = header;
		ListNode cur1 = list1;
		ListNode cur2 = list2;
		while (cur1 != null && cur2 != null) {
			if (cur1.val <= cur2.val) {
				res.next = cur1;
				cur1 = cur1.next;
			} else {
				res.next = cur2;
				cur2 = cur2.next;
			}
			res = res.next;
		}
		// 两个链表的节点数相同，可以直接用if-else
		if (cur1 != null) {
			res.next = cur1;
		} else {
			res.next = cur2;
		}
		return header.next;
	}

}

