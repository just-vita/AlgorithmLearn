package top.vita.stack_queue;

import java.util.ArrayDeque;
import java.util.LinkedList;
import java.util.Stack;

public class StackQuestion {

	/*
	 * 1249. �Ƴ���Ч������ 
	 * ����һ���� '('��')' ��Сд��ĸ��ɵ��ַ��� s��
	 * ����Ҫ���ַ�����ɾ��������Ŀ�� '(' ���� ')' ������ɾ������λ�õ�����)��
	 * ʹ��ʣ�µġ������ַ�������Ч��
	 */
    public String minRemoveToMakeValid(String s) {
    	Stack<Integer> stack = new Stack<>();
    	// �洢�������������Ƿ���Ч
    	boolean[] isBrackets = new boolean[s.length()];
    	StringBuilder res = new StringBuilder();
    	for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == '(') {
				// ����������ջ
				stack.push(i);
				// ����ǰ������λ����Ϊtrue
				isBrackets[i] = true;
			}else if(s.charAt(i) == ')') {
				// ���ջ���Ѿ�������������ջ���ж�Ӧ������
				if (!stack.isEmpty()) {
					// ����Ӧ���ŵ�����ȡ������Ϊfalse����Ϊ��Ч������
					isBrackets[stack.pop()] = false;
				} else {
					// ����Ӧλ����Ϊtrue������˴�������Ч
					isBrackets[i] = true;
				}
			}
		}
    	// ����Ч�ַ���������
    	for (int i = 0; i < isBrackets.length; i++) {
			if (!isBrackets[i]) {
				res.append(s.charAt(i));
			}
		}
    	return res.toString();
    }
    
    /*
	 * 1823. �ҳ���Ϸ�Ļ�ʤ�� 
	 * ���� n ��С���һ������Ϸ��С�����Χ��һȦ��
	 * �� ˳ʱ��˳�� �� 1 �� n ��š�ȷ�е�˵���ӵ� i
	 * ��С���˳ʱ���ƶ�һλ�ᵽ��� (i+1) ��С����λ�ã�
	 * ���� 1 <= i < n ���ӵ� n ��С���˳ʱ���ƶ�һλ��ص��� 1 ��С����λ�á�
	 */
    public int findTheWinner(int n, int k) {
    	int p = 0;
    	for (int i = 2; i <= n; i++) {
			p = (p + k) % i;
		}
    	return p + 1;
    }

	/**
	 * 20. ��Ч������
	 */
	public boolean isValid(String s) {
		Stack<Character> stack = new Stack<>();
		for (char c : s.toCharArray()) {
			if (c == '('){
				stack.push(')');
			} else if (c == '['){
				stack.push(']');
			} else if (c == '{'){
				stack.push('}');
			} else if (stack.isEmpty() || stack.peek() != c){
				// �����������ţ�����ջ�ǿյģ������������Ƕ����
				// ��Ϊ�յĻ������ջ��Ԫ������������ŵ����Ͳ�ͬ��Ҳ�������
				return false;
			} else{
				// ������������
				stack.pop();
			}
		}
		return stack.isEmpty();
	}

	/**
	 * 1047. ɾ���ַ����е����������ظ���
	 */
	public String removeDuplicates(String s) {
		ArrayDeque<Character> stack = new ArrayDeque<>();
		for (char ch : s.toCharArray()){
			if (stack.isEmpty() || stack.peek() != ch){
				stack.push(ch);
			} else{
				stack.pop();
			}
		}
		String res = "";
		while (!stack.isEmpty()){
			// ��תջ�е��ַ�
			res = stack.pop() + res;
		}
		return res;
	}

	/**
	 * 150. �沨�����ʽ��ֵ
	 */
	public int evalRPN(String[] tokens) {
		ArrayDeque<Integer> stack = new ArrayDeque<>();
		for (String s : tokens){
			if ("+".equals(s)){
				stack.push(stack.pop() + stack.pop());
			} else if ("-".equals(s)){
				// ���⴦��
				stack.push(-stack.pop() + stack.pop());
			} else if ("*".equals(s)){
				stack.push(stack.pop() * stack.pop());
			} else if ("/".equals(s)){
				// ���⴦��
				Integer tmp1 = stack.pop();
				Integer tmp2 = stack.pop();
				stack.push(tmp2 / tmp1);
			} else{
				stack.push(Integer.valueOf(s));
			}
		}
		return stack.pop();
	}




}
