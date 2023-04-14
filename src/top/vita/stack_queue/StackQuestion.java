package top.vita.stack_queue;

import java.util.ArrayDeque;
import java.util.LinkedList;
import java.util.Stack;

public class StackQuestion {

	/*
	 * 1249. 移除无效的括号 
	 * 给你一个由 '('、')' 和小写字母组成的字符串 s。
	 * 你需要从字符串中删除最少数目的 '(' 或者 ')' （可以删除任意位置的括号)，
	 * 使得剩下的「括号字符串」有效。
	 */
    public String minRemoveToMakeValid(String s) {
    	Stack<Integer> stack = new Stack<>();
    	// 存储索引处的括号是否有效
    	boolean[] isBrackets = new boolean[s.length()];
    	StringBuilder res = new StringBuilder();
    	for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == '(') {
				// 将索引推入栈
				stack.push(i);
				// 将当前索引的位置设为true
				isBrackets[i] = true;
			}else if(s.charAt(i) == ')') {
				// 如果栈中已经有索引，代表栈中有对应的括号
				if (!stack.isEmpty()) {
					// 将对应括号的索引取出，设为false，即为有效的括号
					isBrackets[stack.pop()] = false;
				} else {
					// 将对应位置设为true，代表此处括号无效
					isBrackets[i] = true;
				}
			}
		}
    	// 将有效字符加入结果集
    	for (int i = 0; i < isBrackets.length; i++) {
			if (!isBrackets[i]) {
				res.append(s.charAt(i));
			}
		}
    	return res.toString();
    }
    
    /*
	 * 1823. 找出游戏的获胜者 
	 * 共有 n 名小伙伴一起做游戏。小伙伴们围成一圈，
	 * 按 顺时针顺序 从 1 到 n 编号。确切地说，从第 i
	 * 名小伙伴顺时针移动一位会到达第 (i+1) 名小伙伴的位置，
	 * 其中 1 <= i < n ，从第 n 名小伙伴顺时针移动一位会回到第 1 名小伙伴的位置。
	 */
    public int findTheWinner(int n, int k) {
    	int p = 0;
    	for (int i = 2; i <= n; i++) {
			p = (p + k) % i;
		}
    	return p + 1;
    }

	/**
	 * 20. 有效的括号
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
				// 遇到了右括号，但是栈是空的，代表右括号是多余的
				// 不为空的话，如果栈顶元素与这个右括号的类型不同，也代表多余
				return false;
			} else{
				// 遇到右括号了
				stack.pop();
			}
		}
		return stack.isEmpty();
	}

	/**
	 * 1047. 删除字符串中的所有相邻重复项
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
			// 反转栈中的字符
			res = stack.pop() + res;
		}
		return res;
	}

	/**
	 * 150. 逆波兰表达式求值
	 */
	public int evalRPN(String[] tokens) {
		ArrayDeque<Integer> stack = new ArrayDeque<>();
		for (String s : tokens){
			if ("+".equals(s)){
				stack.push(stack.pop() + stack.pop());
			} else if ("-".equals(s)){
				// 特殊处理
				stack.push(-stack.pop() + stack.pop());
			} else if ("*".equals(s)){
				stack.push(stack.pop() * stack.pop());
			} else if ("/".equals(s)){
				// 特殊处理
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
