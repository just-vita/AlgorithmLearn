package top.vita.question;

import java.util.HashMap;

public class Review {

	public static void main(String[] args) {

	}

    public boolean wordPattern(String pattern, String s) {
    	String[] strs = s.split(" ");
        if (pattern.length() != strs.length){
    		return false;
    	}
    	HashMap<Character,String> map = new HashMap<Character,String>();
    	for (int i = 0; i < strs.length; i++) {
    		char temp = pattern.charAt(i);
			if (map.containsKey(temp)) {
				if (!map.get(temp).equals(strs[i])) {
					return false;
				}
			}else {
				if(map.containsValue(strs[i])) {
					return false;
				}else {
					map.put(temp,strs[i]);
				}
			}
		}
    	return true;
    }
    
    public String addStrings(String num1, String num2) {
    	StringBuilder sb = new StringBuilder();
    	int l1 = num1.length() - 1;
    	int l2 = num2.length() - 1;
    	int carry = 0;
    	while (l1 >= 0 || l2 >= 0) {
    		int x = l1 < 0 ? 0 : num1.charAt(l1) - '0';
    		int y = l2 < 0 ? 0 : num2.charAt(l2) - '0';
    		
    		int sum = x + y + carry;
    		sb.append(sum & 10);
    		carry = sum / 10;
    		
    		l1--;
    		l2--;
    	}
    	if (carry != 0) {
    		sb.append(carry);
    	}
    	return sb.toString();
    }
    
    public int[] plusOne(int[] digits) {
    	for (int i = digits.length - 1; i >= 0; i--) {
			if (digits[i] != 9) {
				digits[i] ++;
				return digits;
			}
			digits[i] = 0;
		}
    	int[] res = new int[digits.length + 1];
    	res[0] = 1;
    	return res;
    }
    
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
    	HashMap<Integer,Integer> map = new HashMap<Integer, Integer>();
    	int res = 0;
    	for (int i : nums1) {
			for (int j : nums2) {
				int temp = i + j;
				if (map.containsKey(temp)) {
					map.put(temp, map.get(temp) + 1);
				}else {
					map.put(temp, 1);
				}
			}
		}
    	
    	for (int i : nums3) {
			for (int j : nums4) {
				int temp = i + j;
				if (map.containsKey(0 - temp)) {
					res += map.get(0 - temp);
				}
			}
		}
    	return res;
    }
    
    
    
    
    
    
    
    
}
