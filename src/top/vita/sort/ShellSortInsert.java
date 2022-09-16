package top.vita.sort;

import java.util.Arrays;

public class ShellSortInsert {

	public static void main(String[] args) {
		int[] nums = {-4,-1,2,5,0,3,10};
		shellSortInsert(nums);
	}

	public static void shellSortInsert(int[] nums) {
		for (int gap = nums.length / 2; gap > 0; gap /= 2) {
			for (int i = gap; i < nums.length; i++) {
				int j = i;
				int temp = nums[j];
				while (j - gap >= 0 && temp < nums[j - gap]) {
					nums[j] = nums[j - gap];
					j = j - gap;
				}
				
				nums[j] = temp;
			}
		}
		System.out.println(Arrays.toString(nums));
	}
}
