package top.vita.zuo;

public class FixedX2Y {

	public static void main(String[] args) {

	}
	
	// 题目：将固定但不相等的随机概率转为相等概率
	// 以固定概率返回 0 和 1 
	public static int x() {
		return Math.random() < 0.88 ? 0 : 1;
	}
	
	public static int y(){
		int res = 0;
		do {
			res = x();
		}while (res == x()); // 如果两次函数值都相等，则重新取值 即值只取 1 0 , 0 1
		return res;
	}

}
