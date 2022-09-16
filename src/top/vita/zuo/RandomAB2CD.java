package top.vita.zuo;

public class RandomAB2CD {
	public static void main(String[] args) {
		int time = 10000;
		int[] counts = new int[64];
		int count = 0;
		for (int i = 0; i < time; i++) {
			counts[g2()]++;
		}
//		for (int i = 1;i < 6; i++) {
//			System.out.printf("%d 出现了 %d 次 \n",i,counts[i]);
//		}
//		for (int i = 0;i <= 1; i++) {
//			System.out.printf("%d 出现了 %d 次 \n",i,counts[i]);
//		}
//		for (int i = 0; i <= 7; i++) {
//			System.out.printf("%d 出现了 %d 次 \n", i, counts[i]);
//		}
		for (int i = 0; i <= 63; i++) {
			System.out.printf("%d 出现了 %d 次 \n", i, counts[i]);
		}
	}
	
	// -------- 等概率获取 1 - 7 --------------
	
	// 原函数
	// 等概率获取 1 - 5
	public static int f1() {
		return (int)(Math.random() * 5 + 1);
	}
	
	// 等概率获取 0 和 1
	public static int f2() {
		int res = 0;
		do {
			res = f1();
		}while (res == 3);
		return res < 3 ? 0 : 1;
	}
	
	// 等概率获取 0 - 7
	public static int f3() {
		// 三位二进制位
		return (f2() << 2) + (f2() << 1) + (f2() << 0);
	}
	
	// 等概率获取 0 - 6
	public static int f4() {
		int res = 0;
		do {
			res = f3();
		}while (res == 7);
		return res;
	}
	
	// 等概率获取 1 - 7
	public static int g1() {
		return f4() + 1;
	}
	
	// -------- 等概率获取 3 - 19 --------------
	
	// 原函数
	// 等概率获取 1 - 5
	public static int f5() {
		return (int) (Math.random() * 5 + 1);
	}
	
	// 等概率获取 0 和 1
	public static int f6() {
		int res = 0;
		do {
			res = f5();
		}while(res == 3);
		return res < 3 ? 0 : 1;
	}
	
	// 等概率获取 0 - 31
	public static int f7() {
		return ((f6() << 4) + (f6() << 3) + (f6() << 2) + (f6() << 1) + (f6() << 0));
	}

	// 等概率获取 3 - 19
	public static int g2() {
		int res = 0;
		do {
			res = f7();
		}while (res > 16);
		return res + 3;
	}
	
	// -------- 等概率获取 17 - 56 --------------
	
	// 原函数
	// 等概率获取 1 - 5
	public static int f8() {
		return (int) (Math.random() * 5 + 1);
	}
	
	// 等概率获取 0 和 1
	public static int f9() {
		int res = 0;
		do {
			res = f8();
		}while (res == 3);
		return res < 3 ? 0 : 1;
	}
	
	// 等概率获取 0 - 63
	public static int f10() {
		// 六位二进制位
		return (f9() << 5) + (f9() << 4) + (f9() << 3) + 
				(f9() << 2) + (f9() << 1) + (f9() << 0);
	}
	
	// 等概率获取 0 - 38
	public static int f11() {
		int res = 0;
		do {
			res = f10();
		}while (res > 39);
		return res;
	}
	
	// 等概率获取 17 - 56
	public static int g3() {
		return f11() + 17;
	}
	
	// ----------------------------
	// 1 - 7 -> 1 - 10
    public int rand10() {
    	int res = 0;
    	do {
    		res = (f() << 3) + (f() << 2) + (f() << 1) + (f() << 0);
    	}while (res > 9);
        return res + 1;
    }

    public int f(){
        int res = 0;
        do {
        	res=rand7();
        }while (res == 4);
        return res < 4 ? 0 : 1; 
    }
    
	private static int rand7() {
		return (int)(Math.random() * 7 + 1);
	}
}
