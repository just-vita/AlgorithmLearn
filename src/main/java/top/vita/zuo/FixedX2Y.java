package top.vita.zuo;

public class FixedX2Y {

	public static void main(String[] args) {

	}
	
	// ��Ŀ�����̶�������ȵ��������תΪ��ȸ���
	// �Թ̶����ʷ��� 0 �� 1 
	public static int x() {
		return Math.random() < 0.88 ? 0 : 1;
	}
	
	public static int y(){
		int res = 0;
		do {
			res = x();
		}while (res == x()); // ������κ���ֵ����ȣ�������ȡֵ ��ֵֻȡ 1 0 , 0 1
		return res;
	}

}
