package top.vita.sort;

import cn.hutool.core.util.ArrayUtil;
import cn.hutool.core.util.RandomUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * �����㷨
 *
 * @Author vita
 * @Date 2023/8/4 12:24
 */
public class SortAlgorithm {

    public static void main(String[] args) {
        SortAlgorithm sortAlgorithm = new SortAlgorithm();
        int[] arr = RandomUtil.randomInts(100);
        int[] arr2 = RandomUtil.randomInts(100);
        int[] ints = ArrayUtil.addAll(arr, arr2, new int[]{-1, -2});
        System.out.println(Arrays.toString(ints));
//        sortAlgorithm.quickSort1(arr, 0, arr.length - 1);
        sortAlgorithm.bucketSort(ints);
//        System.out.println(Arrays.toString(arr));
//        arr = sortAlgorithm.bucketSort(ints);
        System.out.println(Arrays.toString(ints));
    }

    public void bubbleSort(int[] arr) {
        for (int i = 1; i < arr.length; i++) {
            boolean flag = false;
            for (int j = 0; j < arr.length - i; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                    flag = true;
                }
            }
            if (!flag) {
                break;
            }
        }
    }

    public void selectSort(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            int minIndex = i;
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[j] < arr[minIndex]) {
                    minIndex = j;
                }
            }
            if (minIndex != i) {
                int temp = arr[minIndex];
                arr[minIndex] = arr[i];
                arr[i] = temp;
            }
        }
    }

    public void insertSort(int[] arr) {
        for (int i = 1; i < arr.length; i++) {
            int preIndex = i - 1;
            int cur = arr[i];
            while (preIndex >= 0 && cur < arr[preIndex]) {
                // ����������������У���ǰ���Ƚϴ�������һλ
                arr[preIndex + 1] = arr[preIndex];
                // ȥ��ǰһ����
                preIndex--;
            }
            // ����С������λ���ҵ��ˣ�����С�����������
            arr[preIndex + 1] = cur;
        }
    }

    public void shellSort(int[] arr) {
        for (int gap = arr.length / 2; gap > 0; gap /= 2) {
            for (int i = 0; i < arr.length; i++) {
                int preIndex = i - gap;
                int cur = arr[i];
                while (preIndex >= 0 && cur < arr[preIndex]) {
                    // ����������������У���ǰ���Ƚϴ�������һλ
                    arr[preIndex + gap] = arr[preIndex];
                    // ȥ��ǰһ����
                    preIndex -= gap;
                }
                // ����С������λ���ҵ��ˣ�����С�����������
                arr[preIndex + gap] = cur;
            }
        }
    }

    public int[] mergeSort(int[] arr) {
        if (arr.length <= 1) {
            return arr;
        }
        int mid = arr.length / 2;
        int[] arr1 = Arrays.copyOfRange(arr, 0, mid);
        int[] arr2 = Arrays.copyOfRange(arr, mid, arr.length);
        return merge(mergeSort(arr1), mergeSort(arr2));
    }

    private int[] merge(int[] arr1, int[] arr2) {
        int[] sortedArr = new int[arr1.length + arr2.length];
        int sortedIndex = 0;
        int arr1Index = 0;
        int arr2Index = 0;
        while (arr1Index < arr1.length && arr2Index < arr2.length) {
            if (arr1[arr1Index] < arr2[arr2Index]) {
                sortedArr[sortedIndex++] = arr1[arr1Index++];
            } else {
                sortedArr[sortedIndex++] = arr2[arr2Index++];
            }
        }
        // ѭ�������������л�����û��ȡ��������ֱ��ȫ����������
        while (arr1Index < arr1.length) {
            sortedArr[sortedIndex++] = arr1[arr1Index++];
        }
        while (arr2Index < arr2.length) {
            sortedArr[sortedIndex++] = arr2[arr2Index++];
        }
        return sortedArr;
    }

    public void quickSort(int[] arr, int left, int right) {
        if (left < right) {
            // �õ�����õ��������ֵ�λ��
            int partition = partition(arr, left, right);
            // �������������ҵ����ֵݹ�����
            quickSort(arr, left, partition - 1);
            quickSort(arr, partition + 1, right);
        }
    }

    private int partition(int[] arr, int left, int right) {
        // ȡ���Ҳ��������Ϊ��������
        int pivot = arr[right];
        // ָ����������ִ��ָ��
        int pointer = left;
        for (int i = left; i < right; i++) {
            if (arr[i] <= pivot) {
                // ��Ӧ��������������ߵ�����Ӧ�����ұߵ�����λ��
                int temp = arr[i];
                arr[i] = arr[pointer];
                arr[pointer] = temp;
                pointer++;
            }
        }
        // ���������ֺ�ָ��ָ������ֽ���λ��
        int temp = arr[right];
        arr[right] = arr[pointer];
        arr[pointer] = temp;
        return pointer;
    }

    public void quickSort1(int[] arr, int L, int R) {
        // ָ��Ӧ�������������ұߵ���
        int left = L;
        // ָ��Ӧ��������������ߵ���
        int right = R;
        // �������������Ϊ���������ó���������ʹ��ճ���λ
        int pivot = arr[left];
        while (left < right) {
            // Ѱ��Ӧ�������������ұߵ���
            while (left < right && arr[left] < pivot) {
                left++;
            }
            // �ҵ����֣�ֱ�Ӹ�Ӧ������ߵ����ֻ�λ�ã���ͬ���滻
            if (arr[left] > pivot && arr[left] != arr[right]) {
                arr[right] = arr[left];
            }
            // Ѱ��Ӧ��������������ߵ���
            while (left < right && arr[right] > pivot) {
                right--;
            }
            // �ҵ����֣�ֱ�Ӹ�Ӧ�����ұߵ����ֻ�λ�ã���ͬ���滻
            if (arr[right] < pivot && arr[left] != arr[right]) {
                arr[left] = arr[right];
            }
            // ���ָ���غϣ�������Ѿ��ҵ��м�λ�ã����������ַ���
            if (left == right) {
                arr[left] = pivot;
            }
        }
        if (right - 1 >= L) {
            quickSort1(arr, L, right - 1);
        }
        if (left + 1 <= R) {
            quickSort1(arr, left + 1, R);
        }
    }

    public void heapSort(int[] arr) {
        // ���ҵ�һ����Ҷ�ӽڵ�Ĺ�ʽΪ n / 2 - 1
        for (int i = arr.length / 2 - 1; i >= 0; i--) {
            // ��ʼ��һ���󶥶ѣ���ʱ i ��Խ��ԽС
            adjustHead(arr, i, arr.length);
        }
        // ���������֣����ڵ㣩���β���ֽ���
        for (int j = arr.length - 1; j >= 0; j--) {
            // ����С�����ŵ� j ��λ��
            int temp = arr[0];
            arr[0] = arr[j];
            arr[j] = temp;
            // ������������ʱ j ��Խ��ԽС
            // ���뷽��ʱ j ָ�������С���������������[0, j]��������
            adjustHead(arr, 0, j);
        }
    }

    private void adjustHead(int[] arr, int i, int length) {
        // ���浱ǰ�ڵ��ֵ���ֲ��ĸ��ڵ�
        int temp = arr[i];
        // ���ҽڵ�����ӽڵ�Ĺ�ʽΪ n * 2 + 1
        for (int k = i * 2 + 1; k < length; k = k * 2 + 1) {
            // ��������ӽڵ����kָ�����ӽڵ㣨ָ�����ҽڵ������Ľڵ�
            if (k + 1 < length && arr[k] < arr[k + 1]) {
                k++;
            }
            // ����ӽڵ�ȸ��ڵ㻹��
            if (arr[k] > temp) {
                // ����ǰ�ĸ��ڵ������Ϊ��ǰ������
                arr[i] = arr[k];
                // ��¼�±꣬������ѭ��������ֱ�ӽ�ԭ���ĸ��ڵ����õ����ʵ�λ��
                i = k;
            } else {
                // ����ӽڵ��λ�ö���ȷ�Ļ����ʹ�����ζѵĵ���������
                // ֻ��Ϊ�˼ӿ������ٶ�
                break;
            }
        }
        // ��ԭ���ĸ��ڵ�ŵ����ʵ�λ��
        arr[i] = temp;
    }

    public int[] countingSort(int[] arr) {
        int max = 0;
        for (int i : arr) {
            if (max < i) {
                max = i;
            }
        }
        int[] countArr = new int[max + 1];
        // ��¼���ֳ��ֵĴ���
        for (int i = 0; i < arr.length; i++) {
            countArr[arr[i]]++;
        }
        // �������ǰλ�õ�ǰ׺��
        for (int i = 1; i < countArr.length; i++) {
            countArr[i] += countArr[i - 1];
        }
        int[] res = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            // ͨ�����ֵ�ǰ׺�ͼ�һ���ɵõ�����Ӧ�÷ŵ�λ��
            res[countArr[arr[i]] - 1] = arr[i];
            // ���ָ�����һ
            countArr[arr[i]]--;
        }
        return res;
    }

    public int[] countingSort1(int[] arr) {
        int max = 0;
        int min = 0;
        for (int i : arr) {
            if (max < i) {
                max = i;
            } else if (min > i) {
                min = i;
            }
        }
        // ����и������ڣ���ȥ�������൱�ڼ��ϸ����ľ���ֵ
        // ����и��������ڣ��������������ظ����������Խ�ʡһ�������ֵĿռ�
        // ���� [1, 1, 2, 3] ��ǰ׺������ֻ��Ҫ��� [1, 2, 3]
        int[] countArr = new int[max - min + 1];
        // ��¼���ֳ��ֵĴ���
        for (int i = 0; arr.length > i; i++) {
            countArr[arr[i] - min]++;
        }
        // �������ǰλ�õ�ǰ׺��
        for (int i = 1; i < countArr.length; i++) {
            countArr[i] += countArr[i - 1];
        }
        int[] res = new int[arr.length];
        for (int j : arr) {
            // ͨ�����ֵ�ǰ׺�ͼ�ȥ��Сֵ���ɵõ�����Ӧ�÷ŵ�λ��
            res[countArr[j - min] - 1] = j;
            // ���ָ�����һ
            countArr[j - min]--;
        }
        return res;
    }

    public int[] countingSort2(int[] arr) {
        int max = 0;
        int min = 0;
        for (int i : arr) {
            if (max < i) {
                max = i;
            } else if (min > i) {
                min = i;
            }
        }
        // ����и������ڣ���ȥ�������൱�ڼ��ϸ����ľ���ֵ
        // ����и��������ڣ��������������ظ����������Խ�ʡһ�������ֵĿռ�
        // ���� [1, 1, 2, 3] ��ǰ׺������ֻ��Ҫ��� [1, 2, 3]
        int[] countArr = new int[max - min + 1];
        // ��¼���ֳ��ֵĴ���
        for (int j : arr) {
            countArr[j - min]++;
        }
        // �������ǰλ�õ�ǰ׺��
        for (int i = 1; i < countArr.length; i++) {
            countArr[i] += countArr[i - 1];
        }
        int[] res = new int[arr.length];
        for (int j : arr) {
            // ͨ�����ֵ�ǰ׺�ͼ�ȥ��Сֵ���ɵõ�����Ӧ�÷ŵ�λ��
            res[countArr[j - min] - 1] = j;
            // ���ָ�����һ
            countArr[j - min]--;
        }
        return res;
    }

    public void bucketSort(int[] arr) {
        if (arr.length == 0) {
            return;
        }
        // ÿ��Ͱ���Դ�ŵ�����
        int bucketSize = 30;

        // �ҵ������е����ֵ����Сֵ
        int minValue = arr[0];
        int maxValue = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] < minValue) {
                minValue = arr[i];
            } else if (arr[i] > maxValue) {
                maxValue = arr[i];
            }
        }

        // ����Ͱ������
        int bucketCount = (maxValue - minValue) / bucketSize + 1;
        ArrayList<ArrayList<Integer>> buckets = new ArrayList<>(bucketCount);
        for (int i = 0; i < bucketCount; i++) {
            buckets.add(new ArrayList<>());
        }

        // ��Ԫ�ط��䵽��Ӧ��Ͱ��
        for (int i : arr) {
            int bucketIndex = (i - minValue) / bucketSize;
            buckets.get(bucketIndex).add(i);
        }

        // ��ÿ��Ͱ�������򣬲���������Ԫ�طŻ�ԭ����
        int currentIndex = 0;
        for (int i = 0; i < bucketCount; i++) {
            ArrayList<Integer> bucket = buckets.get(i);
            Collections.sort(bucket);
            for (Integer integer : bucket) {
                arr[currentIndex++] = integer;
            }
        }
    }

}

