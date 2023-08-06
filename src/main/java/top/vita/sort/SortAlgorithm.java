package top.vita.sort;

import cn.hutool.core.util.ArrayUtil;
import cn.hutool.core.util.RandomUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * 排序算法
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
                // 发现已排序的数组中，当前数比较大，往后移一位
                arr[preIndex + 1] = arr[preIndex];
                // 去看前一个数
                preIndex--;
            }
            // 比它小的数的位置找到了，在最小的数后面插入
            arr[preIndex + 1] = cur;
        }
    }

    public void shellSort(int[] arr) {
        for (int gap = arr.length / 2; gap > 0; gap /= 2) {
            for (int i = 0; i < arr.length; i++) {
                int preIndex = i - gap;
                int cur = arr[i];
                while (preIndex >= 0 && cur < arr[preIndex]) {
                    // 发现已排序的数组中，当前数比较大，往后移一位
                    arr[preIndex + gap] = arr[preIndex];
                    // 去看前一个数
                    preIndex -= gap;
                }
                // 比它小的数的位置找到了，在最小的数后面插入
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
        // 循环结束后数组中还有数没有取出来，就直接全部放入结果集
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
            // 得到排序好的中心数字的位置
            int partition = partition(arr, left, right);
            // 对中心数字左右的数字递归排序
            quickSort(arr, left, partition - 1);
            quickSort(arr, partition + 1, right);
        }
    }

    private int partition(int[] arr, int left, int right) {
        // 取最右侧的数字作为中心数字
        int pivot = arr[right];
        // 指向比中心数字大的指针
        int pointer = left;
        for (int i = left; i < right; i++) {
            if (arr[i] <= pivot) {
                // 将应该在中心数字左边的数和应该在右边的数换位置
                int temp = arr[i];
                arr[i] = arr[pointer];
                arr[pointer] = temp;
                pointer++;
            }
        }
        // 将中心数字和指针指向的数字交换位置
        int temp = arr[right];
        arr[right] = arr[pointer];
        arr[pointer] = temp;
        return pointer;
    }

    public void quickSort1(int[] arr, int L, int R) {
        // 指向应该在中心数字右边的数
        int left = L;
        // 指向应该在中心数字左边的数
        int right = R;
        // 把最左的数字作为中心数字拿出来，想象使其空出空位
        int pivot = arr[left];
        while (left < right) {
            // 寻找应该在中心数字右边的数
            while (left < right && arr[left] < pivot) {
                left++;
            }
            // 找到数字，直接跟应该在左边的数字换位置，相同则不替换
            if (arr[left] > pivot && arr[left] != arr[right]) {
                arr[right] = arr[left];
            }
            // 寻找应该在中心数字左边的数
            while (left < right && arr[right] > pivot) {
                right--;
            }
            // 找到数字，直接跟应该在右边的数字换位置，相同则不替换
            if (arr[right] < pivot && arr[left] != arr[right]) {
                arr[left] = arr[right];
            }
            // 如果指针重合，则代表已经找到中间位置，将中心数字放入
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
        // 查找第一个非叶子节点的公式为 n / 2 - 1
        for (int i = arr.length / 2 - 1; i >= 0; i--) {
            // 初始化一个大顶堆，此时 i 会越来越小
            adjustHead(arr, i, arr.length);
        }
        // 将最大的数字（根节点）与结尾数字交换
        for (int j = arr.length - 1; j >= 0; j--) {
            // 将最小的数放到 j 的位置
            int temp = arr[0];
            arr[0] = arr[j];
            arr[j] = temp;
            // 继续调整，此时 j 会越来越小
            // 进入方法时 j 指向的是最小的数，调整后会变成[0, j]中最大的数
            adjustHead(arr, 0, j);
        }
    }

    private void adjustHead(int[] arr, int i, int length) {
        // 保存当前节点的值（局部的根节点
        int temp = arr[i];
        // 查找节点的左子节点的公式为 n * 2 + 1
        for (int k = i * 2 + 1; k < length; k = k * 2 + 1) {
            // 如果是右子节点大，则将k指向右子节点（指向左右节点中最大的节点
            if (k + 1 < length && arr[k] < arr[k + 1]) {
                k++;
            }
            // 如果子节点比根节点还大
            if (arr[k] > temp) {
                // 将当前的根节点的数改为当前最大的数
                arr[i] = arr[k];
                // 记录下标，方便在循环结束后直接将原本的根节点设置到合适的位置
                i = k;
            } else {
                // 如果子节点的位置都正确的话，就代表这次堆的调整结束了
                // 只是为了加快排序速度
                break;
            }
        }
        // 将原本的根节点放到合适的位置
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
        // 记录数字出现的次数
        for (int i = 0; i < arr.length; i++) {
            countArr[arr[i]]++;
        }
        // 计算出当前位置的前缀和
        for (int i = 1; i < countArr.length; i++) {
            countArr[i] += countArr[i - 1];
        }
        int[] res = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            // 通过数字的前缀和减一即可得到数字应该放的位置
            res[countArr[arr[i]] - 1] = arr[i];
            // 数字个数减一
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
        // 如果有负数存在，减去负数就相当于加上负数的绝对值
        // 如果有负数不存在，代表数组中有重复的数，可以节省一点存放数字的空间
        // 比如 [1, 1, 2, 3] 的前缀和数组只需要存放 [1, 2, 3]
        int[] countArr = new int[max - min + 1];
        // 记录数字出现的次数
        for (int i = 0; arr.length > i; i++) {
            countArr[arr[i] - min]++;
        }
        // 计算出当前位置的前缀和
        for (int i = 1; i < countArr.length; i++) {
            countArr[i] += countArr[i - 1];
        }
        int[] res = new int[arr.length];
        for (int j : arr) {
            // 通过数字的前缀和减去最小值即可得到数字应该放的位置
            res[countArr[j - min] - 1] = j;
            // 数字个数减一
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
        // 如果有负数存在，减去负数就相当于加上负数的绝对值
        // 如果有负数不存在，代表数组中有重复的数，可以节省一点存放数字的空间
        // 比如 [1, 1, 2, 3] 的前缀和数组只需要存放 [1, 2, 3]
        int[] countArr = new int[max - min + 1];
        // 记录数字出现的次数
        for (int j : arr) {
            countArr[j - min]++;
        }
        // 计算出当前位置的前缀和
        for (int i = 1; i < countArr.length; i++) {
            countArr[i] += countArr[i - 1];
        }
        int[] res = new int[arr.length];
        for (int j : arr) {
            // 通过数字的前缀和减去最小值即可得到数字应该放的位置
            res[countArr[j - min] - 1] = j;
            // 数字个数减一
            countArr[j - min]--;
        }
        return res;
    }

    public void bucketSort(int[] arr) {
        if (arr.length == 0) {
            return;
        }
        // 每个桶可以存放的数量
        int bucketSize = 30;

        // 找到数组中的最大值和最小值
        int minValue = arr[0];
        int maxValue = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] < minValue) {
                minValue = arr[i];
            } else if (arr[i] > maxValue) {
                maxValue = arr[i];
            }
        }

        // 计算桶的数量
        int bucketCount = (maxValue - minValue) / bucketSize + 1;
        ArrayList<ArrayList<Integer>> buckets = new ArrayList<>(bucketCount);
        for (int i = 0; i < bucketCount; i++) {
            buckets.add(new ArrayList<>());
        }

        // 将元素分配到对应的桶中
        for (int i : arr) {
            int bucketIndex = (i - minValue) / bucketSize;
            buckets.get(bucketIndex).add(i);
        }

        // 对每个桶进行排序，并将排序后的元素放回原数组
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

