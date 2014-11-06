package zju.util;

import java.util.ArrayList;

public class CombinationAlg {

    public CombinationAlg() {
    }

    //calculate factorial
    public static long factorial(int num) {
        long res = 1;
        int i;
        for (i = 1; i <= num; i++) {
            res = res * i;
        }
        return res;
    }

    public void show_v(ArrayList comb) {
        for (Object aComb : comb) {
            //printit
            System.out.println(aComb.toString());
        }
    }

    @SuppressWarnings("unchecked")
    protected void action(int v[], int k) {
        String n = "";
        for (int i = 1; i <= k; i++) {
            n = n + String.valueOf(v[i]) + ",";
        }
        System.out.println(n);
    }

    /*
    combin is a recursive function for creating n per k combinations
    For example if n=5 and k=2 then the derived comb would have the form:
    comb={1,2, 1,3, 1,4, 1,5, 2,3, 2,4, 2,5, 3,4, 3,5, 4,5}
     */
    void combin(int v[], int m, int n, int k) {
        int i;
        if (m > k) {
            action(v, k);
        } else {
            for (i = v[m - 1] + 1; i <= n - k + m; i++) {
                v[m] = i;
                //recursive function, it calls itself
                combin(v, m + 1, n, k);
            }
        }
    }

    public void nchoosek(int n, int k) {
        int m = 1;
        //find the number of conbinations (rows) that will be created
        //int rows = (int) (factorial(n) / (factorial(k) * factorial(n - k)));
        int[] v = new int[k + 1];
        //put 0s to v vector
        //initialize(v,rows);
        //combine numbers n per k and add them to vector comb
        combin(v, m, n, k);
    } // end of nchoosek

    void initialize(double v[], double matrix_len) {
        for (int i = 0; i < matrix_len; i++) {
            v[i] = 0;
        }
    }

    public static void main(String[] args) {
        int n = 15;
        int k = 4;
        CombinationAlg aa = new CombinationAlg();
        ArrayList comb = new ArrayList();
        aa.nchoosek(n, k);
        aa.show_v(comb);
    }
}
