package zju.se.bdfilter;

import zju.ieeeformat.BranchData;
import zju.measure.MeasureInfo;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-11-26
 */
public class BusBasedFilter {
    //母线所连接的支路
    private BranchData[] branches;
    //有功量测
    private MeasureInfo[] meases;
    //相角
    private double[] state;

    public BusBasedFilter() {
    }

    public BusBasedFilter(BranchData[] branches, MeasureInfo[] meases) {
        setBranches(branches);
        this.meases = meases;
    }

    public boolean isWorthCluster() {
        return false;//todo:
    }

    public void doFilter() {
        int num = 1;
        int[] tmp = new int[branches.length];
        for (int i = 0; i < branches.length; i++) {
            num *= 2;
            tmp[i] = 1;
            for (int j = 0; j < i; j++)
                tmp[i] *= 2;
        }
        //第一步，所有有功量测均来自于线路
        boolean flag;
        for (int k = 0; k < num; k++) {
            flag = true;
            for (int i = 0; i < branches.length; i++) {
                if ((k & tmp[i]) == 0) { //pij
                    if (meases[2 * i + 1] == null) {
                        flag = false;
                        break;
                    }
                    state[i] = meases[2 * i + 1].getValue() / branches[i].getBranchX();
                } else { //pji
                    if (meases[2 * i + 2] == null) {
                        flag = false;
                        break;
                    }
                    state[i] = -meases[2 * i + 2].getValue() / branches[i].getBranchX();
                }
            }
            if (!flag)
                continue;
            //AVector.printOnScreen(state);
            //AVector.printOnScreen(BranchBasedFilter.numFormat, state);
        }

        //第二步，有一个有功量测来自节点，其余有功量测均来自于线路
        num /= 2;
        int tmp2;
        double p_notselected;
        for (int notSelected = 0; notSelected < branches.length; notSelected++) {
            flag = true;
            for (int k = 0; k < num; k++) {
                p_notselected = meases[0].getValue();
                for (int i = 0; i < branches.length; i++) {
                    if (i == notSelected)
                        continue;
                    if(i < notSelected)
                        tmp2 = tmp[i];
                    else
                        tmp2 = tmp[i - 1];
                    if ((k & tmp2) == 0) {
                        if (meases[2 * i + 1] == null) {
                            flag = false;
                            break;
                        }
                        state[i] = meases[2 * i + 1].getValue() / branches[i].getBranchX();
                        p_notselected -= meases[2 * i + 1].getValue();
                    } else {
                        if (meases[2 * i + 2] == null) {
                            flag = false;
                            break;
                        }
                        state[i] = -meases[2 * i + 2].getValue() / branches[i].getBranchX();
                        p_notselected += meases[2 * i + 2].getValue();
                    }
                    if(!flag)
                        break;
                }
                if (!flag)
                    continue;
                state[notSelected] = p_notselected / branches[notSelected].getBranchX();
                //AVector.printOnScreen(state);
                //AVector.printOnScreen(BranchBasedFilter.numFormat, state);
            }
        }
    }

    public void setBranches(BranchData[] branches) {
        this.branches = branches;
        state = new double[branches.length];
    }

    public void setMeases(MeasureInfo[] meases) {
        this.meases = meases;
    }
}