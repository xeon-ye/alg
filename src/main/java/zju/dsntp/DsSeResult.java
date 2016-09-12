package zju.dsntp;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-11-11
 */
public class DsSeResult implements Serializable {
    int activeIslandNum;

    int convergedNum;

    public int getActiveIslandNum() {
        return activeIslandNum;
    }

    public void setActiveIslandNum(int activeIslandNum) {
        this.activeIslandNum = activeIslandNum;
    }

    public int getConvergedNum() {
        return convergedNum;
    }

    public void setConvergedNum(int convergedNum) {
        this.convergedNum = convergedNum;
    }
}
