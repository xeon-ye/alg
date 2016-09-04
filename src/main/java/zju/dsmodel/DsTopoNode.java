package zju.dsmodel;

import zju.devmodel.MapObject;

import java.io.Serializable;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-10
 */
public class DsTopoNode implements Serializable {
    //节点类型
    public static final int TYPE_SUB = 0;   //变电站节点
    public static final int TYPE_PQ = 1;    //负荷
    public static final int TYPE_DG = 2;    //分布式电源
    public static final int TYPE_LINK = 3;  //联络节点

    private int[] connectedBusNo;

    private int tnNo;

    private double baseKv;

    private List<DsConnectNode> connectivityNodes;

    private int type;

    double[] finalV;

    double[] finalAngle;

    private MapObject[] connectedDev;

    private DsTopoIsland island;

    private int[] phases;

    public void initialPhases() {
        int phaseNum = 0;
        for(int i = 0; i < 3; i++) {
            for (MapObject obj : island.getGraph().edgesOf(this)) {
                if (island.getBranches().containsKey(obj))
                    if (island.getBranches().get(obj).containsPhase(i)) {
                        phaseNum++;
                        break;
                    }
            }
        }
        //通过分析连在节点上的线路，通过@GeneralBranch的containsPhase方法判断是否包含某相
        phases = new int[phaseNum];
        phaseNum = 0;
        for(int i = 0; i < 3; i++) {
            for (MapObject obj : island.getGraph().edgesOf(this)) {
                if (island.getBranches().containsKey(obj))
                    if (island.getBranches().get(obj).containsPhase(i)) {
                        phases[phaseNum++] = i;
                        break;
                    }
            }
        }
        //phases = new int[]{0,1,2};
    }

    /**
     * 判断是否包含某一相位
     * @param phase 相位
     * @return 是否包含某一相位
     */
    public boolean containsPhase(int phase) {
        for (int p : phases)
            if (p == phase)
                return true;
        return false;
    }

    public int getPhaseIndex(int phase) {
        for (int i = 0; i < phases.length; i++)
            if(phases[i] == phase)
                return i;
        return -1;
        //return phase;
    }

    public List<DsConnectNode> getConnectivityNodes() {
        return connectivityNodes;
    }

    public void setConnectivityNodes(List<DsConnectNode> connectivityNodes) {
        this.connectivityNodes = connectivityNodes;
    }

    public int getTnNo() {
        return tnNo;
    }

    public void setTnNo(int tnNo) {
        this.tnNo = tnNo;
    }

    public double getBaseKv() {
        return baseKv;
    }

    public void setBaseKv(double baseKv) {
        this.baseKv = baseKv;
    }

    public int getType() {
        return type;
    }

    public void setType(int type) {
        this.type = type;
    }

    public double[] getFinalV() {
        return finalV;
    }

    public void setFinalV(double[] finalV) {
        this.finalV = finalV;
    }

    public double[] getFinalAngle() {
        return finalAngle;
    }

    public void setFinalAngle(double[] finalAngle) {
        this.finalAngle = finalAngle;
    }

    public int[] getConnectedBusNo() {
        return connectedBusNo;
    }

    public void setConnectedBusNo(int[] connectedBusNo) {
        this.connectedBusNo = connectedBusNo;
    }

    public MapObject[] getConnectedDev() {
        return connectedDev;
    }

    public void setConnectedDev(MapObject[] connectedDev) {
        this.connectedDev = connectedDev;
    }

    public DsTopoIsland getIsland() {
        return island;
    }

    public void setIsland(DsTopoIsland island) {
        this.island = island;
    }

    public int[] getPhases() {
        return phases;
    }
}
