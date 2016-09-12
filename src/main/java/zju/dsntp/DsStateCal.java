package zju.dsntp;

import zju.devmodel.MapObject;
import zju.dsmodel.DsTopoIsland;
import zju.dsmodel.DsTopoNode;
import zju.matrix.AVector;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;
import zju.util.MathUtil;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2009-3-25
 */
public class DsStateCal implements MeasTypeCons {//todo: test is needed and not finished...

    private DsTopoIsland island;

    private double[] tmpV;

    private double[] tmpI;

    public DsStateCal() {
        tmpV = new double[2];
        tmpI = new double[2];
    }

    public DsStateCal(DsTopoIsland island) {
        this();
        this.island = island;
    }

    /**
     * @param meas measurement vector
     */
    public void getEstimatedZ(MeasVector meas) {
        AVector result = meas.getZ_estimate();
        int index = 0;
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                //to get the estimated voltage value
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        int num = meas.getBus_v_pos()[i];
                        int phase = meas.getBus_v_phase()[i];
                        double[] v = island.getBusV().get(island.getTnNoToTn().get(num))[phase];
                        if (island.isVCartesian())
                            result.setValue(index, v[0] * v[0] + v[1] * v[1]);//todo:
                        else
                            result.setValue(index, v[0]);
                    }
                    break;
                //to get the estimated bus active power
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        int phase = meas.getBus_p_phase()[i];
                        result.setValue(index, calBusPQ(island.getTnNoToTn().get(num), phase)[0]);
                    }
                    break;
                //to get the estimated bus reactive power
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        int num = meas.getBus_q_pos()[i];
                        int phase = meas.getBus_q_phase()[i];
                        result.setValue(index, calBusPQ(island.getTnNoToTn().get(num), phase)[1]);
                    }
                    break;
                //to get the estimated line active power from the bus
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int num = meas.getLine_from_p_pos()[k];
                        int phase = meas.getLine_from_p_phase()[k];
                        MapObject obj = island.getIdToBranch().get(num);
                        result.setValue(index, calLinePQFrom(obj, phase)[0]);
                    }
                    break;
                //to get the estimated line reactive power from the bus
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        int num = meas.getLine_from_q_pos()[k];
                        int phase = meas.getLine_from_q_phase()[k];
                        MapObject obj = island.getIdToBranch().get(num);
                        result.setValue(index, calLinePQFrom(obj, phase)[1]);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int num = meas.getLine_to_p_pos()[k];
                        int phase = meas.getLine_to_p_phase()[k];
                        MapObject obj = island.getIdToBranch().get(num);
                        result.setValue(index, calLinePQTo(obj, phase)[0]);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        int num = meas.getLine_to_q_pos()[k];
                        int phase = meas.getLine_to_q_phase()[k];
                        MapObject obj = island.getIdToBranch().get(num);
                        result.setValue(index, calLinePQTo(obj, phase)[1]);
                    }
                    break;
                //to get the estimated branch current amplitude
                case TYPE_LINE_FROM_CURRENT:
                    for (int k = 0; k < meas.getLine_from_i_amp_pos().length; k++, index++) {
                        Integer num = meas.getLine_from_i_amp_pos()[k];//num starts from 1
                        int phase = meas.getLine_from_i_amp_phase()[k];//num starts from 1
                        MapObject obj = island.getIdToBranch().get(num);
                        double[] c = island.getBranchHeadI().get(obj)[phase];
                        if (island.isICartesian()) {
                            result.setValue(index, c[0] * c[0] + c[1] * c[1]);//todo:
                        } else
                            result.setValue(index, c[0]);
                    }
                    break;
                case TYPE_LINE_TO_CURRENT:
                    for (int k = 0; k < meas.getLine_to_i_amp_pos().length; k++, index++) {
                        Integer num = meas.getLine_to_i_amp_pos()[k];//num starts from 1
                        int phase = meas.getLine_to_i_amp_phase()[k];//num starts from 1
                        MapObject obj = island.getIdToBranch().get(num);
                        double[] c = island.getBranchTailI().get(obj)[phase];
                        if (island.isICartesian()) {
                            result.setValue(index, c[0] * c[0] + c[1] * c[1]);//todo:
                        } else
                            result.setValue(index, c[0]);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    public double[] calBusPQ(DsTopoNode tn, int phase) {
        double[] value = new double[2];
        for (int b : tn.getConnectedBusNo()) {//branch number = tail bus's number - 1
            DsTopoNode tn2 = tn.getIsland().getTnNoToTn().get(b);
            MapObject edge = tn.getIsland().getGraph().getEdge(tn, tn2);
            if (tn.getTnNo() < b) {
                double[] doubles = calLinePQFrom(edge, phase);
                value[0] += doubles[0];
                value[1] += doubles[1];
            } else {
                double[] doubles = calLinePQTo(edge, phase);
                value[0] += doubles[0];
                value[1] += doubles[1];
            }
        }
        return value;
    }

    /**
     * to calculate the line active and reactive power from the bus
     *
     * @param pos   branch's id in radical island
     * @param phase phase
     * @return p and q on branch's head terminal
     */
    public double[] calLinePQFrom(MapObject pos, int phase) {
        DsTopoNode tn1 = island.getGraph().getEdgeSource(pos);
        DsTopoNode tn2 = island.getGraph().getEdgeTarget(pos);
        DsTopoNode tn = tn1.getTnNo() < tn2.getTnNo() ? tn1 : tn2;
        double[] v = island.getBusV().get(tn)[phase];
        if (!island.isVCartesian()) {
            MathUtil.trans_polar2rect(v, tmpV);
            v = tmpV;
        }
        double[][] i = island.getBranchHeadI().get(pos);
        if (!island.isICartesian()) {
            tmpI[0] = i[phase][0] * Math.cos(i[phase][1]);
            tmpI[1] = i[phase][0] * Math.sin(i[phase][1]);
        } else {
            tmpI[0] = i[phase][0];
            tmpI[1] = i[phase][1];
        }
        return new double[]{v[0] * tmpI[0] + v[1] * tmpI[1], -v[0] * tmpI[1] + v[1] * tmpI[0]};
    }

    /**
     * to calculate the line active and reactive power to the bus
     *
     * @param pos   branch's id in radical island
     * @param phase phase
     * @return p and q on branch's tail terminal
     */
    public double[] calLinePQTo(MapObject pos, int phase) {
        DsTopoNode tn1 = island.getGraph().getEdgeSource(pos);
        DsTopoNode tn2 = island.getGraph().getEdgeTarget(pos);
        DsTopoNode tn = tn1.getTnNo() > tn2.getTnNo() ? tn1 : tn2;
        double[] v = island.getBusV().get(tn)[phase];
        if (!island.isVCartesian()) {
            MathUtil.trans_polar2rect(v, tmpV);
            v = tmpV;
        }
        double[][] i = island.getBranchTailI().get(pos);
        if (!island.isICartesian()) {
            tmpI[0] = -i[phase][0] * Math.cos(i[phase][1]);
            tmpI[1] = -i[phase][0] * Math.sin(i[phase][1]);
        } else {
            tmpI[0] = -i[phase][0];
            tmpI[1] = -i[phase][1];
        }
        return new double[]{v[0] * tmpI[0] + v[1] * tmpI[1], -v[0] * tmpI[1] + v[1] * tmpI[0]};
    }

    /**
     * @param injectionPQ p and q injection at bus
     * @param v           voltage at bus
     * @return injiection current
     */
    public double[] calBusInjectionI(double[] injectionPQ, double[] v) {
        double[] current = new double[2];
        current[0] = (injectionPQ[0] * v[0] + injectionPQ[1] * v[1]) / (v[0] * v[0] + v[1] * v[1]);
        current[1] = (-injectionPQ[1] * v[0] + injectionPQ[0] * v[1]) / (v[0] * v[0] + v[1] * v[1]);
        return current;
    }

    public void setIsland(DsTopoIsland island) {
        this.island = island;
    }
}
