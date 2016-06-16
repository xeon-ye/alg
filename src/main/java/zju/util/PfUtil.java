package zju.util;

import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.AVector;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2009-3-22
 */
public class PfUtil implements MeasTypeCons {

    public static MeasVector formMeasVector(IEEEDataIsland island, int[] ignorePBus, int[] ignoreQBus, int[] ignoreVBus, int[] ignoreABus) {
        MeasVector meas = new MeasVector();
        Map<Integer, Double> pMeas = new HashMap<Integer, Double>(island.getPvBusSize() + island.getPqBusSize());
        Map<Integer, Double> qMeas = new HashMap<Integer, Double>(island.getPqBusSize());
        Map<Integer, Double> vMeas = new HashMap<Integer, Double>(island.getPvBusSize() + island.getSlackBusSize());
        Map<Integer, Double> aMeas = new HashMap<Integer, Double>(island.getSlackBusSize());

        double baseMVA = island.getTitle().getMvaBase();
        for (BusData bus : island.getBuses()) {
            int type = bus.getType();
            switch (type) {
                case BusData.BUS_TYPE_LOAD_PQ://
                case BusData.BUS_TYPE_GEN_PQ://
                    double pm = (bus.getGenerationMW() - bus.getLoadMW()) / baseMVA;
                    double qm = (bus.getGenerationMVAR() - bus.getLoadMVAR()) / baseMVA;
                    pMeas.put(bus.getBusNumber(), pm);
                    qMeas.put(bus.getBusNumber(), qm);
                    break;
                case BusData.BUS_TYPE_GEN_PV://
                    pm = (bus.getGenerationMW() - bus.getLoadMW()) / baseMVA;
                    pMeas.put(bus.getBusNumber(), pm);
                    if(bus.getDesiredVolt() > 0.5 && bus.getDesiredVolt() < 1.5)
                        vMeas.put(bus.getBusNumber(), bus.getDesiredVolt());
                    else
                        vMeas.put(bus.getBusNumber(), bus.getFinalVoltage());
                    break;
                case BusData.BUS_TYPE_SLACK://
                    vMeas.put(bus.getBusNumber(), bus.getFinalVoltage());
                    aMeas.put(bus.getBusNumber(), bus.getFinalAngle() * Math.PI / 180.0);
                    break;
                default://
                    break;
            }
        }

        for (int num : ignoreQBus)
            qMeas.remove(num);
        for (int num : ignorePBus)
            pMeas.remove(num);
        for (int num : ignoreVBus)
            vMeas.remove(num);
        for (int num : ignoreABus)
            aMeas.remove(num);

        int index;
        int indexZ = 0;
        AVector z = new AVector(pMeas.size() + qMeas.size() + vMeas.size() + aMeas.size());

        int[] a_pos = new int[aMeas.size()];
        index = 0;
        for (int busNumber : aMeas.keySet()) {
            a_pos[index] = busNumber;
            z.setValue(indexZ, aMeas.get(busNumber));
            index++;
            indexZ++;
        }

        int[] v_pos = new int[vMeas.size()];
        index = 0;
        for (int busNumber : vMeas.keySet()) {
            v_pos[index] = busNumber;
            z.setValue(indexZ, vMeas.get(busNumber));
            index++;
            indexZ++;
        }
        int[] q_pos = new int[qMeas.size()];
        index = 0;
        for (int busNumber : qMeas.keySet()) {
            q_pos[index] = busNumber;
            z.setValue(indexZ, qMeas.get(busNumber));
            index++;
            indexZ++;
        }
        int[] p_pos = new int[pMeas.size()];
        index = 0;
        for (int busNumber : pMeas.keySet()) {
            p_pos[index] = busNumber;
            z.setValue(indexZ, pMeas.get(busNumber));
            index++;
            indexZ++;
        }
        meas.setBus_a_pos(a_pos);
        meas.setBus_p_pos(p_pos);
        meas.setBus_q_pos(q_pos);
        meas.setBus_v_pos(v_pos);
        meas.setZ(z);
        meas.setMeasureOrder(new int[]{
                TYPE_BUS_ANGLE,
                TYPE_BUS_VOLOTAGE,
                TYPE_BUS_REACTIVE_POWER,
                TYPE_BUS_ACTIVE_POWER,
        });
        meas.setBus_a_index(0);
        meas.setBus_v_index(a_pos.length);
        meas.setBus_q_index(a_pos.length + v_pos.length);
        meas.setBus_p_index(a_pos.length + v_pos.length + q_pos.length);
        return meas;
    }

    public static MeasVector formMeasVector(IEEEDataIsland island) {
        return formMeasVector(island, new int[0], new int[0], new int[0], new int[0]);
    }

    /**
     * used in cpf
     *
     * @param island the target island to form
     * @return measure vector cotains every bus'(except slack bus) p and PQ bus' q measureinfo
     */
    public static MeasVector formPQMeasure(IEEEDataIsland island) {
        MeasVector meas = new MeasVector();
        Map<Integer, Double> pMeas = new HashMap<Integer, Double>(island.getPvBusSize() + island.getPqBusSize());
        Map<Integer, Double> qMeas = new HashMap<Integer, Double>(island.getPqBusSize());

        double baseMVA = island.getTitle().getMvaBase();
        for (BusData bus : island.getBuses()) {
            int type = bus.getType();
            switch (type) {
                case BusData.BUS_TYPE_LOAD_PQ://
                case BusData.BUS_TYPE_GEN_PQ://
                    double pm = (bus.getGenerationMW() - bus.getLoadMW()) / baseMVA;
                    double qm = (bus.getGenerationMVAR() - bus.getLoadMVAR()) / baseMVA;
                    pMeas.put(bus.getBusNumber(), pm);
                    qMeas.put(bus.getBusNumber(), qm);
                    break;
                case BusData.BUS_TYPE_GEN_PV://
                    pm = (bus.getGenerationMW() - bus.getLoadMW()) / baseMVA;
                    pMeas.put(bus.getBusNumber(), pm);
                    break;
                case BusData.BUS_TYPE_SLACK:
                    break;
                default://
                    break;
            }
        }

        int index;
        int indexZ = 0;
        AVector z = new AVector(pMeas.size() + qMeas.size());

        int[] q_pos = new int[qMeas.size()];
        index = 0;
        for (int i = 1; i <= island.getPqBusSize(); i++) {
            q_pos[index] = i;
            z.setValue(indexZ, qMeas.get(i));
            index++;
            indexZ++;
        }
        int[] p_pos = new int[pMeas.size()];
        index = 0;
        for (int i = 1; i <= island.getPvBusSize() + island.getPqBusSize(); i++) {
            p_pos[index] = i;
            z.setValue(indexZ, pMeas.get(i));
            index++;
            indexZ++;
        }
        meas.setBus_p_pos(p_pos);
        meas.setBus_q_pos(q_pos);
        meas.setZ(z);
        meas.setMeasureOrder(new int[]{
                TYPE_BUS_REACTIVE_POWER,
                TYPE_BUS_ACTIVE_POWER,
        });
        meas.setBus_q_index(0);
        meas.setBus_p_index(q_pos.length);
        return meas;
    }
}

