package zju.ieeeformat;

import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * Class BusData
 * <p> bus data in ieee common data format</P>
 * Copyright (c) Dong Shufeng
 *
 * @author Dong Shufeng
 * @version 1.0
 * @Date: 2006-7-16
 */
public class BusData implements Comparable, Serializable, Cloneable {
    private static final long serialVersionUID = -3198863102119374691L;

    public static final int BUS_TYPE_LOAD_PQ = 0;  //0 - unregulated(load, PQ)
    public static final int BUS_TYPE_GEN_PQ = 1;        //1 - hold MVAR generation within voltage limits,(gen, PQ)
    public static final int BUS_TYPE_GEN_PV = 2;  //2 - hold voltage within VAR limits(gen, PV)
    public static final int BUS_TYPE_SLACK = 3;  //3 -  hold voltage and angle(swing, V-Theta; must always have one)

    //this valiable is used for compare to another bus
    private static String[] compareOrder;
    public static final String VAR_NAME = "Name";
    public static final String VAR_NUMBER = "BusNumber";
    public static final String VAR_TYPE = "Type";
    public static final String VAR_PGEN = "GenerationMW";
    public static final String VAR_QGEN = "GenerationMVAR";
    public static final String VAR_PLOAD = "LoadMW";
    public static final String VAR_QLOAD = "LoadMVAR";

    private int busNumber;
    private String name = "";
    private int area; //don't use zone!
    private int lossZone;
    /**
     * 0 - unregulated(load, PQ)
     * 1 - hold MVAR generation within voltage limits,(gen, PQ)
     * 2 - hould voltage within VAR limits(gen, PV)
     * 3 - hould voltage and angle(swing, V-Theta; must always have one)
     */
    private int type;
    private double finalVoltage; //p.u
    private double finalAngle; //degrees
    private double loadMW;
    private double loadMVAR;
    private double generationMW;
    private double generationMVAR;
    private double baseVoltage; //KV
    private double desiredVolt; //this is desired remote voltage if this bus is controlling another bus
    private double maximum; //maximum MVAR or voltage limit
    private double minimum; //minimum MVAR or voltage limit
    private double shuntConductance; //G per unit
    private double shuntSusceptance; //B per unit
    private int remoteControlBusNumber;

    public int getBusNumber() {
        return busNumber;
    }

    public void setBusNumber(int busNumber) {
        this.busNumber = busNumber;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getArea() {
        return area;
    }

    public void setArea(int area) {
        this.area = area;
    }

    public int getLossZone() {
        return lossZone;
    }

    public void setLossZone(int lossZone) {
        this.lossZone = lossZone;
    }

    /**
     * @return 0 - unregulated(load, PQ) <br>
     *         1 - hold MVAR generation within voltage limits,(gen, PQ) <br>
     *         2 - hould voltage within VAR limits(gen, PV) <br>
     *         3 - hould voltage and angle(swing, V-Theta; must always have one) <br>
     */
    public int getType() {
        return type;
    }

    /**
     * @param type 0 - unregulated(load, PQ) <br>
     *             1 - hold MVAR generation within voltage limits,(gen, PQ) <br>
     *             2 - hould voltage within VAR limits(gen, PV) <br>
     *             3 - hould voltage and angle(swing, V-Theta; must always have one) <br>
     */
    public void setType(int type) {
        this.type = type;
    }

    /**
     * @return p.u
     */
    public double getFinalVoltage() {
        return finalVoltage;
    }

    /**
     * @param finalVoltage p.u
     */
    public void setFinalVoltage(double finalVoltage) {
        this.finalVoltage = finalVoltage;
    }

    /**
     * @return degrees
     */
    public double getFinalAngle() {
        return finalAngle;
    }

    /**
     * @param finalAngle degrees
     */
    public void setFinalAngle(double finalAngle) {
        this.finalAngle = finalAngle;
    }

    public double getLoadMW() {
        return loadMW;
    }

    public void setLoadMW(double loadMW) {
        this.loadMW = loadMW;
    }

    public double getLoadMVAR() {
        return loadMVAR;
    }

    public void setLoadMVAR(double loadMVAR) {
        this.loadMVAR = loadMVAR;
    }

    public double getGenerationMW() {
        return generationMW;
    }

    public void setGenerationMW(double generationMW) {
        this.generationMW = generationMW;
    }

    public double getGenerationMVAR() {
        return generationMVAR;
    }

    public void setGenerationMVAR(double generationMVAR) {
        this.generationMVAR = generationMVAR;
    }

    /**
     * @return KV
     */
    public double getBaseVoltage() {
        return baseVoltage;
    }

    /**
     * @param baseVoltage KV
     */
    public void setBaseVoltage(double baseVoltage) {
        this.baseVoltage = baseVoltage;
    }

    public double getDesiredVolt() {
        return desiredVolt;
    }

    public void setDesiredVolt(double desiredVolt) {
        this.desiredVolt = desiredVolt;
    }

    /**
     * @return maximum MVAR or voltage limit
     */
    public double getMaximum() {
        return maximum;
    }

    /**
     * @param maximum maximum MVAR or voltage limit
     */
    public void setMaximum(double maximum) {
        this.maximum = maximum;
    }

    /**
     * @return minimum MVAR or voltage limit
     */
    public double getMinimum() {
        return minimum;
    }

    /**
     * @param minimum minimum MVAR or voltage limit
     */
    public void setMinimum(double minimum) {
        this.minimum = minimum;
    }

    /**
     * @return G per unit
     */
    public double getShuntConductance() {
        return shuntConductance;
    }

    /**
     * @param shuntConductance G per unit
     */
    public void setShuntConductance(double shuntConductance) {
        this.shuntConductance = shuntConductance;
    }

    /**
     * @return B per unit
     */
    public double getShuntSusceptance() {
        return shuntSusceptance;
    }

    /**
     * @param shuntSusceptance B per unit
     */
    public void setShuntSusceptance(double shuntSusceptance) {
        this.shuntSusceptance = shuntSusceptance;
    }

    public int getRemoteControlBusNumber() {
        return remoteControlBusNumber;
    }

    public void setRemoteControlBusNumber(int remoteControlBusNumber) {
        this.remoteControlBusNumber = remoteControlBusNumber;
    }

    /**
     * @return compare order
     */
    public static String[] getCompareOrder() {
        return compareOrder;
    }

    /**
     * @param compareOrder field valiables' order is list in the parameter
     */
    public static void setCompareOrder(String[] compareOrder) {
        BusData.compareOrder = compareOrder;
    }

    /**
     * compare to another bus according static variable compareOrder<br>
     *
     * @param anotherBus object to compare with
     * @return a negative integer, zero, or a positive integer as this object
     *         is less than, equal to, or greater than the specified object
     */
    @SuppressWarnings({"EmptyCatchBlock"})
    public int compareTo(Object anotherBus) {
        if (compareOrder != null && anotherBus instanceof BusData) {
            for (String variableName : compareOrder) {
                try {
                    Method method = this.getClass().getMethod("get" + variableName);
                    Object value1 = method.invoke(this);
                    Object value2 = method.invoke(anotherBus);
                    int result = 0;
                    if (value1 instanceof String) {
                        result = ((String) value1).compareTo((String) value2);
                    } else if (value1 instanceof Integer) {
                        result = ((Integer) value1).compareTo((Integer) value2);
                    } else if (value1 instanceof Double) {
                        result = ((Double) value1).compareTo((Double) value2);
                    }
                    if (result == 0)
                        continue;
                    return result;
                } catch (IllegalAccessException e) {
                } catch (NoSuchMethodException e) {
                } catch (InvocationTargetException e) {
                }
            }
        }
        return 0;
    }


    @SuppressWarnings({"CloneDoesntDeclareCloneNotSupportedException"})
    public BusData clone() {
        try {
            return (BusData) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return null;
    }
}
