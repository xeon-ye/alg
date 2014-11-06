package zju.ieeeformat;

import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * Class BranchData
 * <p> branch data in ieee common data format</P>
 *
 * @author Dong Shufeng
 * @version 1.0
 * @Date: 2006-7-16
 */
public class BranchData implements Serializable, Comparable, Cloneable {
    private static final long serialVersionUID = -3198863102114791691L;
    private int id;

    public static final int BRANCH_TYPE_ACLINE = 0; //0 - transmission line
    public static final int BRANCH_TYPE_TF_FIXED_TAP = 1; //1 - fixed tap
    public static final int BRANCH_TYPE_TF_V_CTRL = 2; //2 - variable tap for voltage control(TCUL, LTC)
    public static final int BRANCH_TYPE_TF_MVAR_CTRL = 3; //3 - variable tap(turns ratio) for MVAR control
    public static final int BRANCH_TYPE_PHASE_SHIFTER = 4; //4 - variable phase angle for MW control(phase shifter)

    //this valiable is used for compare to another bus
    private static String[] compareOrder;
    public static final String VAR_TAP_BUS_NUMBER = "TapBusNumber";
    public static final String VAR_Z_BUS_NUMBER = "ZBusNumber";
    public static final String VAR_TYPE = "Type";
    public static final String VAR_R = "BranchR";
    public static final String VAR_X = "BranchX";

    private int tapBusNumber; //for transformers or phase shifters, the side of the model the non-unity tap is on
    private int zBusNumber; //for transformers and phase shifters, the side of the model the device impedance is on
    private int area;
    private int lossZone;
    private int circuit; //use 1 for single lines
    /**
     * 0 - transmission line
     * 1 - fixed tap
     * 2 - variable tap for voltage control(TCUL, LTC)
     * 3 - variable tap(turns ratio) for MVAR control
     * 4 - variable phase angle for MW control(phase shifter)
     */
    private int type;
    private double branchR; //resistance R, per unit
    private double branchX; //reactance X, per unit, no zero impedance lines
    private double lineB; //line charging B, per unit(total line charging, +B)
    private int mvaRating1;
    private int mvaRating2;
    private int mvaRating3;
    private int controlBusNumber;
    /**
     * 0 - controlled bus is one of the termials
     * 1 - controlled bus is near the tap side
     * 2 - controlled bus is near the impedance side (Z bus)
     */
    private int side;
    private double transformerRatio;
    private double transformerAngle;
    private double minimumTap; //minimum tap of phase shift
    private double maximumTap; //maximum tap or phase shift
    private double stepSize;
    private double minimum; //minimum voltage, MVAR or MW limit
    private double maximum; //maximum voltage, MVAR or MW limit

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public int getTapBusNumber() {
        return tapBusNumber;
    }

    public void setTapBusNumber(int tapBusNumber) {
        this.tapBusNumber = tapBusNumber;
    }

    public int getZBusNumber() {
        return zBusNumber;
    }

    public void setZBusNumber(int zBusNumber) {
        this.zBusNumber = zBusNumber;
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

    public int getCircuit() {
        return circuit;
    }

    public void setCircuit(int circuit) {
        this.circuit = circuit;
    }

    /**
     * @return 0 - transmission line <br>
     *         1 - fixed tap <br>
     *         2 - variable tap for voltage control(TCUL, LTC) <br>
     *         3 - variable tap(turns ratio) for MVAR control <br>
     *         4 - variable phase angle for MW control(phase shifter) <br>
     */
    public int getType() {
        return type;
    }

    /**
     * @param type 0 - transmission line <br>
     *             1 - fixed tap <br>
     *             2 - variable tap for voltage control(TCUL, LTC) <br>
     *             3 - variable tap(turns ratio) for MVAR control <br>
     *             4 - variable phase angle for MW control(phase shifter) <br>
     */
    public void setType(int type) {
        this.type = type;
    }

    public double getBranchR() {
        return branchR;
    }

    public void setBranchR(double branchR) {
        this.branchR = branchR;
    }

    public double getBranchX() {
        return branchX;
    }

    public void setBranchX(double branchX) {
        this.branchX = branchX;
    }

    public double getLineB() {
        return lineB;
    }

    public void setLineB(double lineB) {
        this.lineB = lineB;
    }

    public int getMvaRating1() {
        return mvaRating1;
    }

    public void setMvaRating1(int mvaRating1) {
        this.mvaRating1 = mvaRating1;
    }

    public int getMvaRating2() {
        return mvaRating2;
    }

    public void setMvaRating2(int mvaRating2) {
        this.mvaRating2 = mvaRating2;
    }

    public int getMvaRating3() {
        return mvaRating3;
    }

    public void setMvaRating3(int mvaRating3) {
        this.mvaRating3 = mvaRating3;
    }

    public int getControlBusNumber() {
        return controlBusNumber;
    }

    public void setControlBusNumber(int controlBusNumber) {
        this.controlBusNumber = controlBusNumber;
    }

    /**
     * @return 0 - controlled bus is one of the termials<Br>
     *         1 - controlled bus is near the tap side<Br>
     *         2 - controlled bus is near the impedance side (Z bus)
     */
    public int getSide() {
        return side;
    }

    /**
     * @param side 0 - controlled bus is one of the termials<Br>
     *             1 - controlled bus is near the tap side<Br>
     *             2 - controlled bus is near the impedance side (Z bus)
     */
    public void setSide(int side) {
        this.side = side;
    }

    public double getTransformerRatio() {
        return transformerRatio;
    }

    public void setTransformerRatio(double transformerRatio) {
        this.transformerRatio = transformerRatio;
    }

    public double getTransformerAngle() {
        return transformerAngle;
    }

    public void setTransformerAngle(double transformerAngle) {
        this.transformerAngle = transformerAngle;
    }

    public double getMinimumTap() {
        return minimumTap;
    }

    public void setMinimumTap(double minimumTap) {
        this.minimumTap = minimumTap;
    }

    public double getMaximumTap() {
        return maximumTap;
    }

    public void setMaximumTap(double maximumTap) {
        this.maximumTap = maximumTap;
    }

    public double getStepSize() {
        return stepSize;
    }

    public void setStepSize(double stepSize) {
        this.stepSize = stepSize;
    }

    /**
     * @return minimum voltage, MVAR or MW limit
     */
    public double getMinimum() {
        return minimum;
    }

    /**
     * @param minimum voltage, MVAR or MW limit
     */
    public void setMinimum(double minimum) {
        this.minimum = minimum;
    }

    /**
     * @return maximum voltage, MVAR or MW limit
     */
    public double getMaximum() {
        return maximum;
    }

    /**
     * @param maximum maximum voltage, MVAR or MW limit
     */
    public void setMaximum(double maximum) {
        this.maximum = maximum;
    }

    public static String[] getCompareOrder() {
        return compareOrder;
    }

    public static void setCompareOrder(String[] compareOrder) {
        BranchData.compareOrder = compareOrder;
    }

    /**
     * Two branches are equal when and only when branch's tap bus number, z bus number and circuit number are equal
     *
     * @param obj obj to compare with
     * @return is equal
     */
    public boolean equals(Object obj) {
        if (obj instanceof BranchData) {
            BranchData b = (BranchData) obj;
            return b.getTapBusNumber() == this.getTapBusNumber() && b.getZBusNumber() == this.getZBusNumber() && b.getCircuit() == this.getCircuit();
        }
        return false;
    }

    @SuppressWarnings({"EmptyCatchBlock"})
    public int compareTo(Object anotherBranch) {
        if (compareOrder != null && anotherBranch instanceof BranchData) {
            for (String variableName : compareOrder) {
                try {
                    Method method = this.getClass().getMethod("get" + variableName);
                    Object value1 = method.invoke(this);
                    Object value2 = method.invoke(anotherBranch);
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
    public BranchData clone() {
        try {
            return (BranchData) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return null;
    }
}
