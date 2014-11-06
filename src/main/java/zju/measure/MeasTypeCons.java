package zju.measure;

/**
 * defining constant valiables of measurement type
 * eight types are defined:<br>
 * 0 Bus angle measure<br>
 * 1 Bus voltage measure<br>
 * 2 Bus injection active power measure<br>
 * 3 Bus injection inactive power measure<br>
 * 4 Line's head active power measure<br>
 * 5 Line's tail active power measure<br>
 * 6 Line's head inactive power measure <br>
 * 7 Line's tail inactive power measure<br>
 *
 * @author Dong Shufeng
 *         Date: 2007-12-11
 */
public interface MeasTypeCons {

    //模拟量量测类型定义
    public final int TYPE_BUS_ANGLE = 100;
    public final int TYPE_BUS_VOLOTAGE = 101;
    public final int TYPE_BUS_ACTIVE_POWER = 102;
    public final int TYPE_BUS_REACTIVE_POWER = 103;

    public final int TYPE_LINE_FROM_ACTIVE = 104;
    public final int TYPE_LINE_FROM_REACTIVE = 105;
    public final int TYPE_LINE_TO_ACTIVE = 106;
    public final int TYPE_LINE_TO_REACTIVE = 107;

    public final int TYPE_LINE_CURRENT = 108;
    public final int TYPE_LINE_CURRENT_ANGLE = 109;
    public final int TYPE_LINE_FROM_CURRENT = 110;
    public final int TYPE_LINE_FROM_CURRENT_ANGLE = 111;
    public final int TYPE_LINE_TO_CURRENT = 112;
    public final int TYPE_LINE_TO_CURRENT_ANGLE = 113;

    //离散量量测类型定义
    public final int TYPE_SWITCH_POS = 200;
    public final int TYPE_TAP_POS = 201;

    public static final int LINE_POWER_FROM = 8;
    public static final int LINE_POWER_TO = 9;
    public static final int POWER_GEN = 10;
    public static final int POWER_LOAD = 11;
    public static final int POWER_COMPENSATOR = 41;

    public static final int GEN_ACTIVE = 12;
    public static final int GEN_REACTIVE = 13;
    public static final int LOAD_ACTIVE = 14;
    public static final int LOAD_REACTIVE = 15;
    public static final int XFMR_FROM_ACTIVE = 16;
    public static final int XFMR_FROM_REACTIVE = 17;
    public static final int XFMR_TO_ACTIVE = 18;
    public static final int XFMR_TO_REACTIVE = 19;
    public static final int LINE_FROM_ACTIVE = 20;
    public static final int LINE_FROM_REACTIVE = 21;
    public static final int LINE_TO_ACTIVE = 22;
    public static final int LINE_TO_REACTIVE = 23;

    public static final int[] DEFAULT_TYPES = {
            TYPE_BUS_ANGLE,
            TYPE_BUS_VOLOTAGE,
            TYPE_BUS_ACTIVE_POWER,
            TYPE_BUS_REACTIVE_POWER,
            TYPE_LINE_FROM_ACTIVE,
            TYPE_LINE_FROM_REACTIVE,
            TYPE_LINE_TO_ACTIVE,
            TYPE_LINE_TO_REACTIVE,
            TYPE_LINE_CURRENT,
            TYPE_LINE_CURRENT_ANGLE,
            TYPE_LINE_FROM_CURRENT,
            TYPE_LINE_FROM_CURRENT_ANGLE,
            TYPE_LINE_TO_CURRENT,
            TYPE_LINE_TO_CURRENT_ANGLE
    };

    public static final int[] BUS_TYPE = {
            TYPE_BUS_ANGLE,
            TYPE_BUS_VOLOTAGE,
            TYPE_BUS_ACTIVE_POWER,
            TYPE_BUS_REACTIVE_POWER,
    };

    public static final int[] BRANCH_TYPE = {
            TYPE_LINE_FROM_ACTIVE,
            TYPE_LINE_FROM_REACTIVE,
            TYPE_LINE_TO_ACTIVE,
            TYPE_LINE_TO_REACTIVE,
    };
}
