package zju.ieeeformat;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-11-13
 */
public class KVLevelPicker {
    public static final int KV_3 = 0;
    public static final int KV_6 = 1;
    public static final int KV_10 = 2;
    public static final int KV_13 = 3;
    public static final int KV_15 = 4;
    public static final int KV_18 = 5;
    public static final int KV_20 = 6;
    public static final int KV_22 = 7;
    public static final int KV_24 = 8;
    public static final int KV_35 = 9;
    public static final int KV_60 = 10;
    public static final int KV_110 = 11;
    public static final int KV_220 = 12;
    public static final int KV_330 = 13;
    public static final int KV_500 = 14;
    public static final int KV_1050 = 15;

    /**
     * return voltage level, -1 will be returned if intput base voltage is not belong to any voltage level.
     *
     * @param baseV base voltage
     * @return votage level
     */
    public static int getKVLevel(double baseV) {
        if (baseV > 2.0 && baseV < 4.0)
            return KV_3;
        if (baseV > 5.0 && baseV < 7.0)
            return KV_6;
        if (baseV > 9.0 && baseV < 12.0)
            return KV_10;
        if (baseV > 12.0 && baseV < 14.0)
            return KV_13;
        if (baseV > 14.0 && baseV < 16.0)
            return KV_15;
        if (baseV > 17.0 && baseV < 19.0)
            return KV_18;
        if (baseV > 19.0 && baseV < 21.0)
            return KV_20;
        if (baseV > 21.0 && baseV < 23.0)
            return KV_22;
        if (baseV > 23.0 && baseV < 25.0)
            return KV_24;
        if (baseV > 34.0 && baseV < 40.0)
            return KV_35;
        if (baseV > 50.0 && baseV < 70.0)
            return KV_60;
        if (baseV > 100.0 && baseV < 130.0)
            return KV_110;
        if (baseV > 200.0 && baseV < 250.0)
            return KV_220;
        if (baseV > 400.0 && baseV < 550.0)
            return KV_500;
        if (baseV > 1000. && baseV < 1500.)
            return KV_1050;
        return -1;
    }

}
