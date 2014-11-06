package zju.lfp.utils;

import zju.lfp.forecasters.analogyExtrapolation.IAnalogyExtrapolationInput;
import zju.lfp.forecasters.singleDimensionalForecaster.arma.IArmaInput;
import zju.lfp.forecasters.singleDimensionalForecaster.extrapolation.IExtrapolationInput;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-9-25
 * Time: 14:43:59
 */
public class DataServiceFactory {
    private static String appName;

    static {
        appName = ConfigUtil.getApplicationName();
    }

    public static DataService getService(String forecasterName) throws Exception {
        // todo
        return (DataService) Class.forName("zju.lfp.engine." + appName + "." + forecasterName + "ForecasterDataServiceImpl").newInstance();
    }

    public static IExtrapolationInput getExtrapolationService() throws Exception {
        return (IExtrapolationInput) Class.forName("zju.lfp.engine." + appName + ".ExtrapolationInputImpl").newInstance();
    }

    public static IArmaInput getArmaService() throws Exception {
        return (IArmaInput) Class.forName("zju.lfp.engine." + appName + ".ArmaInputImpl").newInstance();
    }

    public static IAnalogyExtrapolationInput getAnalogyExtrapolationService() throws Exception {
        return (IAnalogyExtrapolationInput) Class.forName("zju.lfp.engine." + appName + ".AnalogyExtrapolationInputImpl").newInstance();
    }
}
