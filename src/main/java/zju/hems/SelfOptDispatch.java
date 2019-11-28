package zju.hems;

import java.util.Map;

public class SelfOptDispatch implements DispatchOpt {

    @Override
    public Map<String, UserResult> doDispatchOpt(Microgrid microgrid, double dispatchTime, int periodNum,
                                                 double[] elecPrices, double[] gasPrices, double[] steamPrices) {
        DispatchOptModel dispatchOptModel = new DispatchOptModel(microgrid, dispatchTime, periodNum,
                elecPrices, gasPrices, steamPrices);
        dispatchOptModel.mgDispatchOpt();
        return dispatchOptModel.getMicrogridResult();
    }
}
