package zju.dsntp;

import java.io.IOException;

/**
 * Created by meditation on 2017/12/14.
 */
public class main {
    public static void main(String[] args) throws Exception {
        PSOInTSC psoInTSC = new PSOInTSC();
        psoInTSC.initial(20,44,50);
        psoInTSC.run();
        psoInTSC.showResult();

//        PsoInRoad psoInRoad = new PsoInRoad();
//        psoInRoad.initial(50,35);
//        psoInRoad.run();
//        psoInRoad.showResult();

    }
}
