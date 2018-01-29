package zju.dsntp;

import java.io.IOException;

/**
 * Created by meditation on 2017/12/14.
 */
public class main {
    public static void main(String[] args) throws IOException {
//        PSOInTSC psoInTSC = new PSOInTSC();
//        psoInTSC.initial();
//        psoInTSC.run();
//        psoInTSC.showResult();

        PsoInRoad psoInRoad = new PsoInRoad();
        psoInRoad.initial(20,20);
        psoInRoad.run();
        psoInRoad.showResult();

    }
}
