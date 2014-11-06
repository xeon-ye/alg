package zju.pf;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2008-8-9
 */
public class PfAlgFactory implements PfConstants {

    public static PolarPf createAlgorithm(String algName) {
        if (algName.equals(ALG_IPOPT))
            return new IpoptPf();
        else if (algName.equals(ALG_NEWTON))
            return new PolarPf();
        else
            return new PolarPf();
    }

    public static PolarPf createAlgorithm() {
        return new IpoptPf();
    }
}
