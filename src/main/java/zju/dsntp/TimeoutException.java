package zju.dsntp;

/**
 * Created by meditation on 2017/12/18.
 */
public class TimeoutException extends RuntimeException {
    //εΊεεε·
    private static final long serialVersionUID = -8078853655388692688L;

    public TimeoutException(String errMessage) {
        super(errMessage);
    }
}