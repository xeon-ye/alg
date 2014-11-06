package zju.bpamodel.pf;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-11-8
 */
public class PfCtrlInfo implements Serializable {
    public static final String MVA_BASE = "MVA_BASE";
    public static final Double DEFAULT_MVA_BASE = 100.0;
    private Map<String, Object> ctrlProperties = new HashMap<String, Object>();

    public PfCtrlInfo() {
        ctrlProperties.put(MVA_BASE, DEFAULT_MVA_BASE);
    }

    public void addCtrlInfo(String toParse) {
        if (toParse.startsWith("/") && toParse.endsWith("\\")) {
            //todo:
        }
    }

    public Object getProperty(String key) {
        return ctrlProperties.get(key);
    }

    public Map<String, Object> getCtrlProperties() {
        return ctrlProperties;
    }

    public void setCtrlProperties(Map<String, Object> ctrlProperties) {
        this.ctrlProperties = ctrlProperties;
    }
}
