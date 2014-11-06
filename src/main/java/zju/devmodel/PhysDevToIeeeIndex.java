package zju.devmodel;

import zju.util.SerializableUtil;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 11-8-25
 */
public class PhysDevToIeeeIndex implements Serializable {
    private static final String ITEM_SEPARATOR = "%";

    private Map<String, Integer> winding2branch;
    private Map<String, Integer> acline2branch;
    private Map<Integer, String> branch2acline;

    private Map<Integer, List<String>> generators;
    private Map<Integer, List<String>> loads;
    private Map<Integer, List<String>> busbars;
    private Map<Integer, List<String>> shuntCompensators;

    public static PhysDevToIeeeIndex parse(String content) {
        PhysDevToIeeeIndex result = new PhysDevToIeeeIndex();
        String tem1;
        String tem2;
        int index1 = content.indexOf(ITEM_SEPARATOR);
        int index2 = content.indexOf(ITEM_SEPARATOR, index1 + 1);
        do {
            tem1 = content.substring(0, index1);
            tem2 = content.substring(index1 + ITEM_SEPARATOR.length(), index2);
            content = content.substring(index2 + ITEM_SEPARATOR.length());
            index1 = content.indexOf(ITEM_SEPARATOR);
            index2 = content.indexOf(ITEM_SEPARATOR, index1 + 1);
            try {
                if (tem1.equals("winding2branch")) {
                    result.setWinding2branch(SerializableUtil.createMap(tem2, SerializableUtil.TYPE_STRING, SerializableUtil.TYPE_INTEGER));
                } else if (tem1.equals("acline2branch")) {
                    result.setAcline2branch(SerializableUtil.createMap(tem2, SerializableUtil.TYPE_STRING, SerializableUtil.TYPE_INTEGER));
                } else if (tem1.equals("branch2acline")) {
                    result.setBranch2acline(SerializableUtil.createMap(tem2, SerializableUtil.TYPE_INTEGER, SerializableUtil.TYPE_STRING));
                } else if (tem1.equals("generators")) {
                    result.setGenerators(SerializableUtil.createMap(tem2, SerializableUtil.TYPE_INTEGER, SerializableUtil.TYPE_STRING_LIST));
                } else if (tem1.equals("loads")) {
                    result.setLoads(SerializableUtil.createMap(tem2, SerializableUtil.TYPE_INTEGER, SerializableUtil.TYPE_STRING_LIST));
                } else if (tem1.equals("busbars")) {
                    result.setBusbars(SerializableUtil.createMap(tem2, SerializableUtil.TYPE_INTEGER, SerializableUtil.TYPE_STRING_LIST));
                } else if (tem1.equals("shuntCompensators")) {
                    result.setShuntCompensators(SerializableUtil.createMap(tem2, SerializableUtil.TYPE_INTEGER, SerializableUtil.TYPE_STRING_LIST));
                }
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        } while (index1 != -1);
        return result;
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();

        builder.append("winding2branch").append(ITEM_SEPARATOR);
        for (String key : winding2branch.keySet())
            builder.append(key).append(SerializableUtil.PROP_SEPARATOR).append(winding2branch.get(key)).append(SerializableUtil.ENTITY_SEPARATOR);
        builder.append(ITEM_SEPARATOR);

        builder.append("acline2branch").append(ITEM_SEPARATOR);
        for (String key : acline2branch.keySet())
            builder.append(key).append(SerializableUtil.PROP_SEPARATOR).append(acline2branch.get(key)).append(SerializableUtil.ENTITY_SEPARATOR);
        builder.append(ITEM_SEPARATOR);

        builder.append("branch2acline").append(ITEM_SEPARATOR);
        for (Integer key : branch2acline.keySet())
            builder.append(key).append(SerializableUtil.PROP_SEPARATOR).append(branch2acline.get(key)).append(SerializableUtil.ENTITY_SEPARATOR);
        builder.append(ITEM_SEPARATOR);

        builder.append("generators").append(ITEM_SEPARATOR);
        for (Integer key : generators.keySet()) {
            List<String> list = generators.get(key);
            builder.append(key).append(SerializableUtil.PROP_SEPARATOR).append(SerializableUtil.formString(list)).append(SerializableUtil.ENTITY_SEPARATOR);
        }
        builder.append(ITEM_SEPARATOR);

        builder.append("loads").append(ITEM_SEPARATOR);
        for (Integer key : loads.keySet()) {
            List<String> list = loads.get(key);
            builder.append(key).append(SerializableUtil.PROP_SEPARATOR).append(SerializableUtil.formString(list)).append(SerializableUtil.ENTITY_SEPARATOR);
        }
        builder.append(ITEM_SEPARATOR);

        builder.append("busbars").append(ITEM_SEPARATOR);
        for (Integer key : busbars.keySet()) {
            List<String> list = busbars.get(key);
            builder.append(key).append(SerializableUtil.PROP_SEPARATOR).append(SerializableUtil.formString(list)).append(SerializableUtil.ENTITY_SEPARATOR);
        }
        builder.append(ITEM_SEPARATOR);

        builder.append("shuntCompensators").append(ITEM_SEPARATOR);
        for (Integer key : shuntCompensators.keySet()) {
            List<String> list = shuntCompensators.get(key);
            builder.append(key).append(SerializableUtil.PROP_SEPARATOR).append(SerializableUtil.formString(list)).append(SerializableUtil.ENTITY_SEPARATOR);
        }
        builder.append(ITEM_SEPARATOR);
        return builder.toString();
    }

    public Map<String, Integer> getWinding2branch() {
        return winding2branch;
    }

    public void setWinding2branch(Map<String, Integer> winding2branch) {
        this.winding2branch = winding2branch;
    }

    public Map<String, Integer> getAcline2branch() {
        return acline2branch;
    }

    public void setAcline2branch(Map<String, Integer> acline2branch) {
        this.acline2branch = acline2branch;
    }

    public Map<Integer, String> getBranch2acline() {
        return branch2acline;
    }

    public void setBranch2acline(Map<Integer, String> branch2acline) {
        this.branch2acline = branch2acline;
    }

    public Map<Integer, List<String>> getGenerators() {
        return generators;
    }

    public void setGenerators(Map<Integer, List<String>> generators) {
        this.generators = generators;
    }

    public Map<Integer, List<String>> getLoads() {
        return loads;
    }

    public void setLoads(Map<Integer, List<String>> loads) {
        this.loads = loads;
    }

    public Map<Integer, List<String>> getBusbars() {
        return busbars;
    }

    public void setBusbars(Map<Integer, List<String>> busbars) {
        this.busbars = busbars;
    }

    public Map<Integer, List<String>> getShuntCompensators() {
        return shuntCompensators;
    }

    public void setShuntCompensators(Map<Integer, List<String>> shuntCompensators) {
        this.shuntCompensators = shuntCompensators;
    }
}


