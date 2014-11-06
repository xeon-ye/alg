package zju.dsmodel;

import zju.devmodel.MapObject;

import java.io.Serializable;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-10
 */
public class DsConnectNode implements Serializable, Comparable {
    private String id;

    private List<MapObject> connectedObjs;

    private Double baseKv;

    public DsConnectNode() {
    }

    public DsConnectNode(String id) {
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public List<MapObject> getConnectedObjs() {
        return connectedObjs;
    }

    public void setConnectedObjs(List<MapObject> connectedObjs) {
        this.connectedObjs = connectedObjs;
    }

    public Double getBaseKv() {
        return baseKv;
    }

    public void setBaseKv(Double baseKv) {
        this.baseKv = baseKv;
    }

    public int compareTo(Object o) {
        if (!(o instanceof DsConnectNode))
            return 0;
        DsConnectNode n = (DsConnectNode) o;
        return id.compareTo(n.getId());
    }
}
