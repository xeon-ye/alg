package zju.devmodel;

import zju.util.SerializableUtil;

import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-2-10
 */
public class MapObject implements Serializable, Cloneable, Transferable {

    static final String TYPE_SEPARATOR = "-!-";
    public static final String KEY_ID = "id";
    private static final String KEY_NAME = "name";
    static final String KEY_CLASS = "class";
    private static final String KEY_TYPE = "type";
    private static final String KEY_CONTAINER = "container";

    private static int NEW_OBJ_COUNT = 1;

    private Map<String, String> properties = new HashMap<String, String>();

    private static String createID() {
        return System.currentTimeMillis() + "_" + NEW_OBJ_COUNT++;
    }

    public String getId() {
        return getProperty(KEY_ID);
    }

    public void setId(String id) {
        setProperty(KEY_ID, id);
    }

    public String getName() {
        return getProperty(KEY_NAME);
    }

    public void setName(String name) {
        setProperty(KEY_NAME, name);
    }

    public String getClassName() {
        return getProperty(KEY_CLASS);
    }

    public String getType() {
        return getProperty(KEY_TYPE);
    }

    public void setType(String type) {
        setProperty(KEY_TYPE, type);
    }

    public void setContainerId(String father) {
        setProperty(KEY_CONTAINER, father);
    }

    public String getContainerId() {
        return getProperty(KEY_CONTAINER);
    }

    public MapObject() {
        properties.put(KEY_ID, createID());
        properties.put(KEY_CLASS, this.getClass().getName());
    }

    public MapObject(String name) {
        this();
        properties.put(KEY_NAME, name);
    }

    public Map<String, String> getProperties() {
        return properties;
    }

    public void setProperties(Map<String, String> properties) {
        this.properties = properties;
    }

    public void setProperty(String key, String value) {
        this.properties.put(key, value);
    }

    public String getProperty(String key) {
        return this.properties.get(key);
    }

    public String toString() {
        return properties.get(KEY_NAME);
    }

    public Object clone() {
        try {
            MapObject o = this.getClass().newInstance();
            o.setProperties(new HashMap<String, String>(this.getProperties().size()));
            for (String key : this.getProperties().keySet())
                o.getProperties().put(key, this.getProperties().get(key));
            //MapObject o = (MapObject) super.clone();
            o.properties.put(KEY_ID, createID());
            return o;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public DataFlavor[] getTransferDataFlavors() {
        DataFlavor[] flavor = new DataFlavor[2];
        //get class
        Class clazz = this.getClass();
        String mimeType = "application/x-java-jvm-local-objectref;class=" + clazz.getName();

        try {
            flavor[0] = new DataFlavor(mimeType);
            flavor[1] = DataFlavor.stringFlavor;
        } catch (ClassNotFoundException e) {
            return null;
        }
        return flavor;
    }

    public boolean isDataFlavorSupported(DataFlavor flavor) {
        return flavor.equals(DataFlavor.stringFlavor) ||
                flavor.getPrimaryType().equals("application") &&
                        flavor.getSubType().equals("x-java-jvm-local-objectref") &&
                        flavor.getRepresentationClass().isAssignableFrom(this.getClass());
    }

    public Object getTransferData(DataFlavor flavor) throws UnsupportedFlavorException, IOException {
        if (!isDataFlavorSupported(flavor))
            throw new UnsupportedFlavorException(flavor);
        if (flavor.equals(DataFlavor.stringFlavor)) {
            StringBuffer buffer = new StringBuffer();
            for (String key : properties.keySet())
                buffer.append(key).append(SerializableUtil.PROP_SEPARATOR).append(properties.get(key)).append("\n");
            return buffer.toString();
        }
        return this;
    }

    public static MapObject createObj(String content) {
        MapObject result = new MapObject();
        Map m = SerializableUtil.createMap(content, SerializableUtil.TYPE_STRING, SerializableUtil.TYPE_STRING);
        result.setProperties(m);
        return result;
    }

    public static void main(String[] args) {
        MapObject obj = new MapObject();
        obj.getProperties().put(KEY_ID, "id1");
        obj.getProperties().put(KEY_NAME, "name1");
        obj.getProperties().put("p1", "property1");
        String s2 = "id-=-210000099-@-type-=-Substation-@-class-=-Substation-@-name-=-中双港变电站-@-aliasName-=-中双港变电站-@-";
        MapObject object = createObj(s2);
        System.out.println(object.getId());

    }
}
