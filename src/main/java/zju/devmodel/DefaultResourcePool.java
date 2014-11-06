package zju.devmodel;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2011-4-7
 */
public class DefaultResourcePool implements Serializable {
    private Map<String, MapObject> id2Obj = new HashMap<String, MapObject>();
    private Map<String, List<MapObject>> classToList = new HashMap<String, List<MapObject>>();
    private Map<String, List<MapObject>> typeToList = new HashMap<String, List<MapObject>>();

    public List<MapObject> getResourceByClass(String className) {
        List<MapObject> l = classToList.get(className);
        if (l != null) return l;
        List<MapObject> result = new ArrayList<MapObject>();//todo: not efficient
        for (MapObject d : id2Obj.values())
            if (className.equals(d.getProperty(MapObject.KEY_CLASS)))
                result.add(d);
        classToList.put(className, result);
        return result;
    }

    public List<MapObject> getResourceByClass(Class c) {
        return getResourceByClass(c.getName());
    }

    public List<MapObject> getResourceByType(String type) {
        return getResourceByType(new String[]{type});
    }

    /**
     * this method is not efficient yet!
     * @param types types to found
     * @return Map
     */
    public List<MapObject> getResourceByType(String[] types) {
        Map<String, MapObject> tmpMap = new HashMap<String, MapObject>();
        for(String type : types) {
            List<MapObject> l = typeToList.get(type);
            if(l == null) {
                l = new ArrayList<MapObject>();
                for (MapObject d : id2Obj.values()) {
                    if(d.getType()  == null)
                        continue;
                    String[] str = d.getType().split(MapObject.TYPE_SEPARATOR);
                    for(String s : str) {
                        if (!type.equals(s))
                            continue;
                        l.add(d);
                        break;
                    }
                }
                typeToList.put(type, l);
            }
            for(MapObject obj : l)
                tmpMap.put(obj.getId(), obj);
        }
        return new ArrayList<MapObject>(tmpMap.values());
    }

    public List<MapObject> getResourceAll() {
        return new ArrayList<MapObject>(id2Obj.values());
    }

    public MapObject getResource(String id) {
        return id2Obj.get(id);
    }

    public void addResource(MapObject obj) {
        id2Obj.put(obj.getId(), obj);
        getResourceByClass(obj.getClassName()).add(obj);
        if(obj.getType() != null) {
            String[] types = obj.getType().split(MapObject.TYPE_SEPARATOR);
            for(String type : types)
                getResourceByType(type).add(obj);
        }
    }

    public void removeResource(MapObject obj) {
        removeResource(obj.getId());
    }

    //remove obj is not efficient, do not invoke method frequently!
    //son devices of the object is not removed
    public void removeResource(String id) {
        MapObject obj = id2Obj.remove(id);
        if(obj == null)
            return;
        classToList.get(obj.getClassName()).remove(obj);
        if(obj.getType() == null)
            return;
        String[] types = obj.getType().split(MapObject.TYPE_SEPARATOR);
        for(String type : types)
            getResourceByType(type).remove(obj);
    }

    public void clear() {
        id2Obj.clear();
        classToList.clear();
        typeToList.clear();
    }

    //------------------ getter and setter starts ----------------
    public Map<String, MapObject> getId2Obj() {
        return id2Obj;
    }

    public void setId2Obj(Map<String, MapObject> id2Obj) {
        this.id2Obj = id2Obj;
    }

    public Map<String, List<MapObject>> getClassToList() {
        return classToList;
    }

    public Map<String, List<MapObject>> getTypeToList() {
        return typeToList;
    }

    public void setClassToList(Map<String, List<MapObject>> classToList) {
        this.classToList = classToList;
    }

    public void setTypeToList(Map<String, List<MapObject>> typeToList) {
        this.typeToList = typeToList;
    }

}
