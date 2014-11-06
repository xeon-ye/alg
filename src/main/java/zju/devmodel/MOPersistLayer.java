package zju.devmodel;

import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @Author: Yanxue
 * Date: 2011-3-28
 */
public interface MOPersistLayer {

    public void initial();

    public boolean save(MapObject mapObject);

    public boolean save(List<MapObject> list);

    public MapObject query(String sid);

    public List<MapObject> queryByProperty(String propertyName, String value);

    public boolean update(MapObject mapObject);

    public boolean delete(String sid);

    public Map<String, MapObject> getAllObjectMap();

}
