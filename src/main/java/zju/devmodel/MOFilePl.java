package zju.devmodel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.util.JOFileUtil;

import java.io.File;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2011-4-1
 */
public class MOFilePl implements MOPersistLayer {
    private static Logger log = LogManager.getLogger(MOFilePl.class);

    private File dir;//file dirtory of device objects to store

    public MOFilePl(File dir) {
        this.dir = dir;
    }

    public MOFilePl() {
    }

    public void initial() {
    }

    public boolean save(MapObject mapObject) {
        if(dir == null || !dir.exists() || !dir.isDirectory()) {
            log.error("Directory for storing device objects is not exist or has not been set yet!");
            return false;
        }
        JOFileUtil.encode2XML(mapObject, dir.getPath() + File.separator + mapObject.getId());
        return true;
    }

    public boolean save(List<MapObject> list) {
        return false;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public void save(MapObject[] mapObject) {
        //To change body of implemented methods use File | Settings | File Templates.
    }

    public MapObject query(String sid) {
        if(dir == null || !dir.exists() || !dir.isDirectory()) {
            log.error("Directory for storing device objects is not exist or has not been set yet!");
            return null;
        }
        File f = new File(dir.getPath() + File.separator + sid);
        if(f.exists())
            return (MapObject) JOFileUtil.decode4XML(f);
        return null;
    }

    public List<MapObject> queryByProperty(String propertyName, String value) {
        //todo: finish it..
        return null;
    }

    public boolean update(MapObject mapObject) {
        if(dir == null || !dir.exists() || !dir.isDirectory()) {
            log.error("Directory for storing device objects is not exist or has not been set yet!");
            return false;
        }
        JOFileUtil.encode2XML(mapObject, dir.getPath() + File.separator + mapObject.getId());
        return true;
    }

    public boolean delete(String sid) {
        if(dir == null || !dir.exists() || !dir.isDirectory()) {
            log.error("Directory for storing device objects is not exist or has not been set yet!");
            return false;
        }
        File f = new File(dir.getPath() + File.separator + sid);
        if(f.exists()) {
            if(!f.delete()) {
                log.error("Failed to delete file : " + f.getPath());
                return false;
            }
            return true;
        }else {
            log.warn("File is not exist: " + f.getPath());
            return false;
        }
    }

    public Map<String, MapObject> getAllObjectMap() {
        //todo: not finished
        return null;
    }

    public File getDir() {
        return dir;
    }

    public void setDir(File dir) {
        this.dir = dir;
    }
}
