package zju.util;

import java.beans.XMLDecoder;
import java.beans.XMLEncoder;
import java.io.*;

public class JOFileUtil {
    public static String encode2XML(Object obj) {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        XMLEncoder en = new XMLEncoder(stream);
        en.writeObject(obj);
        en.close();
        return stream.toString();
    }

    public static void encode2XML(Object obj, File file) {
        try {
            XMLEncoder en = new XMLEncoder(new BufferedOutputStream(new FileOutputStream(file)));
            en.writeObject(obj);
            en.close();
        } catch (FileNotFoundException e1) {
            e1.printStackTrace();
        }
    }

    public static void encode2XML(Object obj, String file) {
        encode2XML(obj, new File(file));
    }

    public static Object decode4XML(File file) {
        try {
            return decode4XML(new FileInputStream(file));
        } catch (FileNotFoundException e) {
            return null;
        }
    }

    public static Object decode4XML(String file) {
        return decode4XML(new File(file));
    }

    public static Object decode4XML(InputStream stream) {
        XMLDecoder d = new XMLDecoder(stream);
        Object result = d.readObject();
        d.close();
        return result;
    }

    public static void encode4Ser(Object obj, String file) {
        ObjectOutputStream oos;
        try {
            FileOutputStream fos = new FileOutputStream(file);
            oos = new ObjectOutputStream(fos);
            oos.writeObject(obj);
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Object decode4Ser(String file) {
        ObjectInputStream ois;
        try {
            FileInputStream fis = new FileInputStream(file);
            ois = new ObjectInputStream(fis);
            Object obj = ois.readObject();
            ois.close();
            return obj;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static Object decode4Ser(InputStream stream) {
        try {
            ObjectInputStream d = new ObjectInputStream(stream);
            Object result = d.readObject();
            d.close();
            return result;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static Object cloneObj(Serializable obj) {
        try {
            // create a buffer in memory
            ByteArrayOutputStream bout = new ByteArrayOutputStream();
            ObjectOutputStream out = new ObjectOutputStream(bout);
            //write serializable obj to the buffer
            out.writeObject(obj);
            out.close();

            // find the buffer the object is written to
            ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
            ObjectInputStream in = new ObjectInputStream(bin);
            //read the content in the buffer and form a new object
            Object ret = in.readObject();
            in.close();
            //copy finished.
            return ret;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}

