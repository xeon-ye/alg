package zju.bpamodel;

import junit.framework.TestCase;

import java.text.SimpleDateFormat;
import java.util.Date;

public class ExeBpaTest extends TestCase {

    public void testOpenPSDEdit() {
        ExeBpa.openPSDEdit("C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/PSDEdit.exe");
    }

    public void testExePf() {
        long currentTime = System.currentTimeMillis();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
        Date date = new Date(currentTime);
        System.out.println(formatter.format(date));
        ExeBpa.exePf("d:/", "C:/Users/bingtekeji/Desktop/BPA/PSDEditCEPRI20180409-01/PFNT/PFNT.exe", "d:/XIAOJIN1.DAT");
    }

    public void testExeSw() {
        long currentTime = System.currentTimeMillis();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
        Date date = new Date(currentTime);
        System.out.println(formatter.format(date));
        ExeBpa.exeSw("d:/", "C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/SWNT/swnt.exe",
                "d:/XIAOJIN1.BSE", "d:/XIAOJIN1.SWI");
    }

    public void testClosePSDEdit() {
        ExeBpa.closePSDEdit();
    }
}