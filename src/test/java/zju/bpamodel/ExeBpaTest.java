package zju.bpamodel;

import junit.framework.TestCase;

public class ExeBpaTest extends TestCase {

    public void testOpenPSDEdit() {
        ExeBpa.openPSDEdit("C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/PSDEdit.exe");
    }

    public void testExePf() {
        ExeBpa.exePf("C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/PFNT/PFNT.exe", "C:/Users/Administrator/Desktop/BPA用户手册（PDF）/XIAOJIN.DAT");
    }

    public void testExeSw() {
        ExeBpa.exeSw("C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/SWNT/SWNT.exe",
                "C:/Users/Administrator/Desktop/BPA用户手册（PDF）/88000007.BSE", "C:/Users/Administrator/Desktop/BPA用户手册（PDF）/XIAOJIN.SWI");
    }

    public void testClosePSDEdit() {
        ExeBpa.closePSDEdit();
    }
}