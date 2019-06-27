package zju.bpamodel;

import junit.framework.TestCase;

public class ExeBpaTest extends TestCase {

    public void testOpenPSDEdit() {
        ExeBpa.openPSDEdit("C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/PSDEdit.exe");
    }

    public void testExePf() {
        ExeBpa.exePf("d:", "C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/PFNT/PFNT.exe", "d:/XIAOJIN.DAT");
    }

    public void testExeSw() {
        ExeBpa.exeSw("d:", "C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/SWNT/SWNT.exe",
                "d:/88000007.BSE", "C:/Users/Administrator/Desktop/BPA用户手册（PDF）/XIAOJIN.SWI");
    }

    public void testClosePSDEdit() {
        ExeBpa.closePSDEdit();
    }
}