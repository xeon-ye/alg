package zju.bpamodel;

import junit.framework.TestCase;

public class ExeBpaTest extends TestCase {

    public void testOpenPSDEdit() {
        ExeBpa.openPSDEdit("C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/PSDEdit.exe");
    }

    public void testExePf() {
        ExeBpa.exePf("d:/", "C:/Users/bingtekeji/Desktop/BPA/PSDEditCEPRI20180409-01/PFNT/PFNT.exe", "d:/XIAOJIN1.DAT");
    }

    public void testExeSw() {
        ExeBpa.exeSw("d:/", "C:/Users/bingtekeji/Desktop/BPA/PSDEditCEPRI20180409-01/SWNT/swnt.exe",
                "d:/XIAOJIN1.BSE", "d:/XIAOJIN1.SWI");
    }

    public void testClosePSDEdit() {
        ExeBpa.closePSDEdit();
    }
}