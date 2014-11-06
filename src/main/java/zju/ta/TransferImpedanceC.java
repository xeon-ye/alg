package zju.ta;

import org.apache.log4j.Logger;
import zju.bpamodel.sccpc.SccBusResult;
import zju.bpamodel.swi.Generator;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-11-4
 */
public class TransferImpedanceC {
    private static Logger log = Logger.getLogger(TransferImpedanceC.class);
    SccBusResult shuntResult;
    HeffronPhilipsSystem hpSys;
    public double infiniteBusVx;
    public double infiniteBusVy;
    public double busToInfiniteBusX;

    public TransferImpedanceC() {
    }

    public TransferImpedanceC(SccBusResult shuntResult, HeffronPhilipsSystem hpSys) {
        this.shuntResult = shuntResult;
        this.hpSys = hpSys;
    }

    /**
     * reference is: He Yangzhan's book Power System Analysis example 6-4
     */
    public void cal() {
        //---------- the following process is according to bpa SCCPC handbook
        double xdpp = 0.0;
        if(hpSys.getGenDw() == null) {
            Generator gen = hpSys.getGen();
            switch (gen.getSubType()) {
                case 'C':
                    xdpp = hpSys.getGenXdp();
                    break;
                case 'F':
                case 'G':
                    if(Math.abs(gen.getXd() - gen.getXq()) < 1e-4) {
                        double kg1 = 0.65;//according to bpa software
                        double b = 1.00;  //according to bpa software
                        xdpp = b * kg1 * hpSys.getGenXdp();
                    } else {
                        double kg1 = 0.72; //according to bpa software
                        xdpp = kg1 * hpSys.getGenXdp();
                    }
                    break;
                default:
                    log.warn("Not support generator type when getting xdpp of " + gen.getBusName() + ", the type is:" + gen.getType());
            }
        }else
            xdpp = hpSys.getGenXdpp();
        //-------------------------------- end of xdpp calcualtion -------------------------------
        double x1 = xdpp;
        x1 += hpSys.getTransformerX();
        double baseZ = hpSys.getHighVBusPf().getBaseKv() * hpSys.getHighVBusPf().getBaseKv() / hpSys.getSysBaseMva();//todo: base mva is not perfect
        double positiveSequenceX = shuntResult.getPositiveSequenceX() / baseZ;
        setBusToInfiniteBusX(x1 * positiveSequenceX / (x1 - positiveSequenceX));

        //double genVx = hpSys.getGenBusPf().getvInPu() * Math.cos(hpSys.getGenBusPf().getAngleInArc());
        //double genVy = hpSys.getGenBusPf().getvInPu() * Math.sin(hpSys.getGenBusPf().getAngleInArc());
        //double fai = hpSys.getGenBusPf().getAngleInArc() - hpSys.getiAngleInArc();
        //double eppx = genVx + xdpp * hpSys.getiAmpl() * Math.cos(hpSys.getiAngleInArc()) * Math.sin(fai);
        //double eppy = genVy + xdpp * hpSys.getiAmpl() * Math.sin(hpSys.getiAngleInArc()) * Math.sin(fai);
        //infiniteBusVx = (highVx * (x1 + busToInfiniteBusX) - eppx * x1) / busToInfiniteBusX;
        //infiniteBusVy = (highVy * (x1 + busToInfiniteBusX) - eppy * x1) / busToInfiniteBusX;

    }

    /**
     * reference is He Yangzhan's book Power System Analysis (6-36)
     */
    public void cal2() {//todo: plant with many generators connected to a same bus is not considered
        setBusToInfiniteBusX(hpSys.getSysBaseMva() / shuntResult.getShuntCapacity());
    }

    public double getInfiniteBusVx() {
        return infiniteBusVx;
    }

    public double getInfiniteBusVy() {
        return infiniteBusVy;
    }

    public void setShuntResult(SccBusResult shuntResult) {
        this.shuntResult = shuntResult;
    }

    public void setHpSys(HeffronPhilipsSystem hpSys) {
        this.hpSys = hpSys;
    }

    public double getBusToInfiniteBusX() {
        return busToInfiniteBusX;
    }

    public void setBusToInfiniteBusX(double busToInfiniteBusX) {
        this.busToInfiniteBusX = busToInfiniteBusX;
        double highVx = hpSys.getHighVBusPf().getvInPu() * Math.cos(hpSys.getHighVBusPf().getAngleInArc());
        double highVy = hpSys.getHighVBusPf().getvInPu() * Math.sin(hpSys.getHighVBusPf().getAngleInArc());
        double ix = hpSys.getiAmpl() * Math.cos(hpSys.getiAngleInArc());
        double iy = hpSys.getiAmpl() * Math.sin(hpSys.getiAngleInArc());
        infiniteBusVx = highVx + iy * busToInfiniteBusX ;
        infiniteBusVy = highVy - ix * busToInfiniteBusX;
    }
}

