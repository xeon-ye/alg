package zju.ta;

import zju.bpamodel.pf.AcLine;
import zju.bpamodel.pf.Bus;
import zju.bpamodel.pf.Transformer;
import zju.bpamodel.pfr.BusPfResult;
import zju.bpamodel.swi.Exciter;
import zju.bpamodel.swi.ExciterExtraInfo;
import zju.bpamodel.swi.Generator;
import zju.bpamodel.swi.GeneratorDW;
import zju.tmodel.TGenModel;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-10-25
 */
public class HeffronPhilipsSystem {
    private double sysBaseMva = 100.0;

    //swi related devices
    private Generator gen;
    private GeneratorDW genDw;
    private Exciter exciter;
    private ExciterExtraInfo exciterExtraInfo;

    //powerflow related devices
    private Transformer[] transformers;
    private AcLine[] aclines;
    private Bus genBus;
    private Bus virtualBus;
    private Bus highVBus;
    private Bus infiniteBus;

    //power flow results
    private BusPfResult genBusPf;
    private BusPfResult highVBusPf;

    //parameters
    private double k1;
    private double k2;
    private double k3;
    private double k4;
    private double k5;
    private double k6;
    private double xe;
    private double xdSigma;
    private double xdpSigma;
    private double xqSigma;
    private double xqpSigma;
    private double us;
    private double ut;
    private double iAmpl;
    private double iAngleInArc;
    private TGenModel genModel;
    private double uh1X;
    private double uh1Y;
    private double thetaUh1;
    private double thetaUh1q;
    public double genXd;
    public double genXq;
    public double genXdp;
    public double genXqp;
    private double genXdpp;
    private double genXqpp;

    public void initialPara() {
        xe = 0.0;
        xe += getAclineX();
        xe += getTransformerX();
        genXd = gen.getXd();
        genXd = gen.getXd();
        genXq = gen.getXq();
        genXdp = gen.getXdp();
        genXqp = gen.getXqp();
        if(genDw != null) {
            genXdpp = genDw.getXdpp();
            genXqpp = genDw.getXqpp();
        }
        xdSigma = genXd + xe;
        xdpSigma = genXdp + xe;
        xqSigma = genXq + xe;
        xqpSigma = genXqp + xe;
        us = highVBusPf.getvInPu();//todo:There should infinite bus's voltage amplitude.
        ut = genBusPf.getvInPu();
        double temp1 = genBusPf.getvInPu() * Math.cos(genBusPf.getAngleInArc()) - highVBusPf.getvInPu() * Math.cos(highVBusPf.getAngleInArc());
        double temp2 = genBusPf.getvInPu() * Math.sin(genBusPf.getAngleInArc()) - highVBusPf.getvInPu() * Math.sin(highVBusPf.getAngleInArc());
        iAmpl = Math.sqrt(temp1 * temp1 + temp2 * temp2) / xe;
        iAngleInArc = Math.atan2(-temp1, temp2);
        genModel = new TGenModel(gen);
        genModel.cal(genBusPf.getvInPu(), genBusPf.getAngleInArc(), iAmpl, iAngleInArc, sysBaseMva);
        genModel.calThirdOrderInitial();
        double delta = genModel.getDelta();
        double eqp = genModel.getEqp();
        k1 = eqp * us * Math.cos(delta) / xdpSigma + (genXdp - genXd) * us * us * Math.cos(2.0 * delta) / (xdpSigma * xdSigma);
        k2 = us * Math.sin(delta) / xdpSigma;
        k3 = xdpSigma / xdSigma;
        k4 = (genXd - genXdp) * ut * Math.sin(delta) / xdpSigma;
        k5 = genModel.getUd() * us * genXq * Math.cos(delta) / (ut * xqSigma) - genModel.getUq() * us * genXdp * Math.sin(delta) / (ut * xdpSigma);
        k6 = genModel.getUq() * (xdSigma - genXd) / (ut * xdpSigma);
        //double thetaUh1q = delta - x[3];
    }

    /**
     * @param xc xc of exciter
     * @return deltaTs and deltaTd
     */
    public double[] setExciterXc(double xc) {
        double delta = genModel.getDelta();
        uh1X = us * Math.cos(highVBusPf.getAngleInArc()) - iAmpl * Math.sin(iAngleInArc) * (xe - xc);
        uh1Y = us * Math.sin(highVBusPf.getAngleInArc()) + iAmpl * Math.cos(iAngleInArc) * (xe - xc);
        thetaUh1 = Math.atan2(uh1Y, uh1X);
        thetaUh1q = delta - thetaUh1;
        //todo: k5 may wrong, according Chen Lin's paper
        k5 = Math.sin(thetaUh1q) * us * (genXq + xc) * Math.cos(delta) / xqSigma - Math.cos(thetaUh1q) * us * (genXdp + xc) * Math.sin(delta) / xdpSigma;
        k6 = Math.cos(thetaUh1q) * (xdSigma - genXd - xc) / xdpSigma;
        double ka = getExciterK();
        //the following fou may have problems
        double tmp1 = (1.0 / k3) + k6 * ka - gen.getTdop() * exciter.getTa();
        double tmp2 = gen.getTdop() + exciter.getTa() / k3;
        double deltaTs = -k2 * k5 * ka * tmp1 / (tmp1 * tmp1 + tmp2 * tmp2);
        double deltaTd = k2 * k5 * ka * tmp2 / (tmp1 * tmp1 + tmp2 * tmp2);
        return new double[]{deltaTs, deltaTd};
    }

    public String toBpaSwiData() {
        StringBuilder swiStr = new StringBuilder();

        if (genDw != null)
            swiStr.append(genDw.toString()).append("\n");
        swiStr.append(gen.toString()).append("\n");
        swiStr.append(exciter.toString()).append("\n");
        if(exciterExtraInfo != null)
            swiStr.append(exciterExtraInfo.toString()).append("\n");

        Generator infiniteGen = new Generator();
        infiniteGen.setBusName(infiniteBus.getName());
        infiniteGen.setBaseKv(highVBusPf.getBaseKv());
        infiniteGen.setSubType('C');
        infiniteGen.seteMWS(999999);
        infiniteGen.seteMWS(999999);
        infiniteGen.setXdp(0.0100);
        swiStr.append(infiniteGen.toString()).append("\n");
        return swiStr.toString();
    }

    public String toBpaPfData() {
        StringBuilder str = new StringBuilder();
        str.append(genBus.toString()).append("\n");
        if(virtualBus != null)
            str.append(virtualBus.toString()).append("\n");
        str.append(highVBus.toString()).append("\n");
        str.append(infiniteBus.toString()).append("\n");
        for(AcLine line : aclines)
            str.append(line.toString()).append("\n");
        for(Transformer t : transformers)
            str.append(t.toString()).append("\n");
        return str.toString();
    }

    //---------------- getters and setters --------------------

    public double getSysBaseMva() {
        return sysBaseMva;
    }

    public void setSysBaseMva(double sysBaseMva) {
        this.sysBaseMva = sysBaseMva;
    }

    public Generator getGen() {
        return gen;
    }

    public void setGen(Generator gen) {
        this.gen = gen;
    }

    public GeneratorDW getGenDw() {
        return genDw;
    }

    public void setGenDw(GeneratorDW genDw) {
        this.genDw = genDw;
    }

    public Exciter getExciter() {
        return exciter;
    }

    public void setExciter(Exciter exciter) {
        this.exciter = exciter;
    }

    public ExciterExtraInfo getExciterExtraInfo() {
        return exciterExtraInfo;
    }

    public void setExciterExtraInfo(ExciterExtraInfo exciterExtraInfo) {
        this.exciterExtraInfo = exciterExtraInfo;
    }

    public Transformer[] getTransformers() {
        return transformers;
    }

    public void setTransformers(Transformer[] transformers) {
        this.transformers = transformers;
    }

    public AcLine[] getAclines() {
        return aclines;
    }

    public void setAclines(AcLine[] aclines) {
        this.aclines = aclines;
    }

    public BusPfResult getGenBusPf() {
        return genBusPf;
    }

    public void setGenBusPf(BusPfResult genBusPf) {
        this.genBusPf = genBusPf;
    }

    public BusPfResult getHighVBusPf() {
        return highVBusPf;
    }

    public void setHighVBusPf(BusPfResult highVBusPf) {
        this.highVBusPf = highVBusPf;
    }

    public double getK1() {
        return k1;
    }

    public double getK2() {
        return k2;
    }

    public double getK3() {
        return k3;
    }

    public double getK4() {
        return k4;
    }

    public double getK5() {
        return k5;
    }

    public double getK6() {
        return k6;
    }

    public double getXe() {
        return xe;
    }

    public double getXdSigma() {
        return xdSigma;
    }

    public double getXdpSigma() {
        return xdpSigma;
    }

    public double getXqSigma() {
        return xqSigma;
    }

    public double getXqpSigma() {
        return xqpSigma;
    }

    public double getUs() {
        return us;
    }

    public double getUt() {
        return ut;
    }

    public double getiAmpl() {
        return iAmpl;
    }

    public double getiAngleInArc() {
        return iAngleInArc;
    }

    public double getUh1X() {
        return uh1X;
    }

    public double getUh1Y() {
        return uh1Y;
    }

    public double getThetaUh1() {
        return thetaUh1;
    }

    public double getThetaUh1q() {
        return thetaUh1q;
    }

    public TGenModel getGenModel() {
        return genModel;
    }

    public Bus getGenBus() {
        return genBus;
    }

    public void setGenBus(Bus genBus) {
        this.genBus = genBus;
    }

    public Bus getHighVBus() {
        return highVBus;
    }

    public void setHighVBus(Bus highVBus) {
        this.highVBus = highVBus;
    }

    public Bus getInfiniteBus() {
        return infiniteBus;
    }

    public void setInfiniteBus(Bus infiniteBus) {
        this.infiniteBus = infiniteBus;
    }

    public Bus getVirtualBus() {
        return virtualBus;
    }

    public void setVirtualBus(Bus virtualBus) {
        this.virtualBus = virtualBus;
    }

    public double getExciterK() {
        double ka = exciter.getKa();
        double kb = 1.0;
        if(exciterExtraInfo != null) {
            if(exciterExtraInfo.getKb() > 1e-2)
                kb = exciterExtraInfo.getKb();
        }
        if(ka > 1e-2) {
            if(exciter.getK() > 1e-2)
                return exciter.getK() * ka * kb;
            else
                return ka * kb;
        } else {
            if(exciter.getK() > 1e-2)
                return exciter.getK() * kb;
            else
                return kb;
        }
    }

    public double getExciterT() {
        double ta = exciter.getTa();
        //if(exciterExtraInfo != null)
        //    ta *= exciterExtraInfo.getKb();
        return ta;
    }

    public double getGenXd() {
        return genXd;
    }

    public double getGenXq() {
        return genXq;
    }

    public double getGenXdp() {
        return genXdp;
    }

    public double getGenXqp() {
        return genXqp;
    }

    public double getGenXdpp() {
        return genXdpp;
    }

    public double getGenXqpp() {
        return genXqpp;
    }

    public double getTransformerX() {
        double transformerX = 0.0;
        if (transformers != null)
            for (Transformer t : transformers)
                transformerX += t.getX();
        return gen.getBaseMva() > 0 ? transformerX * gen.getBaseMva() / sysBaseMva: transformerX;
    }

    public double getAclineX() {
        double t = 0;
        double aclineX;
        if(aclines != null)
            for (AcLine line : aclines)
                t += 1.0 / line.getX();
        if(t > 0)
            aclineX = 1.0 / t;
        else
            aclineX = 0.0;
        return gen.getBaseMva() > 0 ? aclineX * gen.getBaseMva() / sysBaseMva: aclineX;
    }
}
