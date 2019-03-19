package zju.bpamodel;

import zju.bpamodel.swi.*;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-11
 */
public class BpaSwiModel implements Serializable {
    List<GeneratorDW> generatorDws;
    List<Generator> generators;
    List<Exciter> exciters;
    List<ExciterExtraInfo> exciterExtraInfos;
    List<Load> loads;
    FFCard ff;
    Map<Generator, Exciter> exciterMap;
    Map<Generator, GeneratorDW> generatorDwMap;
    Map<Exciter, ExciterExtraInfo> exciterExtraInfoMap;

    public void buildMaps() {
        Map<String, GeneratorDW> idToGenDw = new HashMap<String, GeneratorDW>(generatorDws.size());
        for(GeneratorDW genDw : getGeneratorDws())
            idToGenDw.put(genDw.getBusName() + "_" + genDw.getId(), genDw);
        generatorDwMap = new HashMap<Generator, GeneratorDW>(generatorDws.size());
        for(Generator gen : generators) {
            String key = gen.getBusName() + "_" + gen.getId();
            if(idToGenDw.containsKey(key))
                generatorDwMap.put(gen, idToGenDw.get(key));
        }
        Map<String, Exciter> idToExciter = new HashMap<String, Exciter>(exciters.size());
        for(Exciter exciter : getExciters())
            idToExciter.put(exciter.getBusName() + "_" + exciter.getGeneratorCode(), exciter);
       //for(Exciter exciter : getExciters())
       //     if(exciter.getGeneratorCode() == ' ')
       //     System.out.println(exciter.getBusName() + "_" + (char)exciter.getGeneratorCode());
       exciterMap = new HashMap<Generator, Exciter>(exciters.size());
        for(Generator gen : generators) {
            String key = gen.getBusName() + "_" + gen.getId();
            if(idToExciter.containsKey(key))
                exciterMap.put(gen, idToExciter.get(key));
            else
                System.out.println("No exciter for generator is found: " + key);
        }
        Map<String, ExciterExtraInfo> idToExciterExtra = new HashMap<String, ExciterExtraInfo>(exciterExtraInfos.size());
        for(ExciterExtraInfo exciterExtra : getExciterExtraInfos())
            idToExciterExtra.put(exciterExtra.getBusName() + "_" + exciterExtra.getGeneratorCode(), exciterExtra);
        exciterExtraInfoMap = new HashMap<Exciter, ExciterExtraInfo>(exciterExtraInfos.size());
        for(Exciter exciter : exciters) {
            String key = exciter.getBusName() + "_" + exciter.getGeneratorCode();
            if(idToExciterExtra.containsKey(key))
                exciterExtraInfoMap.put(exciter, idToExciterExtra.get(key));
        }
    }
    public List<Generator> getGenerators() {
        return generators;
    }

    public void setGenerators(List<Generator> generators) {
        this.generators = generators;
    }

    public List<Exciter> getExciters() {
        return exciters;
    }

    public void setExciters(List<Exciter> exciters) {
        this.exciters = exciters;
    }

    public List<GeneratorDW> getGeneratorDws() {
        return generatorDws;
    }

    public void setGeneratorDws(List<GeneratorDW> generatorDws) {
        this.generatorDws = generatorDws;
    }

    public List<ExciterExtraInfo> getExciterExtraInfos() {
        return exciterExtraInfos;
    }

    public void setExciterExtraInfos(List<ExciterExtraInfo> exciterExtraInfos) {
        this.exciterExtraInfos = exciterExtraInfos;
    }

    public List<Load> getLoads() {
        return loads;
    }

    public void setLoads(List<Load> loads) {
        this.loads = loads;
    }

    public FFCard getFf() {
        return ff;
    }

    public void setFf(FFCard ff) {
        this.ff = ff;
    }

    public Map<Generator, Exciter> getExciterMap() {
        return exciterMap;
    }

    public void setExciterMap(Map<Generator, Exciter> exciterMap) {
        this.exciterMap = exciterMap;
    }

    public Map<Generator, GeneratorDW> getGeneratorDwMap() {
        return generatorDwMap;
    }

    public void setGeneratorDwMap(Map<Generator, GeneratorDW> generatorDwMap) {
        this.generatorDwMap = generatorDwMap;
    }

    public Map<Exciter, ExciterExtraInfo> getExciterExtraInfoMap() {
        return exciterExtraInfoMap;
    }

    public void setExciterExtraInfoMap(Map<Exciter, ExciterExtraInfo> exciterExtraInfoMap) {
        this.exciterExtraInfoMap = exciterExtraInfoMap;
    }

}
