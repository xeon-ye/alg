package zju.ieeeformat;

import java.util.*;

/**
 * Class IcfDataUtil
 * <p>util class providing methods to deal with IEEEDataIsland</P>
 *
 * @author Dong Shufeng
 * @version 1.0
 * @Date: 2007-3-28
 */
public class IcfDataUtil {
    public static final IEEEDataIsland ISLAND_14;
    public static final IEEEDataIsland ISLAND_30;
    public static final IEEEDataIsland ISLAND_39;
    public static final IEEEDataIsland ISLAND_57;
    public static final IEEEDataIsland ISLAND_118;
    public static final IEEEDataIsland ISLAND_300;
    public static final String ISLAND_14_PATH = "/ieee/case14.txt";
    public static final String ISLAND_30_PATH = "/ieee/case30.txt";
    public static final String ISLAND_39_PATH = "/ieee/case39.txt";
    public static final String ISLAND_57_PATH = "/ieee/case57.txt";
    public static final String ISLAND_118_PATH = "/ieee/case118.txt";
    public static final String ISLAND_300_PATH = "/ieee/case300.txt";

    static {
        DefaultIcfParser parser = new DefaultIcfParser();
        ISLAND_14 = parser.parse(IcfDataUtil.class.getResourceAsStream(ISLAND_14_PATH));
        ISLAND_30 = parser.parse(IcfDataUtil.class.getResourceAsStream(ISLAND_30_PATH));
        ISLAND_39 = parser.parse(IcfDataUtil.class.getResourceAsStream(ISLAND_39_PATH));
        ISLAND_57 = parser.parse(IcfDataUtil.class.getResourceAsStream(ISLAND_57_PATH));
        ISLAND_118 = parser.parse(IcfDataUtil.class.getResourceAsStream(ISLAND_118_PATH));
        ISLAND_300 = parser.parse(IcfDataUtil.class.getResourceAsStream(ISLAND_300_PATH));
    }

    public static IEEEDataIsland sortByBusData(IEEEDataIsland toSort, String[] sortOrder) {
        String[] original = BusData.getCompareOrder();
        BusData.setCompareOrder(sortOrder);
        Collections.sort(toSort.getBuses());
        BusData.setCompareOrder(original);
        return toSort;
    }

    public static IEEEDataIsland sortByBrandData(IEEEDataIsland toSort, String[] sortOrder) {
        String[] original = BranchData.getCompareOrder();
        BranchData.setCompareOrder(sortOrder);
        Collections.sort(toSort.getBranches());
        BranchData.setCompareOrder(original);
        return toSort;
    }

    /**
     * @param island ieee data island
     * @return key = num1->num2, value = branch's numbers
     *         num1 = branch's tap bus number<br>
     *         num2 = branch's z bus number<br>
     */
    public static Map<String, List<Integer>> getBranchIds(IEEEDataIsland island) {
        Map<String, List<Integer>> branchIds = new HashMap<String, List<Integer>>();
        for (BranchData branch : island.getBranches()) {
            String key1 = getBus2BusId(branch.getTapBusNumber(), branch.getZBusNumber());
            if (branchIds.containsKey(key1))
                branchIds.put(key1, new ArrayList<Integer>(3));
            branchIds.get(key1).add(branch.getId());
        }
        return branchIds;
    }

    /**
     * branch id is num1->num2<br>
     * num1 = branch's tap bus number<br>
     * num2 = branch's z bus number<br>
     * this method is used when these is at most one branch bitween tow buses.
     *
     * @param island ieee data island
     * @return key = branch's id, value = branch's number(start from 1)
     */
    public static Map<String, Integer> getBranchId2(IEEEDataIsland island) {
        HashMap<String, Integer> branchIds = new HashMap<String, Integer>(island.getBranches().size());
        for (BranchData branch : island.getBranches()) {
            String key = getBus2BusId(branch.getTapBusNumber(), branch.getZBusNumber());
            branchIds.put(key, branch.getId());
        }
        return branchIds;
    }

    /**
     * branch id is num1->num2.circuit<br>
     * num1 = branch's tap bus number<br>
     * num2 = branch's z bus number<br>
     * circuit = branch's circuit
     *
     * @param island ieee data island
     * @return key = branch's id, value = branch's number(start from 1)
     */
    public static Map<String, Integer> getBranchId(IEEEDataIsland island) {
        HashMap<String, Integer> branchIds = new HashMap<String, Integer>(island.getBranches().size());
        for (BranchData branch : island.getBranches()) {
            String key = getBranchId(branch.getTapBusNumber(), branch.getZBusNumber(), branch.getCircuit());
            branchIds.put(key, branch.getId());
        }
        return branchIds;
    }

    public static String getBus2BusId(int tapBusNumber, int zBusNumber) {
        if (tapBusNumber < zBusNumber)
            return tapBusNumber + "->" + zBusNumber;
        else
            return zBusNumber + "->" + tapBusNumber;
    }

    public static String getBranchId(int tapBusNumber, int zBusNumber, int circuit) {
        return getBus2BusId(tapBusNumber, zBusNumber) + "." + circuit;
    }

    public static String getBranchId(BusData tapBus, BusData zBus, int circuit) {
        return getBranchId(tapBus.getBusNumber(), zBus.getBusNumber(), circuit);
    }

    public static int[] getBranchIds(int tapBusNumber, int zBusNumber, Map<String, Integer> branchIds) {
        int[] tmp = new int[3];//todo: not perfect
        int index = 0;
        for (int i = 0; i < 3; i++) {
            String key = getBranchId(tapBusNumber, zBusNumber, i);
            if (branchIds.containsKey(key)) {
                tmp[index] = branchIds.get(key);
                index++;
            }
        }
        int[] result = new int[index];
        System.arraycopy(tmp, 0, result, 0, index);
        return result;
    }

    public static int[] getBranchIds(BusData tapBus, BusData zBus, Map<String, Integer> branchIds) {
        return getBranchIds(tapBus.getBusNumber(), zBus.getBusNumber(), branchIds);
    }

    /**
     * @param island ieee data island
     * @return a map key: bus number, value: number list of buses connect to this bus
     */
    public static Map<Integer, List<Integer>> getConnectedBuses(IEEEDataIsland island) {
        Map<Integer, List<Integer>> bus2buses = new HashMap<Integer, List<Integer>>(island.getBuses().size());
        for (BusData bus : island.getBuses())
            bus2buses.put(bus.getBusNumber(), new ArrayList<Integer>());
        for (BranchData branch : island.getBranches()) {
            if (branch.getTapBusNumber() == branch.getZBusNumber()) {
                System.out.println("Warning! Tap bus number equals z bus number...");
                continue;
            }
            int head = branch.getTapBusNumber();
            int tail = branch.getZBusNumber();
            if (!bus2buses.get(head).contains(new Integer(tail)))
                bus2buses.get(head).add(tail);
            if (!bus2buses.get(tail).contains(new Integer(head)))
                bus2buses.get(tail).add(head);
        }
        return bus2buses;
    }

    public static Map<Integer, List<BranchData>> getBus2Branches(IEEEDataIsland island) {
        Map<Integer, List<BranchData>> map = new HashMap<Integer, List<BranchData>>();
        for (BranchData branch : island.getBranches()) {
            if (!map.containsKey(branch.getTapBusNumber())) {
                map.put(branch.getTapBusNumber(), new ArrayList<BranchData>());
            }
            map.get(branch.getTapBusNumber()).add(branch);
            if (!map.containsKey(branch.getZBusNumber())) {
                map.put(branch.getZBusNumber(), new ArrayList<BranchData>());
            }
            map.get(branch.getZBusNumber()).add(branch);
        }
        return map;
    }

    /**
     * @param island ieee data island
     * @return a map key: bus number, value: number list of buses connect to this bus
     */
    public static Map<BusData, List<BusData>> getBus2Buses(IEEEDataIsland island) {
        Map<BusData, List<BusData>> bus2buses = new HashMap<BusData, List<BusData>>(island.getBuses().size());
        Map<Integer, BusData> busMap = island.getBusMap();
        for (BusData bus : island.getBuses())
            bus2buses.put(bus, new ArrayList<BusData>());
        for (BranchData branch : island.getBranches()) {
            if (branch.getTapBusNumber() == branch.getZBusNumber()) {
                System.out.println("Warning! Tap bus number equals z bus number : " + branch.getTapBusNumber());
                continue;
            }
            BusData head = busMap.get(branch.getTapBusNumber());
            BusData tail = busMap.get(branch.getZBusNumber());
            if (!bus2buses.get(head).contains(tail))
                bus2buses.get(head).add(tail);
            if (!bus2buses.get(tail).contains(head))
                bus2buses.get(tail).add(head);
        }
        return bus2buses;
    }

    /**
     * the method is used to seperate a big island into isolated sub islands.
     *
     * @param island big island with sub islands in it
     * @return array of sub islands
     */
    public static List<IEEEDataIsland> buildIsolateIslands(IEEEDataIsland island) {
        Map<BusData, List<BusData>> bus2buses = getBus2Buses(island);
        Map<Integer, IEEEDataIsland> dealed = new HashMap<Integer, IEEEDataIsland>(bus2buses.size());
        int islandNum = 0;
        for (BusData busData : island.getBuses()) {
            if (!dealed.containsKey(busData.getBusNumber())) {
                dealed.put(busData.getBusNumber(), new IEEEDataIsland());
                islandNum++;
            } else
                continue;
            formIsland(busData, bus2buses, dealed);
        }
        Map<Integer, BusData> busMap = island.getBusMap();
        List<IEEEDataIsland> islands = new ArrayList<IEEEDataIsland>(islandNum);
        Map<IEEEDataIsland, List<BusData>> busLists = new HashMap<IEEEDataIsland, List<BusData>>(islandNum);
        Map<IEEEDataIsland, List<BranchData>> branchLists = new HashMap<IEEEDataIsland, List<BranchData>>(islandNum);
        for (int busNum : dealed.keySet()) {
            if (!busLists.containsKey(dealed.get(busNum))) {
                busLists.put(dealed.get(busNum), new ArrayList<BusData>());
                islands.add(dealed.get(busNum));
            }
            busLists.get(dealed.get(busNum)).add(busMap.get(busNum));
        }
        for (BranchData branchData : island.getBranches()) {
            IEEEDataIsland key = dealed.get(branchData.getTapBusNumber());
            if (!branchLists.containsKey(key))
                branchLists.put(key, new ArrayList<BranchData>());
            branchLists.get(key).add(branchData);
        }
        for (IEEEDataIsland v : islands) {
            v.setTitle(island.getTitle());
            v.setBuses(busLists.get(v));
            v.setBranches(branchLists.get(v) == null ? new ArrayList<BranchData>() : branchLists.get(v));
            v.setTieLines(new ArrayList<TieLineData>());//todo: not finished
            v.setLossZones(new ArrayList<LossZoneData>());
            v.setInterchanges(new ArrayList<InterchangeData>());
        }
        return islands;
    }

    private static void formIsland(BusData toDeal, Map<BusData, List<BusData>> bus2buses, Map<Integer, IEEEDataIsland> dealed) {
        List<BusData> connected = bus2buses.get(toDeal);
        IEEEDataIsland currentIsland = dealed.get(toDeal.getBusNumber());
        //System.out.println("Dealing bus: " + toDeal.getBusNo());
        for (BusData bus : connected) {
            //System.out.println("Dealing connected bus :" + bus.getBusNo());
            dealed.put(bus.getBusNumber(), currentIsland);
            bus2buses.get(bus).remove(toDeal);
            //if(bus.getBaseVoltage() > 400.0)
            //    System.out.println("pause...");
        }
        for (BusData bus : connected)
            formIsland(bus, bus2buses, dealed);
        connected.clear();
    }

    /**
     * get island with maximum buses.<br>
     * when there ara islands with same maximum bus number, the island at front positon in array will be returned.
     *
     * @param islands array of islands
     * @return the island with maximum buses
     */
    public static IEEEDataIsland getLargestIsland(List<IEEEDataIsland> islands) {
        IEEEDataIsland largest = null;
        for (IEEEDataIsland island : islands)
            if (largest == null || island.getBuses().size() > largest.getBuses().size())
                largest = island;
        return largest;
    }

    public static SubareaModel splitTowLayers(IEEEDataIsland island, int highLevel, int lowLevel) {
        //step1: all transformer traversal to find the boundary lines
        Map<Integer, BusData> busMap = island.getBusMap();
        List<BranchData> ties = new ArrayList<BranchData>();
        Map<BusData, List<BusData>> bus2buses = getBus2Buses(island);
        for (BranchData branch : island.getBranches()) {
            if (branch.getType() == 0)
                continue;
            //it is a transformer
            int bus1 = branch.getTapBusNumber();
            int bus2 = branch.getZBusNumber();
            BusData b1 = busMap.get(bus1);
            BusData b2 = busMap.get(bus2);
            double baseV1 = b1.getBaseVoltage();
            double baseV2 = b2.getBaseVoltage();
            int level1 = KVLevelPicker.getKVLevel(baseV1);
            int level2 = KVLevelPicker.getKVLevel(baseV2);

            if (level1 == highLevel && level2 == lowLevel)
                ties.add(branch);
            else if (level1 == lowLevel && level2 == highLevel)
                ties.add(branch);
            else if (level1 == -1) {//its a three phase transformer
                if (level2 == highLevel || level2 == lowLevel)
                    for (BusData b : bus2buses.get(b1)) {
                        int otherlevel = KVLevelPicker.getKVLevel(b.getBaseVoltage());
                        if (b != b2 && otherlevel == highLevel && level2 == lowLevel)
                            ties.add(branch);
                        //else if(b != b2 && otherlevel == lowLevel && level2 == highLevel)
                        //    ties.add(branch);
                    }
            } else if (level2 == -1) {//its a three phase transformer
                if (level1 == highLevel || level1 == lowLevel)
                    for (BusData b : bus2buses.get(b2)) {
                        int otherlevel = KVLevelPicker.getKVLevel(b.getBaseVoltage());
                        if (b != b1 && otherlevel == highLevel && level1 == lowLevel)
                            ties.add(branch);
                        //else if(b != b1 && otherlevel == lowLevel && level1 == highLevel)
                        //    ties.add(branch);
                    }
            }
        }
        //step2: build sub area model
        //SubareaModel model = buildSubarea(island, ties);
        //int i = 1;
        //for(IEEEDataIsland a : model.getIslands()) {
        //Map<BusData, List<BusData>> b2b = getBus2Buses(a);
        //for(BusData b : b2b.keySet())
        //    for(BusData c : b2b.get(b)) {
        //        double baseV1 = b.getBaseVoltage();
        //        double baseV2 = c.getBaseVoltage();
        //        int level1 = KVLevelPicker.getKVLevel(baseV1);
        //        int level2 = KVLevelPicker.getKVLevel(baseV2);
        //        if (level1 == highLevel || level2 == highLevel)
        //            System.out.println(level1 + "\t" + level2);
        //        //else if (level1 == lowLevel && level2 == highLevel)
        //        //    System.out.println("pause...");
        //    }
        //IcfWriter writer = new IcfWriter(a);
        //writer.writeFile("d:\\island" + i + ".txt");
        //i++;
        //}
        //return model;
        return buildSubarea(island, ties);
    }


    /**
     * build islands acording to tie lines, setIslands and setTies is invocated in method,
     * branchId is put to map and buildMeasure is also invocated
     *
     * @param island integrate island
     * @param ties   tie lines
     */
    public static SubareaModel buildSubarea(IEEEDataIsland island, List<BranchData> ties) {
        IEEEDataIsland copy = island.clone();
        HashMap<BranchData, BranchData> new2original = new HashMap<BranchData, BranchData>(copy.getBranches().size());
        HashMap<BranchData, BranchData> original2new = new HashMap<BranchData, BranchData>(copy.getBranches().size());
        for (int i = 0; i < island.getBranches().size(); i++) {
            new2original.put(copy.getBranches().get(i), island.getBranches().get(i));
            original2new.put(island.getBranches().get(i), copy.getBranches().get(i));
        }
        copy.getBranches().removeAll(ties);
        List<IEEEDataIsland> islands = buildIsolateIslands(copy);

        HashMap<BusData, List<BranchData>> boundaryBus2tieline = new HashMap<BusData, List<BranchData>>(ties.size() * 2);
        HashMap<BranchData, IEEEDataIsland[]> tieline2island = new HashMap<BranchData, IEEEDataIsland[]>(ties.size());
        HashMap<BranchData, BusData[]> tieline2boundaryBus = new HashMap<BranchData, BusData[]>(ties.size());
        HashMap<IEEEDataIsland, List<BusData>> island2boundaryBus = new HashMap<IEEEDataIsland, List<BusData>>(ties.size());
        HashMap<IEEEDataIsland, List<BusData>> island2externalBus = new HashMap<IEEEDataIsland, List<BusData>>(ties.size());
        for (BranchData branch : ties) {
            tieline2island.put(branch, new IEEEDataIsland[2]);
            tieline2boundaryBus.put(branch, new BusData[2]);
            for (IEEEDataIsland aIsland : islands) {
                if (island2boundaryBus.get(aIsland) == null)
                    island2boundaryBus.put(aIsland, new ArrayList<BusData>());
                if (island2externalBus.get(aIsland) == null)
                    island2externalBus.put(aIsland, new ArrayList<BusData>());
                BusData tap = aIsland.getBus(branch.getTapBusNumber());
                BusData tail = aIsland.getBus(branch.getZBusNumber());
                if (tap != null) {
                    tieline2island.get(branch)[0] = aIsland;
                    tieline2boundaryBus.get(branch)[0] = tap;
                    if (!island2boundaryBus.get(aIsland).contains(tap))
                        island2boundaryBus.get(aIsland).add(tap);
                    if (!island2externalBus.get(aIsland).contains(tail))
                        island2externalBus.get(aIsland).add(tail);
                    if (boundaryBus2tieline.get(tap) == null)
                        boundaryBus2tieline.put(tap, new ArrayList<BranchData>());
                    boundaryBus2tieline.get(tap).add(branch);
                    continue;
                }
                if (tail != null) {
                    tieline2island.get(branch)[1] = aIsland;
                    tieline2boundaryBus.get(branch)[1] = tail;
                    if (!island2boundaryBus.get(aIsland).contains(tail))
                        island2boundaryBus.get(aIsland).add(tail);
                    if (!island2externalBus.get(aIsland).contains(tap))
                        island2externalBus.get(aIsland).add(tap);
                    if (boundaryBus2tieline.get(tail) == null)
                        boundaryBus2tieline.put(tail, new ArrayList<BranchData>());
                    boundaryBus2tieline.get(tail).add(branch);
                }
            }
        }
        SubareaModel model = new SubareaModel();
        model.setIslands(islands);
        model.setTies(ties);
        model.setOriginalIsland(island);
        model.setNew2original(new2original);
        model.setOriginal2new(original2new);
        model.setBoundaryBus2tieline(boundaryBus2tieline);
        model.setIsland2boundaryBus(island2boundaryBus);
        model.setIsland2externalBus(island2externalBus);
        model.setTieline2island(tieline2island);
        model.setTieline2boundaryBus(tieline2boundaryBus);
        return model;
    }

    public static SubareaModel buildSubareaByTearingBus(IEEEDataIsland island, List<BusData> tearBuses) {
        IEEEDataIsland copy = island.clone();
        HashMap<BranchData, BranchData> new2original = new HashMap<BranchData, BranchData>(copy.getBranches().size());
        HashMap<BranchData, BranchData> original2new = new HashMap<BranchData, BranchData>(copy.getBranches().size());
        for (int i = 0; i < island.getBranches().size(); i++) {
            new2original.put(copy.getBranches().get(i), island.getBranches().get(i));
            original2new.put(island.getBranches().get(i), copy.getBranches().get(i));
        }
        copy.getBuses().removeAll(tearBuses);
        //todo: not finished!
        return null;
    }

    public static void mergeSubareaes(SubareaModel area, List<IEEEDataIsland> islands) {
        IEEEDataIsland largestIsland = getLargestIsland(islands);
        for (IEEEDataIsland island : islands) {
            if (island == largestIsland || !area.getIsland2boundaryBus().containsKey(island)) continue;
            largestIsland.getBuses().addAll(island.getBuses());
            largestIsland.getBranches().addAll(island.getBranches());
            for (BusData bus : area.getIsland2boundaryBus().get(island)) {
                boolean isInnerBus = true;
                List<BranchData> tielines = area.getBoundaryBus2tieline().get(bus);
                List<BranchData> linesToRemove = new ArrayList<BranchData>(tielines.size());
                for (BranchData branch : tielines) {
                    BusData[] buses = area.getTieline2boundaryBus().get(branch);
                    BusData anotherBus = null;
                    for (BusData b : buses) {
                        if (b == bus) continue;
                        anotherBus = b;
                    }
                    IEEEDataIsland[] termialIsland = area.getTieline2island().get(branch);
                    if (termialIsland.length != 2) continue;
                    if (termialIsland[0] == largestIsland || termialIsland[1] == largestIsland) {
                        area.getIsland2boundaryBus().get(largestIsland).remove(anotherBus);//todo: another bus may connnect to the other island
                        area.getBoundaryBus2tieline().remove(anotherBus);
                        area.getIsland2externalBus().get(largestIsland).remove(bus);
                        area.getTieline2boundaryBus().remove(branch);
                        area.getTieline2island().remove(branch);
                        area.getTies().remove(branch);
                        linesToRemove.add(branch);
                    } else if (termialIsland[0] == island) {
                        termialIsland[0] = largestIsland;
                        area.getIsland2externalBus().get(largestIsland).add(anotherBus);
                        isInnerBus = false;
                    } else if (termialIsland[1] == island) {
                        termialIsland[1] = largestIsland;
                        area.getIsland2externalBus().get(largestIsland).add(anotherBus);
                        isInnerBus = false;
                    }
                }
                if (isInnerBus) {
                    area.getBoundaryBus2tieline().remove(bus);
                } else if (linesToRemove.size() > 0) {
                    area.getIsland2boundaryBus().get(largestIsland).add(bus);
                    for (BranchData branch : linesToRemove)
                        area.getBoundaryBus2tieline().get(bus).remove(branch);
                }
            }
            area.getIsland2boundaryBus().remove(island);
            area.getIsland2externalBus().remove(island);
            area.getIslands().remove(island);
        }
        largestIsland.buildBranchIndex();
    }

    public static void main(String[] args) {
        List<BusData> test = new ArrayList<BusData>();
        BusData b = new BusData();
        b.setType(2);
        b.setGenerationMW(21.0);
        test.add(b);

        b = new BusData();
        b.setType(2);
        b.setGenerationMW(10.0);
        test.add(b);

        b = new BusData();
        b.setType(0);
        test.add(b);

        BusData.setCompareOrder(new String[]{BusData.VAR_TYPE, BusData.VAR_PGEN});
        Collections.sort(test);
        for (BusData bus : test)
            System.out.println(bus.getType() + "\t" + bus.getGenerationMW());
    }
}
