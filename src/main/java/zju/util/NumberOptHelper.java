package zju.util;

import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;

import java.util.*;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-2
 */
public class NumberOptHelper {
    Map<Integer, Integer> old2new;
    Map<Integer, Integer> new2old;
    private int max = 1;//the max radical node number that has been used

    public static Map<Integer, List<Integer>> getConnectedBranches(IEEEDataIsland island) {
        Map<Integer, List<Integer>> bus2branches = new HashMap<Integer, List<Integer>>(island.getBuses().size());
        for (BranchData branch : island.getBranches()) {
            int head = branch.getTapBusNumber();
            int tail = branch.getZBusNumber();
            if (!bus2branches.containsKey(head)) {
                bus2branches.put(head, new ArrayList<Integer>());
            }
            bus2branches.get(head).add(branch.getId());
            if (!bus2branches.containsKey(tail)) {
                bus2branches.put(tail, new ArrayList<Integer>());
            }
            bus2branches.get(tail).add(branch.getId());
        }
        return bus2branches;
    }

    public Map<Integer, Integer> tinney2(IEEEDataIsland island) {
        // todo 2
        Map<Integer, Integer> old2new = new HashMap<Integer, Integer>(island.getBuses().size());
        Map<Integer, Integer> new2old = new HashMap<Integer, Integer>(island.getBuses().size());

        Map<Integer, List<Integer>> bus2buses = IcfDataUtil.getConnectedBuses(island);
        List<Integer> residual = new ArrayList<Integer>(island.getBuses().size());
        for (int i = 0; i < island.getBuses().size(); i++) {
            residual.add(island.getBuses().get(i).getBusNumber());
        }
        for (int i = 0; i < island.getBuses().size(); i++) {
            int least = 0;
            //find least fitness
            for (int j = 1; j < residual.size(); j++) {
                if (bus2buses.get(residual.get(j)).size() < bus2buses.get(residual.get(least)).size())
                    least = j;
            }
            int oldBusNum = residual.get(least);
            List<Integer> connectedBuses = bus2buses.get(oldBusNum);
            int connectedNum = connectedBuses.size();
            for (int j = 0; j < connectedNum - 1; j++) {
                int abus = connectedBuses.get(j);
                //bus2buses.get(connectedBuses.get(j));
                for (int k = j + 1; k < connectedBuses.size(); k++) {
                    int b = connectedBuses.get(k);
                    if (!bus2buses.get(b).contains(new Integer(abus))) {
                        bus2buses.get(b).add(abus);
                        bus2buses.get(abus).add(b);
                    }
                }
            }
            for (Integer connectedBuse : connectedBuses) {
                bus2buses.get(connectedBuse).remove(new Integer(oldBusNum));
            }
            old2new.put(oldBusNum, i + 1);
            new2old.put(i + 1, oldBusNum);
            residual.remove(least);
        }
        setOld2new(old2new);
        setNew2old(new2old);
        return old2new;
    }

    //todo: tests are needed!
    public Map<Integer, Integer> radicalBusNumOpt(IEEEDataIsland island, int rootBusNumber) {
        Map<Integer, Integer> old2new = new HashMap<Integer, Integer>(island.getBuses().size());
        Map<Integer, Integer> new2old = new HashMap<Integer, Integer>(island.getBuses().size());

        Map<Integer, List<Integer>> bus2buses = IcfDataUtil.getConnectedBuses(island);
        int root = rootBusNumber;
        if (rootBusNumber == -1)
            root = island.getSlackBusNum();
        max = 1;
        old2new.put(root, max);
        new2old.put(max, root);
        max++;
        radicalNumber(bus2buses, old2new, new2old, root);

        setOld2new(old2new);
        setNew2old(new2old);
        return old2new;
    }

    private void radicalNumber(Map<Integer, List<Integer>> bus2buses,
                               Map<Integer, Integer> old2new, Map<Integer, Integer> new2old, int currentNode) {
        List<Integer> son = bus2buses.get(currentNode);
        List<Integer> toRemove = new ArrayList<Integer>();
        for (int i : son) {
            if (old2new.containsKey(i)) {
                toRemove.add(i);
            }
        }
        son.removeAll(toRemove);

        for (int i : son) {
            if (!old2new.containsKey(i)) {
                old2new.put(i, max);
                new2old.put(max, i);
                max++;
            }
        }
        for (int i : son) {
            radicalNumber(bus2buses, old2new, new2old, i);
        }
    }

    public Map<Integer, Integer> getOld2new() {
        return old2new;
    }

    public Map<Integer, Integer> getNew2old() {
        return new2old;
    }

    public void setOld2new(Map<Integer, Integer> old2new) {
        this.old2new = old2new;
    }

    public void setNew2old(Map<Integer, Integer> new2old) {
        this.new2old = new2old;
    }

    public void trans(IEEEDataIsland island) {
        trans(island, old2new);
    }

    private void trans(IEEEDataIsland island, Map<Integer, Integer> old2new) {
        for (BusData bus : island.getBuses())
            bus.setBusNumber(old2new.get(bus.getBusNumber()));
        for (BranchData branch : island.getBranches()) {
            int head = branch.getTapBusNumber();
            int tail = branch.getZBusNumber();
            branch.setTapBusNumber(old2new.get(head));
            branch.setZBusNumber(old2new.get(tail));
        }
        //IcfDataUtil.sortByBusData(island, new String[]{BusData.VAR_NUMBER});
    }

    public Map<Integer, Integer> tinney1(IEEEDataIsland island) {
        Map<Integer, Integer> old2new = new HashMap<Integer, Integer>(island.getBuses().size());
        Map<Integer, Integer> new2old = new HashMap<Integer, Integer>(island.getBuses().size());

        final Map<Integer, List<Integer>> bus2buses = IcfDataUtil.getConnectedBuses(island);
        List<Integer> keys = new ArrayList<Integer>(bus2buses.keySet());

        Collections.sort(keys, new Comparator<Integer>() {
            public int compare(Integer o1, Integer o2) {
                return new Integer(bus2buses.get(o1).size()).compareTo(bus2buses.get(o2).size());
            }
        });
        int i = 1;
        for (Integer key : keys) {
            old2new.put(key, i);
            new2old.put(i, key);
            i++;
        }
        setOld2new(old2new);
        setNew2old(new2old);
        return old2new;
    }

    public Map<Integer, Integer> simple(IEEEDataIsland island) {
        Map<Integer, Integer> old2new = new HashMap<Integer, Integer>(island.getBuses().size());
        Map<Integer, Integer> new2old = new HashMap<Integer, Integer>(island.getBuses().size());
        int i = 1;
        for (BusData bus : island.getBuses()) {
            old2new.put(bus.getBusNumber(), i);
            new2old.put(i, bus.getBusNumber());
            i++;
        }
        setOld2new(old2new);
        setNew2old(new2old);
        return old2new;
    }

    /**
     * 该方法按照PQ节点、PV节点、平衡节点的顺序进行排序
     *
     * @param island 电气岛
     * @return 新旧编号的对应关系
     */
    public Map<Integer, Integer> simple2(IEEEDataIsland island) {
        Map<Integer, Integer> old2new = new HashMap<Integer, Integer>(island.getBuses().size());
        Map<Integer, Integer> new2old = new HashMap<Integer, Integer>(island.getBuses().size());
        int i = 1;
        int j = island.getPqBusSize() + 1;
        for (BusData bus : island.getBuses()) {
            int type = bus.getType();
            switch (type) {
                case BusData.BUS_TYPE_LOAD_PQ://
                case BusData.BUS_TYPE_GEN_PQ://
                    old2new.put(bus.getBusNumber(), i);
                    new2old.put(i, bus.getBusNumber());
                    i++;
                    break;
                case BusData.BUS_TYPE_GEN_PV://
                    old2new.put(bus.getBusNumber(), j);
                    new2old.put(j, bus.getBusNumber());
                    j++;
                    break;
                case BusData.BUS_TYPE_SLACK://
                    old2new.put(bus.getBusNumber(), island.getPqBusSize() + island.getPvBusSize() + 1);
                    new2old.put(island.getPqBusSize() + island.getPvBusSize() + 1, bus.getBusNumber());
                    break;
                default://
                    break;
            }
        }
        setOld2new(old2new);
        setNew2old(new2old);
        return old2new;
    }

    public void revert(IEEEDataIsland island) {
        trans(island, new2old);
    }
}
