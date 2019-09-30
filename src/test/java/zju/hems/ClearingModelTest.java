package zju.hems;

import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ClearingModelTest extends TestCase {

    public void testCase1() {
        Offer offer1 = new Offer("1", 1, 1.5, 0.2);
        Offer offer2 = new Offer("2", 2, 0.8, 0.5);
        Offer offer3 = new Offer("3", 3, 1, 0.3);
        Map<String, Double> timeShaveCapRatio = new HashMap<>();
        timeShaveCapRatio.put("1", 0.2);
        timeShaveCapRatio.put("2", 0.5);
        timeShaveCapRatio.put("3", 0.3);
        List<Offer> offers = new ArrayList<>(3);
        offers.add(offer1);
        offers.add(offer2);
        offers.add(offer3);
        ClearingModel clearingModel = new ClearingModel(offers, 5, timeShaveCapRatio);
        clearingModel.clearing();
        Map<String, Double> bidRatios = clearingModel.getBidRatios();
        double clearingPrice = clearingModel.getClearingPrice();
        System.out.println(clearingPrice);
        for (String key : bidRatios.keySet()) {
            System.out.println(key + "\t" + bidRatios.get(key));
        }
    }
}