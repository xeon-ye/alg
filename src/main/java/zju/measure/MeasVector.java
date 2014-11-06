package zju.measure;

import zju.matrix.AVector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-5
 */
public class MeasVector implements Serializable, MeasTypeCons {

    AVector z;

    AVector z_estimate = new AVector(0);

    AVector z_true;

    AVector sigma;

    AVector weight;

    int[] measureOrder = new int[8]; //todo: does not include transformer tap measurement

    int[] bus_a_pos = new int[0];
    int[] bus_a_phase = new int[0];
    int bus_a_index = 0;

    int[] bus_v_pos = new int[0];
    int[] bus_v_phase = new int[0];
    int bus_v_index = 0;

    int[] bus_p_pos = new int[0];
    int[] bus_p_phase = new int[0];
    int bus_p_index = 0;

    int[] bus_q_pos = new int[0];
    int[] bus_q_phase = new int[0];
    int bus_q_index = 0;

    int[] line_from_p_pos = new int[0];
    int[] line_from_p_phase = new int[0];
    int line_from_p_index = 0;

    //AVector line_from_q;
    int[] line_from_q_pos = new int[0];
    int[] line_from_q_phase = new int[0];
    int line_from_q_index = 0;

    int[] line_to_p_pos = new int[0];
    int[] line_to_p_phase = new int[0];
    int line_to_p_index = 0;

    int[] line_to_q_pos = new int[0];
    int[] line_to_q_phase = new int[0];
    int line_to_q_index = 0;

    int[] line_i_amp_pos = new int[0];
    int[] line_i_amp_phase = new int[0];
    int line_i_amp_index = 0;

    int[] line_from_i_amp_pos;
    int[] line_from_i_amp_phase;
    int line_from_i_amp_index = 0;

    int[] line_to_i_amp_pos;
    int[] line_to_i_amp_phase;
    int line_to_i_amp_index = 0;

    int[] line_from_i_a_pos;
    int[] line_from_i_a_phase;
    int line_from_i_a_index = 0;

    int[] line_to_i_a_pos;
    int[] line_to_i_a_phase;
    int line_to_i_a_index = 0;

    Collection<Integer> badMeasId;//

    Map<Integer, int[]> typeAndPos;

    public Collection<Integer> getBadMeasId() {
        if (badMeasId == null)
            badMeasId = new ArrayList<Integer>(0);
        return badMeasId;
    }

    public void setBadMeasId(Collection<Integer> badMeasId) {
        this.badMeasId = badMeasId;
    }

    public AVector getZ() {
        return z;
    }

    public void setZ(AVector z) {
        this.z = z;
    }

    public AVector getZ_estimate() {
        return z_estimate;
    }

    public void setZ_estimate(AVector z_estimate) {
        this.z_estimate = z_estimate;
    }

    public int[] getMeasureOrder() {
        return measureOrder;
    }

    public void setMeasureOrder(int[] measureOrder) {
        this.measureOrder = measureOrder;
    }

    public int[] getBus_p_pos() {
        return bus_p_pos;
    }

    public void setBus_p_pos(int[] bus_p_pos) {
        this.bus_p_pos = bus_p_pos;
    }

    public AVector getZ_true() {
        return z_true;
    }

    public void setZ_true(AVector z_true) {
        this.z_true = z_true;
    }

    public AVector getSigma() {
        return sigma;
    }

    public void setSigma(AVector sigma) {
        this.sigma = sigma;
    }

    public AVector getWeight() {
        return weight;
    }

    public void setWeight(AVector weight) {
        this.weight = weight;
    }

    public int[] getBus_a_phase() {
        return bus_a_phase;
    }

    public void setBus_a_phase(int[] bus_a_phase) {
        this.bus_a_phase = bus_a_phase;
    }

    public int[] getBus_v_phase() {
        return bus_v_phase;
    }

    public void setBus_v_phase(int[] bus_v_phase) {
        this.bus_v_phase = bus_v_phase;
    }

    public int[] getBus_p_phase() {
        return bus_p_phase;
    }

    public void setBus_p_phase(int[] bus_p_phase) {
        this.bus_p_phase = bus_p_phase;
    }

    public int[] getBus_q_phase() {
        return bus_q_phase;
    }

    public void setBus_q_phase(int[] bus_q_phase) {
        this.bus_q_phase = bus_q_phase;
    }

    public int[] getLine_from_p_phase() {
        return line_from_p_phase;
    }

    public void setLine_from_p_phase(int[] line_from_p_phase) {
        this.line_from_p_phase = line_from_p_phase;
    }

    public int[] getLine_from_q_phase() {
        return line_from_q_phase;
    }

    public void setLine_from_q_phase(int[] line_from_q_phase) {
        this.line_from_q_phase = line_from_q_phase;
    }

    public int[] getLine_to_p_phase() {
        return line_to_p_phase;
    }

    public void setLine_to_p_phase(int[] line_to_p_phase) {
        this.line_to_p_phase = line_to_p_phase;
    }

    public int[] getLine_to_q_phase() {
        return line_to_q_phase;
    }

    public void setLine_to_q_phase(int[] line_to_q_phase) {
        this.line_to_q_phase = line_to_q_phase;
    }

    public int[] getLine_i_amp_phase() {
        return line_i_amp_phase;
    }

    public void setLine_i_amp_phase(int[] line_i_amp_phase) {
        this.line_i_amp_phase = line_i_amp_phase;
    }

    public int[] getBus_a_pos() {
        return bus_a_pos;
    }

    public void setBus_a_pos(int[] bus_a_pos) {
        this.bus_a_pos = bus_a_pos;
    }

    public int[] getBus_v_pos() {
        return bus_v_pos;
    }

    public void setBus_v_pos(int[] bus_v_pos) {
        this.bus_v_pos = bus_v_pos;
    }

    public int[] getBus_q_pos() {
        return bus_q_pos;
    }

    public void setBus_q_pos(int[] bus_q_pos) {
        this.bus_q_pos = bus_q_pos;
    }

    public int[] getLine_from_p_pos() {
        return line_from_p_pos;
    }

    public void setLine_from_p_pos(int[] line_from_p_pos) {
        this.line_from_p_pos = line_from_p_pos;
    }

    public int[] getLine_from_q_pos() {
        return line_from_q_pos;
    }

    public void setLine_from_q_pos(int[] line_from_q_pos) {
        this.line_from_q_pos = line_from_q_pos;
    }

    public int[] getLine_to_p_pos() {
        return line_to_p_pos;
    }

    public void setLine_to_p_pos(int[] line_to_p_pos) {
        this.line_to_p_pos = line_to_p_pos;
    }

    public int[] getLine_to_q_pos() {
        return line_to_q_pos;
    }

    public void setLine_to_q_pos(int[] line_to_q_pos) {
        this.line_to_q_pos = line_to_q_pos;
    }

    public int[] getLine_i_amp_pos() {
        return line_i_amp_pos;
    }

    public void setLine_i_amp_pos(int[] line_i_amp_pos) {
        this.line_i_amp_pos = line_i_amp_pos;
    }

    public int[] getLine_from_i_amp_pos() {
        return line_from_i_amp_pos;
    }

    public void setLine_from_i_amp_pos(int[] line_from_i_amp_pos) {
        this.line_from_i_amp_pos = line_from_i_amp_pos;
    }

    public int[] getLine_from_i_amp_phase() {
        return line_from_i_amp_phase;
    }

    public void setLine_from_i_amp_phase(int[] line_from_i_amp_phase) {
        this.line_from_i_amp_phase = line_from_i_amp_phase;
    }

    public int[] getLine_to_i_amp_pos() {
        return line_to_i_amp_pos;
    }

    public void setLine_to_i_amp_pos(int[] line_to_i_amp_pos) {
        this.line_to_i_amp_pos = line_to_i_amp_pos;
    }

    public int[] getLine_to_i_amp_phase() {
        return line_to_i_amp_phase;
    }

    public void setLine_to_i_amp_phase(int[] line_to_i_amp_phase) {
        this.line_to_i_amp_phase = line_to_i_amp_phase;
    }

    public int[] getLine_from_i_a_pos() {
        return line_from_i_a_pos;
    }

    public void setLine_from_i_a_pos(int[] line_from_i_a_pos) {
        this.line_from_i_a_pos = line_from_i_a_pos;
    }

    public int[] getLine_from_i_a_phase() {
        return line_from_i_a_phase;
    }

    public void setLine_from_i_a_phase(int[] line_from_i_a_phase) {
        this.line_from_i_a_phase = line_from_i_a_phase;
    }

    public int[] getLine_to_i_a_pos() {
        return line_to_i_a_pos;
    }

    public void setLine_to_i_a_pos(int[] line_to_i_a_pos) {
        this.line_to_i_a_pos = line_to_i_a_pos;
    }

    public int[] getLine_to_i_a_phase() {
        return line_to_i_a_phase;
    }

    public void setLine_to_i_a_phase(int[] line_to_i_a_phase) {
        this.line_to_i_a_phase = line_to_i_a_phase;
    }

    public int getBus_a_index() {
        return bus_a_index;
    }

    public int getBus_v_index() {
        return bus_v_index;
    }

    public int getBus_p_index() {
        return bus_p_index;
    }

    public int getBus_q_index() {
        return bus_q_index;
    }

    public int getLine_from_p_index() {
        return line_from_p_index;
    }

    public int getLine_from_q_index() {
        return line_from_q_index;
    }

    public int getLine_to_p_index() {
        return line_to_p_index;
    }

    public int getLine_to_q_index() {
        return line_to_q_index;
    }

    public int getLine_i_amp_index() {
        return line_i_amp_index;
    }

    public int getLine_from_i_amp_index() {
        return line_from_i_amp_index;
    }

    public int getLine_to_i_amp_index() {
        return line_to_i_amp_index;
    }

    public int getLine_from_i_a_index() {
        return line_from_i_a_index;
    }

    public int getLine_to_i_a_index() {
        return line_to_i_a_index;
    }

    public void setBus_a_index(int bus_a_index) {
        this.bus_a_index = bus_a_index;
    }

    public void setBus_v_index(int bus_v_index) {
        this.bus_v_index = bus_v_index;
    }

    public void setBus_p_index(int bus_p_index) {
        this.bus_p_index = bus_p_index;
    }

    public void setBus_q_index(int bus_q_index) {
        this.bus_q_index = bus_q_index;
    }

    public void setLine_from_p_index(int line_from_p_index) {
        this.line_from_p_index = line_from_p_index;
    }

    public void setLine_from_q_index(int line_from_q_index) {
        this.line_from_q_index = line_from_q_index;
    }

    public void setLine_to_p_index(int line_to_p_index) {
        this.line_to_p_index = line_to_p_index;
    }

    public void setLine_to_q_index(int line_to_q_index) {
        this.line_to_q_index = line_to_q_index;
    }

    public void setLine_i_amp_index(int line_i_amp_index) {
        this.line_i_amp_index = line_i_amp_index;
    }

    public void setLine_from_i_amp_index(int line_from_i_amp_index) {
        this.line_from_i_amp_index = line_from_i_amp_index;
    }

    public void setLine_to_i_amp_index(int line_to_i_amp_index) {
        this.line_to_i_amp_index = line_to_i_amp_index;
    }

    public void setLine_from_i_a_index(int line_from_i_a_index) {
        this.line_from_i_a_index = line_from_i_a_index;
    }

    public void setLine_to_i_a_index(int line_to_i_a_index) {
        this.line_to_i_a_index = line_to_i_a_index;
    }
}
