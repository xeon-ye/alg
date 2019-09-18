package zju.hems;

import java.util.Map;

/**
 * 微网.
 * @author Xu Chengsi
 * @date 2019/7/29
 */
public class Microgrid {

    Map<String, User> users;

    public Microgrid(Map<String, User> users) {
        this.users = users;
    }

    public Map<String, User> getUsers() {
        return users;
    }

    public void setUsers(Map<String, User> users) {
        this.users = users;
    }
}
