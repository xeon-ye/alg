package zju.hems;

import java.util.Map;

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
