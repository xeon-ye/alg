package zju.topo;

import com.mxgraph.swing.mxGraphComponent;
import com.mxgraph.view.mxGraph;
import org.jgrapht.UndirectedGraph;
import zju.devmodel.MapObject;
import zju.dsmodel.DsConnectNode;

import javax.swing.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static zju.dsmodel.DsModelCons.KEY_CONNECTED_NODE;

public class TopoGraph extends JFrame
{

	/**
	 * 
	 */
	private static final long serialVersionUID = -2707712944901661771L;

	public TopoGraph(List<TopoNode> topoNodesList, UndirectedGraph<DsConnectNode, MapObject> origGraph){
		super("DsTopo");
		mxGraph graph = new mxGraph();
		Object parent = graph.getDefaultParent();

		graph.getModel().beginUpdate();

		Map<String, Object> verticesMap = new HashMap<>();

		try {
			for (int i = 0; i < topoNodesList.size(); i++) {
				TopoNode node = topoNodesList.get(i);
				Object vertex = graph.insertVertex(parent, null, node.getName(), node.getxCoordinate(), node.getyCoordinate(), 5, 5, "shape=ellipse");
				verticesMap.put(node.getName(), vertex);
			}

			for (MapObject edge : origGraph.edgeSet()) {
				String[] s = edge.getProperty(KEY_CONNECTED_NODE).split(";");
				Object vertex1 = verticesMap.get(s[0]);
				Object vertex2 = verticesMap.get(s[1]);
				graph.insertEdge(parent, null, null, vertex1, vertex2,"endArrow=null");
			}

		} finally {
			graph.getModel().endUpdate();
		}
		mxGraphComponent graphComponent = new mxGraphComponent(graph);
		getContentPane().add(graphComponent);

	}
}
