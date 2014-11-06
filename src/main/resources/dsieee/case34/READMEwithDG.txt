"case34 with DG" is based on the paper "Induction Machine Test Case for the 34 Bus Test Feeder-Description".
In this case, two more transformers and induction generators are added to the primary IEEE 34 Node test feeder.
To locate these new items, two more nodes (i.e 891 and 892) are newly named.(T1 is between node 848 and 891 while T2 is
between 890 and 892. Thus G1 is connected to 891 while G2 is connected to 892)
In this case, the Line Segment Data should be modified with the data below:
848 891 0   TG1
890 892 0   TG2
When it comes to the distributed generations, Node B is set to be 0.
New types named "Distributed Generation" of items should be added.
For instance,G1 is expresses as below:
891   IM   Gr.Y-Gr.Y  660   0.48    0.0053  0.106   0.007   4.0
If the type is IM which means induction machine, then the meaning of the data as follows:
Node Type   Connection  PowerOut(kW)    RatedVoltage(kV,line to line)    Rs(pu)  Xs(pu)  Rr(pu)  Xr(pu)  Xm(pu)
If the type is constant PQ, PI, PV,Z etc. which means the DG could be regarded as negative loads.
And it could be described like this:
900  LD  Y-PQ   20	16	20	16	20	16
Name Type   Connection  Ph1.P Ph1.Q Ph2.P Ph2.Q Ph3.P Ph3.Q
Just like the Spot loads.(Maybe needs discussed when the model is PV or PI).


