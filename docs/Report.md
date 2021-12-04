# 2.1)

For a random surfer that start his journey on a random node and with a dumping facotr at 0.85, he have these chances to finish at each node:

```
Node 1 : 0.051705016944720766
Node 2 : 0.07367973392063928
Node 3 : 0.05741272838222149
Node 4 : 0.3487031406768242
Node 5 : 0.19990368078312204
Node 6 : 0.2685956992924724
```

This stability is reached after 23 steps

Now, if we change the dumping factor to 0.95, the surfer have these chances to finish at each node:

```
Node 1 : 0.0184753060679744
Node 2 : 0.02729497432834562
Node 3 : 0.020705449096631277
Node 4 : 0.40975518227695357
Node 5 : 0.21408109341351664
Node 6 : 0.30968799481657827
```

This stability is reached after 27 steps

In comparison, degree centrelity gives us:

```
Node 1 : 0.15
Node 2 : 0.1
Node 3 : 0.2
Node 4 : 0.2
Node 5 : 0.2
Node 6 : 0.15
```

# 2.2

For ChaiRank, alway with alpha = 0.85, we got this:

```
Node 1 : 0.35509565140606913
Node 2 : 0.025000000000000012
Node 3 : 0.3758472962936703
Node 4 : 0.09033280507220202
Node 5 : 0.09033280507220204
Node 6 : 0.06339144215585703
```

This stability is reached after 68 steps