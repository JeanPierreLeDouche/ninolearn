#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:51:05 2020

@author: ivo
"""


from igraph import *
import igraph

g = Graph()
g.add_vertices(3)

g.add_edges([(0,1), (1,2)])

g.add_edges([(2,0)])
g.add_vertices(3)
g.add_edges([(2,3), (3,4), (4,5), (5,3)])
print(g)
eid = g.get_eid(2,3)
print(f'edge ID: {eid}')
g.delete_edges(eid)
summary(g)

#%%

g = Graph.Tree(127,2)
summary(g)

g2 = Graph.Tree(127,2)
print(g.get_edgelist() == g2.get_edgelist())

g = Graph.GRG(100, 0.2)
summary(g)

g2 = Graph.GRG(100, 0.2)
summary(g)

#%%

g = Graph([(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)])
g.vs["name"] = ["Alice", "Bob", "Claire", "Dennis", "Esther", "Frank", "George"]
g.vs["age"] = [25, 31, 18, 47, 22, 23, 50]
g.vs["gender"] = ["f", "m", "f", "m", "f", "m", "m"]
g.es["is_formal"] = [False, False, True, True, True, False, True, False, False]

g["date"] = "2009-01-10"
print(g['date'])

g.vs[3]["foo"] = "bar"
print(g.vs["foo"])
del g.vs["foo"]
# g.vs["foo"] # intentionally gives error

print(g.degree())
print(g.degree(6))

### quit at querying vertices and edges based on attributes 
# https://igraph.org/python/doc/tutorial/tutorial.html



