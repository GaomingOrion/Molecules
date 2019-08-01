import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

structures = pd.read_csv('./input/structures.csv')
atomic_radii = dict(C=0.77, F=0.71, H=0.38, N=0.75, O=0.73)
structures['radii'] = structures['atom'].map(atomic_radii)
structures_group = structures.groupby('molecule_name')
molecule_names = structures['molecule_name'].unique()
class MoleculeProperty:
    def __init__(self, molecule_name):
        self.molecule = structures_group.get_group(molecule_name)
        self.coordinates = self.molecule[['x', 'y', 'z']].values
        # self.x_coordinates = self.coordinates[:, 0]
        # self.y_coordinates = self.coordinates[:, 1]
        # self.z_coordinates = self.coordinates[:, 2]
        self.atoms = self.molecule['atom'].tolist()
        self.num = len(self.atoms)
        self.radii = self.molecule['radii'].values
        # atom type
        self.atoms_dict = defaultdict(list)
        for i, atom in enumerate(self.atoms):
            self.atoms_dict[atom].append(i)
        # calculate distance matrix and bulid graph
        self.dist_matrix = self.calc_distance()
        self.bond_distance = self.radii[:, None] + self.radii
        self.graph = nx.Graph(self.dist_matrix < 1.3*self.bond_distance)


    def calc_distance(self):
        d = self.coordinates[:, None, :] - self.coordinates
        return np.sqrt(np.einsum('ijk,ijk->ij', d, d))

    def get_hydrogen_bonds(self):
        hydrogen_bonds = defaultdict(list)
        for start in self.atoms_dict['H'][:]:
            queue = [start]
            visited = {start}
            for i in range(3):
                tmp = []
                while queue:
                    for x1 in self.graph[queue.pop()]:
                        if x1 not in visited:
                            if self.atoms[x1] in {'C', 'N'} or (self.atoms[x1] == 'H' and x1 > start):
                                hydrogen_bonds[f'{i+1}JH{self.atoms[x1]}'].append([start, x1])
                            tmp.append(x1)
                            visited.add(x1)
                queue = tmp
        return hydrogen_bonds




