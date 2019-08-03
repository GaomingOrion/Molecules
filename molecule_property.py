import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

structures = pd.read_csv('./input/structures.csv')
atomic_radii = dict(C=0.77, F=0.71, H=0.38, N=0.75, O=0.73)
atomic_electron = dict(C=2.5, F=3.98, H=2.2, N=3.04, O=3.44)
structures['radii'] = structures['atom'].map(atomic_radii)
structures['electron'] = structures['atom'].map(atomic_electron)
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
        self.electron = self.molecule['electron'].values
        # atom type
        self.atoms_dict = defaultdict(list)
        for i, atom in enumerate(self.atoms):
            self.atoms_dict[atom].append(i)
        # calculate distance matrix and bulid graph
        self.dist_matrix = self.calc_distance()
        self.bond_distance = self.radii[:, None] + self.radii
        self.graph_edges = (self.dist_matrix < 1.3*self.bond_distance) - np.identity(self.num)
        self.graph = nx.Graph(self.graph_edges)
        self.graph_shortest_path = dict(nx.all_pairs_shortest_path(self.graph))
        self.graph_shortest_path_length = {x: {i: len(path)-1 for i, path in v.items()}
                                           for x, v in self.graph_shortest_path.items()}

    def calc_distance(self):
        d = self.coordinates[:, None, :] - self.coordinates
        return np.sqrt(np.einsum('ijk,ijk->ij', d, d))

    def calc_cos(self, a, b, c):
        '''
        cosine of angle abc
        '''
        ba = self.coordinates[a] - self.coordinates[b]
        bc = self.coordinates[c] - self.coordinates[b]
        return np.dot(ba,bc)/(np.linalg.norm(ba)*(np.linalg.norm(bc)))

    def calc_dihedral_angle(self, a, p1, p2, b):
        '''
        dihedral angel between ap_1p_2 and bp_1p_2
        '''
        cos0 = self.calc_cos(a, p1, b)
        cos1 = self.calc_cos(a, p1, p2)
        cos2 = self.calc_cos(b, p1, p2)
        return (cos0 - cos1*cos2)/np.sqrt(1 - cos1*cos1)/np.sqrt(1 - cos2*cos2)

    def get_hydrogen_bonds(self):
        '''
        get all 1JHC, 1JHN, 2JHH, 2JHC, 2JHN, 3JHH, 3JHC, 3JHN pairs
        :return: dict. key: '1JHC' ..., value: pairs list
        '''
        hydrogen_bonds = defaultdict(list)
        for start in self.atoms_dict['H']:
            node_dict = {1: [], 2: [], 3: []}
            for node, path_length in self.graph_shortest_path_length[start].items():
                if path_length in node_dict:
                    node_dict[path_length].append(node)
            for i, nodes in node_dict.items():
                for node in nodes:
                    node_type = self.atoms[node]
                    if node_type in {'C', 'N'} or (node_type == 'H' and node > start):
                        hydrogen_bonds[f'{i}JH{node_type}'].append([start, node])
        return hydrogen_bonds

    def get_molecule_property(self):
        res = self.get_subgraph_property(list(range(self.num)), 'molecule')
        res['molecule#cycle_basis_num'] = nx

    def get_subgraph_property(self, atoms_idx, name_space):
        '''
        :param atoms_idx: subgraph node indices
        :param name_space: name space
        :return: property dict
        '''
        res = {}
        res[f'{name_space}#nodes_num'] = len(atoms_idx)
        s = np.cov(mp.coordinates[atoms_idx, :].T)[0]
        eigen_ratio = np.cumsum(s)/np.sum(s)
        res[f'{name_space}#eigen_1d'] = eigen_ratio[0]
        res[f'{name_space}#eigen_2d'] = eigen_ratio[1]
        res[f'{name_space}#max_distance'] = np.max(self.dist_matrix[atoms_idx, atoms_idx])
        # notH_idx = [i for i in atoms_idx if self.atoms[i] != 'H']
        # s = np.cov(mp.coordinates[notH_idx, :].T)[0]
        # eigen_ratio = np.cumsum(s)/np.sum(s)
        # res[f'{name_space}_notH#eigen_1d'] = eigen_ratio[0]
        # res[f'{name_space}_notH#eigen_2d'] = eigen_ratio[1]
        # res[f'{name_space}_notH#max_distance'] = np.max(self.dist_matrix[notH_idx, notH_idx])
        subgraph = nx.Graph(self.graph_edges[atoms_idx, atoms_idx])
        cycle_basis = nx.cycle_basis(subgraph)
        res[f'{name_space}#cycle_basis_num'] = len(cycle_basis)
        res[f'{name_space}#wiener_index'] = nx.wiener_index(subgraph)
        res[f'{name_space}#algebraic_connectivity'] = nx.algebraic_connectivity(subgraph)
        res[f'{name_space}#algebraic_connectivity_normalized'] = nx.algebraic_connectivity(subgraph, normalized=True)
        return res

    def get_common_property(self, H, x):
        '''
        :param H: hydrogen index
        :param x: the other atom index
        :return: property dict
        '''
        res = {}
        res['distance'] = self.dist_matrix[H][x]
        res['x_bonds'] = len(self.graph[x])
        # graph neighbor property
        path_nodes = self.graph_shortest_path[H][x]
        path_neighbor1 = {}
        for i in path_nodes:
            path_neighbor1.update(set(self.graph[i]))
        res.update(self.get_subgraph_property(list(path_neighbor1), 'path_neighbor1'))
        path_neighbor2 = path_neighbor1.copy()
        for i in path_neighbor1:
            path_neighbor2.update(set(self.graph[i]))
        res.update(self.get_subgraph_property(list(path_neighbor2), 'path_neighbor2'))
        # space neighbor property
        return

    def get_1JHx_property(self, H, x):
        res = {}


    def main(self):
        return

if __name__ == '__main__':
    molecule = molecule_names[100]
    mp = MoleculeProperty(molecule)
    s = np.linalg.eig(np.cov(mp.coordinates.T))
    nx.draw(mp.graph)
    #train = pd.read_csv('./input/train.csv')

