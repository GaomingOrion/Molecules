import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing.dummy import Pool as ThreadPool
import pickle
from collections import defaultdict


structures = pd.read_csv('./input/structures.csv')
atomic_radii = dict(C=0.77, F=0.71, H=0.38, N=0.75, O=0.73)
atomic_electron = dict(C=2.5, F=3.98, H=2.2, N=3.04, O=3.44)
structures['radii'] = structures['atom'].map(atomic_radii)
structures['electron'] = structures['atom'].map(atomic_electron)
structures_group = structures.groupby('molecule_name')
molecule_names = structures['molecule_name'].unique()

class MoleculeFeatures:
    def __init__(self, molecule_name):
        self.molecule = structures_group.get_group(molecule_name)
        self.coordinates = self.molecule[['x', 'y', 'z']].values
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

        self.func_map = {'1JHC': self.get_1JHx_features, '1JHN': self.get_1JHx_features,
                         '2JHC': self.get_2JHx_features, '2JHN': self.get_2JHx_features, '2JHH': self.get_2JHH_features,
                         '3JHC': self.get_3JHx_features, '3JHN': self.get_3JHx_features, '3JHH': self.get_3JHH_features,
                         }


    def calc_distance(self):
        d = self.coordinates[:, None, :] - self.coordinates
        return np.sqrt(np.einsum('ijk,ijk->ij', d, d))

    def calc_cos(self, a, b, c):
        '''
        cosine of angle a-b-c
        '''
        ba = self.coordinates[a] - self.coordinates[b]
        bc = self.coordinates[c] - self.coordinates[b]
        return np.dot(ba,bc)/(np.linalg.norm(ba)*(np.linalg.norm(bc)))

    def calc_dihedral_angle(self, a, p1, p2, b):
        '''
        dihedral angel between a-p1-p2 and b-p1-p2
        '''
        cos0 = self.calc_cos(a, p1, b)
        cos1 = self.calc_cos(a, p1, p2)
        cos2 = self.calc_cos(b, p1, p2)
        if abs(abs(cos1)-1) < 1e-6 or abs(abs(cos2)-1) < 1e-6:
            return 1
        return (cos0 - cos1*cos2)/np.sqrt(1 - cos1*cos1)/np.sqrt(1 - cos2*cos2)

    def get_all_couplings(self):
        '''
        get all 1JHC, 1JHN, 2JHH, 2JHC, 2JHN, 3JHH, 3JHC, 3JHN couplings
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


    def get_subgraph_features(self, atoms_idx, name_space):
        '''
        :param atoms_idx: subgraph node indices
        :param name_space: name space
        :return: features dict
        '''
        res = {}
        res[f'{name_space}#atoms_num'] = len(atoms_idx)
        res[f'{name_space}#electronegativity_sum'] = np.sum(self.electron[atoms_idx])
        atoms_num_dict = {'C':0, 'H':0, 'O':0, 'N':0, 'F':0}
        for i in atoms_idx:
            atoms_num_dict[self.atoms[i]] += 1
        for atom_type, num in atoms_num_dict.items():
            res[f'{name_space}#{atom_type}_num'] = num
        s = np.linalg.eigvalsh(np.cov(self.coordinates[atoms_idx, :].T))[::-1]
        eigen_ratio = np.cumsum(s)/np.sum(s)
        res[f'{name_space}#eigen_ratio_1d'] = eigen_ratio[0]
        res[f'{name_space}#eigen_ratio_2d'] = eigen_ratio[1]
        sub_dist_matrix = self.dist_matrix[atoms_idx][:, atoms_idx]
        res[f'{name_space}#bond_length_max'] = np.max(sub_dist_matrix)

        subgraph = nx.Graph(self.graph_edges[atoms_idx][:, atoms_idx])
        res[f'{name_space}#edges_num'] = len(subgraph.edges)
        cycle_basis = nx.cycle_basis(subgraph)
        res[f'{name_space}#cycle_basis_num'] = len(cycle_basis)
        res[f'{name_space}#triangle_num'] = sum(len(cycle) == 3 for cycle in cycle_basis)
        res[f'{name_space}#wiener_index'] = nx.wiener_index(subgraph)
        res[f'{name_space}#algebraic_connectivity'] = \
            np.linalg.eigvalsh(nx.laplacian_matrix(subgraph).toarray())[1]
        res[f'{name_space}#algebraic_connectivity_normalized'] = \
            np.linalg.eigvalsh(nx.normalized_laplacian_matrix(subgraph).toarray())[1]

        bond_length = [sub_dist_matrix[i1, i2] for i1, i2 in subgraph.edges]
        res[f'{name_space}#bond_length_mean'] = np.mean(bond_length)
        res[f'{name_space}#bond_length_max'] = np.max(bond_length)
        res[f'{name_space}#bond_length_min'] = np.min(bond_length)
        res[f'{name_space}#bond_length_std'] = np.std(bond_length)
        return res

    def get_molecule_features(self):
        return self.get_subgraph_features(list(range(self.num)), 'molecule')

    def get_common_features(self, H, x):
        '''
        :param H: hydrogen index
        :param x: the other atom index
        :return: features dict
        '''
        res = {}
        res['distance'] = self.dist_matrix[H][x]
        # graph neighbor features
        path_nodes = self.graph_shortest_path[H][x]
        path_neighbor1 = set()
        for i in path_nodes:
            path_neighbor1.update(set(self.graph[i]))
        res.update(self.get_subgraph_features(list(path_neighbor1), 'path_neighbor1'))
        path_neighbor2 = path_neighbor1.copy()
        for i in path_neighbor1:
            path_neighbor2.update(set(self.graph[i]))
        res.update(self.get_subgraph_features(list(path_neighbor2), 'path_neighbor2'))
        # space neighbor features
        return res

    def get_neighbor_features(self, x, name_space):
        res = {}
        res[f'{name_space}#x_bonds'] = len(self.graph[x])
        atoms_num_dict = {'C':0, 'H':0, 'O':0, 'N':0, 'F':0}
        electron_sum = 0
        for i in self.graph[x]:
            atoms_num_dict[self.atoms[i]] += 1
            electron_sum += self.electron[i]
        res[f'{name_space}#electronegativity_sum'] = electron_sum
        for atom_type, num in atoms_num_dict.items():
            res[f'{name_space}#{atom_type}_num'] = num
        return res


    def get_1JHx_features(self, H, x):
        res = {}
        res['idx1#neighbor_length_mean'] = np.mean(self.dist_matrix[x, self.graph[x]])
        return res
        
    def get_2JHx_features(self, H, x):
        res = {}
        path_nodes = self.graph_shortest_path[H][x]
        middle_idx = path_nodes[1]
        res['bond_length_H_middle'] = self.dist_matrix[H, middle_idx]
        res['bond_length_middle_x'] = self.dist_matrix[middle_idx, x]
        res.update(self.get_neighbor_features(middle_idx, 'middle'))
        res.update(self.get_neighbor_features(x, 'idx1'))
        res['angle_cos'] = self.calc_cos(H, middle_idx, x)
        return res
 
    def get_2JHH_features(self, H1, H2):
        res = {}
        path_nodes = self.graph_shortest_path[H1][H2]
        middle_idx = path_nodes[1]
        res['bond_length_H1_middle'] = self.dist_matrix[H1, middle_idx]
        res['bond_length_middle_H2'] = self.dist_matrix[middle_idx, H2]
        res['angle_cos'] = self.calc_cos(H1, middle_idx, H2)
        return res
    
    def get_3JHx_features(self, H, x):
        res = {}
        path_nodes = self.graph_shortest_path[H][x]
        middle_idx1, middle_idx2 = path_nodes[1], path_nodes[2]
        res.update(self.get_neighbor_features(middle_idx1, 'middle1'))
        res.update(self.get_neighbor_features(middle_idx2, 'middle2'))
        res.update(self.get_neighbor_features(x, 'idx1'))
        res['bond_length_H_middle1'] = self.dist_matrix[H, middle_idx1]
        res['bond_length_middle1_middle2'] = self.dist_matrix[middle_idx1, middle_idx2]
        res['bond_length_middle2_x'] = self.dist_matrix[middle_idx2, x]
        res['angle1_cos'] = self.calc_cos(H, middle_idx1, middle_idx2)
        res['angle2_cos'] = self.calc_cos(middle_idx1, middle_idx2, x)
        res['dihedral_angle_cos'] = self.calc_dihedral_angle(H, middle_idx1, middle_idx2, x)
        return res

    def get_3JHH_features(self, H1, H2):
        res = {}
        path_nodes = self.graph_shortest_path[H1][H2]
        middle_idx1, middle_idx2 = path_nodes[1], path_nodes[2]
        res.update(self.get_neighbor_features(middle_idx1, 'middle1'))
        res.update(self.get_neighbor_features(middle_idx2, 'middle2'))
        res['bond_length_H1_middle1'] = self.dist_matrix[H1, middle_idx1]
        res['bond_length_middle1_middle2'] = self.dist_matrix[middle_idx1, middle_idx2]
        res['bond_length_middle2_H2'] = self.dist_matrix[middle_idx2, H2]
        res['angle1_cos'] = self.calc_cos(H1, middle_idx1, middle_idx2)
        res['angle2_cos'] = self.calc_cos(middle_idx1, middle_idx2, H2)
        res['dihedral_angle_cos'] = self.calc_dihedral_angle(H1, middle_idx1, middle_idx2, H2)
        return res

    def get_coupling_features(self, coupling_type, idx1, idx2):
        common_features = self.get_common_features(idx1, idx2)
        sp_features = self.func_map[coupling_type](idx1, idx2)
        return {**common_features, **sp_features}
        
    def main(self):
        res = {'molecule_features': self.get_molecule_features(), 'couplings_features': defaultdict(dict)}
        couplings = self.get_all_couplings()
        for coupling_type, couplings_lst in couplings.items():
            for idx1, idx2 in couplings_lst:
                res['couplings_features'][coupling_type][(idx1, idx2)] = self.get_coupling_features(coupling_type, idx1, idx2)
        return res

def pool_worker(molecule):
    with open(f'./result/molecule_features/{molecule}.pkl', 'wb') as f:
        pickle.dump(MoleculeFeatures(molecule).main(), f)

def parse_features_dict(molecule_names):
    coupling_types = ['1JHC', '1JHN', '2JHH', '2JHC', '2JHN', '3JHH', '3JHC', '3JHN']
    # molecule features
    features_molecule = {}
    def molecule_worker(molecule):
        tmp = pickle.load(open(f'./result/molecule_features/{molecule}.pkl', 'rb'))
        features_molecule[molecule] = tmp['molecule_features']
    with ThreadPool() as pool:
        pool.map(molecule_worker, molecule_names)
    features_molecule = pd.DataFrame.from_dict(features_molecule, 'index')
    features_molecule.index.set_names('molecule_name', inplace=True)
    features_molecule.reset_index(inplace=True)
    features_molecule.to_csv('./input/features_molecule.csv', index=False)
    del features_molecule

    # coupling features for each type
    def coupling_worker(coupling_type, res_dict, molecule):
        tmp = pickle.load(open(f'./result/molecule_features/{molecule}.pkl', 'rb'))
        for idx1, idx2 in tmp['couplings_features'][coupling_type]:
            res_dict[(molecule, idx1, idx2)] = tmp['couplings_features'][coupling_type][(idx1, idx2)]
    for coupling_type in coupling_types:
        tmp = {}
        with ThreadPool() as pool:
            pool.map(lambda molecule: coupling_worker(coupling_type, tmp, molecule), molecule_names)
        features_tmp = pd.DataFrame.from_dict(tmp, 'index')
        features_tmp.index.set_names(['molecule_name', 'atom_index_0', 'atom_index_1'], inplace=True)
        features_tmp.reset_index(inplace=True)
        features_tmp.to_csv(f'./input/features_{coupling_type}.csv', index=False)
        del tmp, features_tmp

if __name__ == '__main__':
    res = {}
    for x in molecule_names[1000:1100]:
        pool_worker(x)
    parse_features_dict(molecule_names[1000:1100])

