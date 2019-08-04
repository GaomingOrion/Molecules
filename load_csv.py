import pandas as pd

def load_csv(coupling_type, load_test=True):
    train = pd.read_csv('./input/train.csv', index_col=0)
    train = train[train['type'] == coupling_type]
    molecule_features = pd.read_csv('./input/features_molecule.csv')
    coupling_features = pd.read_csv(f'./input/features_{coupling_type}.csv')
    train = train.merge(molecule_features, how='left', on='molecule_name').merge(
        coupling_features, how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1'])
    if load_test:
        test = pd.read_csv('./input/train.csv', index_col=0)
        test = test[test['type'] == coupling_type]
        test = test.merge(molecule_features, how='left', on='molecule_name').merge(
        coupling_features, how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1'])
        return train, test
    return train

if __name__ == '__main__':
    train, test = load_csv('1JHC')