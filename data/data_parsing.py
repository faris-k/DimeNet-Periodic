import numpy as np
from tqdm import tqdm
from matminer.datasets import load_dataset
from pymatgen.core.structure import Structure


datasets = ['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric',
            'matbench_log_kvrh', 'matbench_log_gvrh', 'matbench_perovskites',
            'matbench_mp_gap', 'matbench_mp_is_metal', 'matbench_mp_e_form']
small_datasets = datasets[0:6]
large_datasets = datasets[6:]


def get_neighbors(structure, cutoff):
    neighbors = structure.get_all_neighbors(cutoff)
    return [sorted(nbrs, key=lambda x: x[1]) for nbrs in neighbors]


def cif_parse(dataset):
    df = load_dataset(dataset)
    df = df.values
    all_data = []
    for row in tqdm(df):
        data = {}
        struct = row[0]
        prop = row[1]
        neighbors = get_neighbors(struct, 8)
        for i in neighbors:
            if i != []:
                sites = i
                break

        r = 9
        while ((len(sites) < 80) or (len(sites) < (struct.num_sites * 2))):
            neighbors = get_neighbors(struct, r)
            sites = neighbors[0]
            r += 1

        supercell = Structure.from_sites(sites)
        data['R'] = supercell.cart_coords
        data['Z'] = np.asarray(supercell.atomic_numbers)
        data['N'] = np.asarray(len(data['R']))
        data['Y'] = np.asarray(prop)
        data['formula'] = struct.formula.replace(' ', '')
        all_data.append(data)
    return all_data


for dataset in datasets:
    parsed = cif_parse(dataset)
    np.save(f'{dataset}.npy', parsed)
    print(f'\n{dataset} parsed!\n')
