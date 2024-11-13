import itertools
import random
from collections import Counter
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from ase.data import chemical_symbols
from sklearn.linear_model import Ridge

import sevenn._keys as KEY
import sevenn.util as util

# building a model with the most basic libaries ofc ............ am i stupid? 

class AtomGraphDataset:
    """
    Deprecated

    class representing dataset of AtomGraphData
    the dataset is handled as dict, {label: data}
    if given data is List, it stores data as {KEY_DEFAULT: data}

    cutoff is for metadata of the graphs not used for some calc
    Every data expected to have one unique cutoff
    No validity or check of the condition is done inside the object

    attribute:
        dataset (Dict[str, List]): key is data label(str), value is list of data
        user_labels (List[str]): list of user labels same as dataset.keys()
        meta (Dict, Optional): metadata of dataset
    for now, metadata 'might' have following keys:
        KEY.CUTOFF (float), KEY.CHEMICAL_SPECIES (Dict)
    """

    DATA_KEY_X = (
        KEY.NODE_FEATURE
    )  # atomic_number > one_hot_idx > one_hot_vector
    DATA_KEY_ENERGY = KEY.ENERGY
    DATA_KEY_FORCE = KEY.FORCE
    KEY_DEFAULT = KEY.LABEL_NONE

    def __init__(
        self,
        dataset: Union[Dict[str, List], List],
        cutoff: float,
        metadata: Optional[Dict] = None,
        x_is_one_hot_idx: bool = False,
    ):
        """
        Default constructor of AtomGraphDataset
        Args:
            dataset (Union[Dict[str, List], List]: dataset as dict or pure list
            metadata (Dict, Optional): metadata of data
            cutoff (float): cutoff radius of graphs inside the dataset
            x_is_one_hot_idx (bool): if True, x is one_hot_idx, else 'Z'

        'x' (node feature) of dataset can have 3 states, atomic_numbers,
        one_hot_idx, or one_hot_vector.

        atomic_numbers is general but cannot directly used for input
        one_hot_idx is can be input of the model but requires 'type_map'
        """
        self.cutoff = cutoff
        self.x_is_one_hot_idx = x_is_one_hot_idx
        if metadata is None:
            metadata = {KEY.CUTOFF: cutoff} # TODO: umm sir? 
        self.meta = metadata
        if type(dataset) is list:
            self.dataset = {self.KEY_DEFAULT: dataset}
        else:
            self.dataset = dataset
        self.user_labels = list(self.dataset.keys())
        # group_by_key here? or not?

    def rewrite_labels_to_data(self):
        #umm userlabels? .. looks important for training datasets AAW ..
        
        """
        Based on self.dataset dict's keys
        write data[KEY.USER_LABEL] to correspond to dict's keys
        Most of times, it is already correctly written
        But required to rewrite if someone rearrange dataset by their own way
        """
        for label, data_list in self.dataset.items():
            for data in data_list:
                data[KEY.USER_LABEL] = label

    def group_by_key(self, data_key: str = KEY.USER_LABEL):
        # redundancy?
        """
        group dataset list by given key and save it as dict
        and change in-place
        Args:
            data_key (str): data key to group by

        original use is USER_LABEL, but it can be used for other keys
        if someone established it from data[KEY.INFO]
        """
        data_list = self.to_list()
        self.dataset = {}
        for datum in data_list:
            key = datum[data_key]
            if key not in self.dataset:
                self.dataset[key] = []
            self.dataset[key].append(datum)
        self.user_labels = list(self.dataset.keys())

    def separate_info(self, data_key: str = KEY.INFO):
        # is it?
        """
        Separate info from data and save it as list of dict
        to make it compatible with torch_geometric and later training
        """
        data_list = self.to_list()
        info_list = []
        for datum in data_list:
            if data_key in datum is False:
                continue
            info_list.append(datum[data_key])
            del datum[data_key]  # It does change the self.dataset
            datum[data_key] = len(info_list) - 1
        self.info_list = info_list

        return (data_list, info_list)

    def get_species(self):
        # why not use chempy? to avoid parsing error .. bet
        """
        You can also use get_natoms and extract keys from there instead of this
        (And it is more efficient)
        get chemical species of dataset
        return list of SORTED chemical species (as str)
        """
        if hasattr(self, 'type_map'):
            natoms = self.get_natoms(self.type_map)
        else:
            natoms = self.get_natoms()
        species = set()
        for natom_dct in natoms.values():
            species.update(natom_dct.keys())
        species = sorted(list(species))
        return species

    def len(self):
        if (
            len(self.dataset.keys()) == 1
            and list(self.dataset.keys())[0] == AtomGraphDataset.KEY_DEFAULT
        ):
            return len(self.dataset[AtomGraphDataset.KEY_DEFAULT])
        else:
            return {k: len(v) for k, v in self.dataset.items()}

    def get(self, idx: int, key: Optional[str] = None):
        if key is None:
            key = self.KEY_DEFAULT
        return self.dataset[key][idx]

    def items(self):
        return self.dataset.items()
    # ofc itssa dataset

    def to_dict(self):
        dct_dataset = {}
        for label, data_list in self.dataset.items():
            dct_dataset[label] = [datum.to_dict() for datum in data_list]
        self.dataset = dct_dataset
        return self
    # save dict - vocab for chemVAE - var name as map oOo
    def x_to_one_hot_idx(self, type_map: Dict[int, int]):
        """
        type_map is dict of {atomic_number: one_hot_idx}
        after this process, the dataset has dependency on type_map !~DEPENDENCY~!
        or chemical species user want to consider 
        """
        assert self.x_is_one_hot_idx is False
    
        for data_list in self.dataset.values():
            for datum in data_list:
                datum[self.DATA_KEY_X] = torch.LongTensor(
                    [type_map[z.item()] for z in datum[self.DATA_KEY_X]]
                )
        self.type_map = type_map
        self.x_is_one_hot_idx = True
    
    # origin for the terminology? start lexi?
    
    def toggle_requires_grad_of_data(
        self, key: str, requires_grad_value: bool
    ):
        """
        set requires_grad of specific key of data(pos, edge_vec, ...)
        """
        for data_list in self.dataset.values():
            for datum in data_list:
                datum[key].requires_grad_(requires_grad_value)

    def divide_dataset(
        self,
        ratio: float,
        constant_ratio_btw_labels: bool = True,
        ignore_test: bool = True
    ):
        """
        divide dataset into 1-2*ratio : ratio : ratio
        return divided AtomGraphDataset
        returned value lost its dict key and became {KEY_DEFAULT: datalist}
        but KEY.USER_LABEL of each data is preserved
        """
        # train test splitting? seems like ..
        # delulu or nesting a function inside a function
        # yoy  #TODO
        
        def divide(ratio: float, data_list: List, ignore_test=True):
            if ratio > 0.5:
                raise ValueError('Ratio must not exceed 0.5')
            data_len = len(data_list)
            random.shuffle(data_list)
            n_validation = int(data_len * ratio)
            if n_validation == 0:
                raise ValueError(
                    '# of validation set is 0, increase your dataset'
                )

            if ignore_test:
                test_list = []
                n_train = data_len - n_validation
                train_list = data_list[0:n_train]
                valid_list = data_list[n_train:]
            else:
                n_train = data_len - 2 * n_validation
                train_list = data_list[0:n_train]
                valid_list = data_list[n_train : n_train + n_validation]
                test_list = data_list[n_train + n_validation : data_len]
            return train_list, valid_list, test_list
        # just evolved to a caveman from a monkey

        lists = ([], [], [])  # train, valid, test
        if constant_ratio_btw_labels:
            for data_list in self.dataset.values():
                for store, divided in zip(lists, divide(ratio, data_list)):
                    store.extend(divided)
        else:
            lists = divide(ratio, self.to_list())

        dbs = tuple(
            AtomGraphDataset(data, self.cutoff, self.meta) for data in lists
        )
        for db in dbs:
            db.group_by_key()
        return dbs

    def to_list(self):
        return list(itertools.chain(*self.dataset.values()))

    def get_natoms(self, type_map: Optional[Dict[int, int]] = None):
        """
        if x_is_one_hot_idx, type_map is required
        type_map: Z->one_hot_index(node_feature)
        return Dict{label: {symbol, natom}]}
        """
        # what a brain twister .. i'm not built for this
        # am i delusional or 
        # assert self.x_is_one_hot_idx is not True or type_map is not None is it?
        # is this basic math? or am i just dumb?
        
        assert not (self.x_is_one_hot_idx is True and type_map is None) 
        natoms = {}
        for label, data in self.dataset.items():
            natoms[label] = Counter()
            # initiate a new counter everytime? label ... right .... a f wait howduzthe dataset look like at the first place
            for datum in data:
                if self.x_is_one_hot_idx and type_map is not None:
                    Zs = util.onehot_to_chem(datum[self.DATA_KEY_X], type_map)
                    # why named Zs .................... what's a Z
                else:
                    Zs = [
                        chemical_symbols[z]
                        for z in datum[self.DATA_KEY_X].tolist()
                    ]
                cnt = Counter(Zs)
                natoms[label] += cnt
            natoms[label] = dict(natoms[label]) # what happens when you dict a counter nvmjsutfoundout
        return natoms

    # per ..? mean of ..? 
    def get_per_atom_mean(self, key: str, key_num_atoms: str = KEY.NUM_ATOMS):
        """
        return per_atom mean of given data key 
        """
        eng_list = torch.Tensor(
            [x[key] / x[key_num_atoms] for x in self.to_list()]
        )
        return float(torch.mean(eng_list))
    # eng for energy .. 
    # alias 
    def get_per_atom_energy_mean(self):
        """
        alias for get_per_atom_mean(KEY.ENERGY)
        """
        return self.get_per_atom_mean(self.DATA_KEY_ENERGY)

    # plshelpimunderdawater
    def get_species_ref_energy_by_linear_comb(self, num_chem_species: int):
        """
        Total energy as y, composition as c_i,
        solve linear regression of y = c_i*X
        sklearn LinearRegression as solver

        x should be one-hot-indexed
        give num_chem_species if possible
        """
        assert self.x_is_one_hot_idx is True
        data_list = self.to_list()

        # torch.bincount Counter?
        c = torch.zeros((len(data_list), num_chem_species))
        for idx, datum in enumerate(data_list):
            c[idx] = torch.bincount(
                datum[self.DATA_KEY_X], minlength=num_chem_species
            )
        y = torch.Tensor([x[self.DATA_KEY_ENERGY] for x in data_list])
        c = c.numpy()
        y = y.numpy()

        # wdym tweak imtheonewhostweakinghere
        # tweak to fine tune training from many-element to small element
        zero_indices = np.all(c == 0, axis=0)
        c_reduced = c[:, ~zero_indices]
        full_coeff = np.zeros(num_chem_species)
        coef_reduced = (
            Ridge(alpha=0.1, fit_intercept=False).fit(c_reduced, y).coef_
        )
        # but why is it Ridge tho
        full_coeff[~zero_indices] = coef_reduced
        # prune ..
        return full_coeff

    def get_force_rms(self):
        force_list = []
        for x in self.to_list():
            force_list.extend(
                x[self.DATA_KEY_FORCE]
                .reshape(
                    -1,
                )
                .tolist()
            )
        force_list = torch.Tensor(force_list)
        return float(torch.sqrt(torch.mean(torch.pow(force_list, 2))))
        
    # torch.mean/pow/sqrt on torch.Tensor
    
    def get_species_wise_force_rms(self, num_chem_species: int):
        """
        Return force rms for each species
        Averaged by each components (x, y, z)
        """
        assert self.x_is_one_hot_idx is True
        data_list = self.to_list()

        # atomsx?

        atomx = torch.concat([d[self.DATA_KEY_X] for d in data_list])
        force = torch.concat([d[self.DATA_KEY_FORCE] for d in data_list])
        # torch.repeat_interleave(x,int)
        index = atomx.repeat_interleave(3, 0).reshape(force.shape)
        rms = torch.zeros(
            (num_chem_species, 3),
            dtype=force.dtype,
            device=force.device
        )
        
        # is torch his first language?
        rms.scatter_reduce_(
            0, index, force.square(),
            reduce='mean', include_self=False
        )
        # why is it called reduction tho
        # k.................
        return torch.sqrt(rms.mean(dim=1))

    def get_avg_num_neigh(self):
        n_neigh = []
        for _, data_list in self.dataset.items():
            for data in data_list:
                n_neigh.extend(
                    np.unique(data[KEY.EDGE_IDX][0], return_counts=True)[1]
                )

        avg_num_neigh = np.average(n_neigh)
        return avg_num_neigh # basically count the number of neighbors .. right? and ..
    # right average the list ...

    def get_statistics(self, key: str):
        """
        return dict of statistics of given key (energy, force, stress)
        key of dict is its label and _total for total statistics
        value of dict is dict of statistics (mean, std, median, max, min)
        """
# statistics~ statistics
        def _get_statistic_dict(tensor_list):
            data_list = torch.cat(
                [
                    tensor.reshape(
                        -1,
                    )
                    for tensor in tensor_list
                ]
            )
            return {
                'mean': float(torch.mean(data_list)),
                'std': float(torch.std(data_list)),
                'median': float(torch.median(data_list)),
                'max': float(torch.max(data_list)),
                'min': float(torch.min(data_list)),
            }
        # regardless of the input i see ..

        # is it a convention? adding a _ to the ... nesting function?
        # what's so good about nesting functions
        res = {} # still doonno what the dataset looks like
        for label, values in self.dataset.items():
            # flatten list of torch.Tensor (values)
            tensor_list = [x[key] for x in values]
            res[label] = _get_statistic_dict(tensor_list)
        tensor_list = [x[key] for x in self.to_list()]
        res['Total'] = _get_statistic_dict(tensor_list)
        return res

    def augment(self, dataset, validator: Optional[Callable] = None):
        """check meta compatibility here
        dataset(AtomGraphDataset): data to augment
        validator(Callable, Optional): function(self, dataset) -> bool

        if validator is None, by default it checks
        whether cutoff & chemical_species are same before augment
        
        check consistent data type, float, double, long integer etc
        """
        # exactly what are we augmenting here?
        def default_validator(db1, db2):
            cut_consis = db1.cutoff == db2.cutoff # consis for consistent?
            # compare unordered lists
            # unordered? wdym unordered .. oh right ....
            # and itssa database..? amiright?actually no ..right itssa dataset .... i think
            x_is_not_onehot = (not db1.x_is_one_hot_idx) and (
                not db2.x_is_one_hot_idx
            )
            return cut_consis and x_is_not_onehot
        # k ....
        if validator is None:
            validator = default_validator # oh
        if not validator(self, dataset): # oh so it's the atomgraphdataset vs a new ... new .... i dunno 
            raise ValueError('given datasets are not compatible check cutoffs')
        for key, val in dataset.items():
            if key in self.dataset:
                self.dataset[key].extend(val) # congrats you're part of the team
            else:
                self.dataset.update({key: val}) # a pioneer ..
        self.user_labels = list(self.dataset.keys()) # a cookie .. nice 

    def unify_dtypes(
        self,
        float_dtype: torch.dtype = torch.float32,
        int_dtype: torch.dtype = torch.int64
    ):
        # no joke. this is what we call from scratch
        data_list = self.to_list()
        for datum in data_list:
            for k, v in list(datum.items()):
                datum[k] = util.dtype_correct(v, float_dtype, int_dtype)

    def delete_data_key(self, key: str):
        for data in self.to_list():
            del data[key]

    # TODO: this by_label is not straightforward
    # thanks for the reminder .. 
    def save(self, path: str, by_label: bool = False):
        if by_label:
            for label, data in self.dataset.items():
                torch.save(
                    AtomGraphDataset(
                        {label: data}, self.cutoff, metadata=self.meta
                    ),
                    f'{path}/{label}.sevenn_data',
                )
        else:
            if path.endswith('.sevenn_data') is False:
                path += '.sevenn_data'
            torch.save(self, path)
