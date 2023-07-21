import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable, List, Optional

from sklearn.model_selection import StratifiedKFold

import torch

import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, HeteroData

from PreprocessScript.PPI_Preprocess import PPIConstructor
from PreprocessScript.KEGG_Preprocess import PathwayConstructor
from PreprocessScript.Protein_Gene_Preprocess import PGConstructor


class HGDataset(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super(HGDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'RawData')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "ProcessedData_Homogeneous")

    @property
    def raw_file_names(self) -> List[str]:
        return ['Associations']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def construct_edge(self):
        print("=======================Start constructing edge_index===============================\n")
        # kegg_path = "RawData\\Associations\\KEGG"
        kegg_path = "RawData/Associations/KEGG"
        kegg_constructor = PathwayConstructor(file_path=kegg_path)
        # edge_index_pathway ==> torch.Size([2, 608064]); unique_gene_id ==> (8200, 2)
        edge_index_pathway, unique_gene_id = kegg_constructor.construct_pathway()

        # ppi_path = "RawData\\Associations\\PPI"
        ppi_path = "RawData/Associations/PPI"
        ppi_constructor = PPIConstructor(file_path=ppi_path)
        # edge_index_ppi ==> torch.Size([2, 505968]) ; unique_protein_id ==> (16814, 2)
        edge_index_ppi, unique_protein_id = ppi_constructor.construct_homogeneous_ppi()

        return edge_index_ppi, edge_index_pathway, unique_protein_id, unique_gene_id

    def construct_gene_feature(self, unique_gene_id):
        """
        Note: 1.Some genes do not have omic feature, consider remove these genes
        :param unique_gene_id:
        :return:
        """
        print("=======================Start constructing gene feature===============================\n")
        # omic_path = "OmicFeatures\\biological_features.csv"
        omic_path = "OmicFeatures/biological_features.csv"
        omic_feature = pd.read_csv(os.path.join(self.raw_dir, omic_path), sep='\t', index_col=0)
        genes_keep = unique_gene_id['GeneSymbol'].values.tolist()
        omic_feature = omic_feature.loc[genes_keep]
        cancers = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD',
                   'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
        omics_temp = [omic_feature[omic_feature.columns[omic_feature.columns.str.contains(
            cancer)]] for cancer in cancers]
        omics_data = pd.concat(omics_temp, axis=1)
        # torch.Size([6585, 48])
        omics_vector = torch.from_numpy(omics_data.values)
        omics_vector = omics_vector.to(torch.float32)


        """
        # Test: Check whether each gene in unique_gene has omic feature
        unique_genes = unique_gene_id['GeneSymbol'].values
        omic_gene = omic_feature.index.values
        count = 0
        pbar = tqdm(unique_genes)
        for u_g in pbar:
            flag = 0
            for o_g in omic_gene:
                if u_g == o_g:
                    flag = 1
                    break
            if not flag:
                print("Can not find omic feature for {}".format(u_g))
                count += 1
        """
        return omics_vector

    def construct_protein_feature(self, unique_protein_id):

        print("=======================Start constructing protein feature===============================\n")
        # omic_path = "OmicFeatures\\biological_features.csv"
        omic_path = "OmicFeatures/biological_features.csv"
        omic_feature = pd.read_csv(os.path.join(self.raw_dir, omic_path), sep='\t', index_col=0)
        genes_keep = unique_protein_id['GeneSymbol'].values.tolist()
        omic_feature = omic_feature.loc[genes_keep]
        cancers = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD',
                   'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
        omics_temp = [omic_feature[omic_feature.columns[omic_feature.columns.str.contains(
            cancer)]] for cancer in cancers]
        omics_data = pd.concat(omics_temp, axis=1)
        omics_vector = torch.from_numpy(omics_data.values)
        omics_vector = omics_vector.to(torch.float32)

        return omics_vector

    def construct_label(self, unique_gene_id, unique_protein_id):

        label_path = "Label/pancan_genelist_for_train.tsv"
        label_df = pd.read_csv(os.path.join(self.raw_dir, label_path), sep='\t')
        # Merge the gene nodes with labels
        genes_match = pd.merge(left=unique_gene_id['GeneSymbol'], right=label_df, how='left', left_on='GeneSymbol', right_on='Hugosymbol')
        idx_list = np.array(genes_match[~genes_match['Label'].isnull()].index)
        print(f'The match number of gene with annotation: {len(idx_list)}')
        label_list = np.array(genes_match['Label'].loc[idx_list])
        unique, counts = np.unique(label_list, return_counts=True)
        print('The label distribution:', dict(zip(unique, counts)))

        # Merge the gene nodes with labels -- protein
        ptoteins_match = pd.merge(left=unique_protein_id['GeneSymbol'], right=label_df, how='left', left_on='GeneSymbol', right_on='Hugosymbol')
        protein_idx_list = np.array(ptoteins_match[~ptoteins_match['Label'].isnull()].index)
        print(f'The match number of gene with annotation: {len(protein_idx_list)}')
        protein_label_list = np.array(ptoteins_match['Label'].loc[protein_idx_list])
        unique, counts = np.unique(protein_label_list, return_counts=True)
        print('The label distribution:', dict(zip(unique, counts)))

        return idx_list, label_list, protein_idx_list, protein_label_list

    def process(self):
        edge_index_ppi, edge_index_pathway, unique_protein_id, unique_gene_id = self.construct_edge()
        omics_vector_gene = self.construct_gene_feature(unique_gene_id=unique_gene_id)
        protein_feature_vector = self.construct_protein_feature(unique_protein_id=unique_protein_id)
        idx_list, label_list, protein_idx_list, protein_label_list = self.construct_label(unique_gene_id, unique_protein_id)

        data = HeteroData()
        data['gene'].x = omics_vector_gene
        data['protein'].x = protein_feature_vector
        data['gene'].y = torch.from_numpy(label_list)
        data['gene'].label_index = torch.from_numpy(idx_list)
        data['protein'].y = torch.from_numpy(protein_label_list)
        data['protein'].label_index = torch.from_numpy(protein_idx_list)
        data['protein', 'to', 'protein'].edge_index = edge_index_ppi
        data['gene', 'to', 'gene'].edge_index = edge_index_pathway
        data = T.ToUndirected()(data)

        if self.pre_filter is not None:
            data = self.pre_filter(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
        print("Finished to construct data !!! ")


# HGDataset(root="/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")