import os
import torch
import pandas as pd
from torch_geometric.utils import from_scipy_sparse_matrix, to_dense_adj
import scipy.sparse as sp

# os.chdir("E:\\xiong\\Bioinfomatics\\DriverGenes\\WorkingSpace\\MyDriveGenes")
os.chdir("/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")


class PGConstructor():
    def __init__(self, file_path):
        super(PGConstructor, self).__init__()
        self.file_path = file_path

    def construct_p_g(self, unique_protein_id, unique_gene_id):
        print("Reading Ensembl Mapping, please wait ... ")
        ensembl_mapping = pd.read_table(os.path.join(self.file_path, "EnsemblMapping.txt"))
        columns_keep = ['Protein stable ID', 'Gene name']
        ensembl_mapping = ensembl_mapping[columns_keep]
        ensembl_mapping = ensembl_mapping.dropna()
        ensembl_mapping = ensembl_mapping.sort_values(by='Protein stable ID')
        ensembl_mapping.reset_index(drop=True, inplace=True)
        # Merge unique_proteins and genes with Ensembl mapping to generate protein-gene associations
        temp = pd.merge(left=ensembl_mapping, right=unique_protein_id, how='inner',
                        left_on='Protein stable ID', right_on='ProteinID')
        temp = pd.merge(left=temp, right=unique_gene_id, how='inner',
                        left_on='Gene name', right_on='GeneSymbol')

        edge_index_p_g = torch.stack([torch.from_numpy(temp['MappedProteinID'].values),
                                      torch.from_numpy(temp['MappedGeneID'].values)], dim=0)

        edge_index_p_g_df = pd.DataFrame(edge_index_p_g)
        edge_index_p_g_df.to_csv("Explain/edge_index_p_g.csv", header=False, index=False)
        temp.to_csv("Explain/g_p_temp.csv")


        # adj_p_g = to_dense_adj(edge_index_p_g)
        # adj_p_g_df = pd.DataFrame(adj_p_g[0], index=unique_protein_id['ProteinID'], columns=unique_protein_id['ProteinID'])
        # adj_p_g_df = pd.DataFrame(adj_p_g[0], index=temp['Protein stable ID'], columns=temp['Gene name'])
        # adj_protein_df.to_csv('Explain/protein-protein.csv')
        return edge_index_p_g
